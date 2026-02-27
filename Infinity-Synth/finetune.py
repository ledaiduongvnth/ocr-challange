import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from jiwer import cer, wer
from peft import PeftModel
from PIL import Image
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    Trainer,
    TrainingArguments,
)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
ASSISTANT_START_PATTERN = [151645, 198, 151644, 77091, 198]
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "26_02_1199"


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = "lightonai/LightOnOCR-2-1B"
    finevision_subset: str = "bill_2"
    base_dir: str = os.getenv("OCR_TRAIN_BASE_DIR", str(DEFAULT_BASE_DIR))
    image_subdir: str = "images"
    label_subdir: str = "label"
    label_extension: str = os.getenv("OCR_TRAIN_LABEL_EXT", ".json")
    save_model_dir: str = "./model"
    max_length: int = int(os.getenv("OCR_MAX_LENGTH", "3072"))
    longest_edge: int = 700
    strict_no_truncation: bool = os.getenv("OCR_STRICT_NO_TRUNCATION", "1") == "1"
    eval_max_new_tokens: int = int(os.getenv("OCR_EVAL_MAX_NEW_TOKENS", "2048"))
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    eval_samples: int = 100
    eval_batch_size: int = 4
    train_subset_size: int = 480
    val_subset_size: int = 100
    num_train_epochs: int = 50
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 6
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-5
    logging_steps: int = 50
    eval_steps: int = 50
    save_steps: int = 500
    warmup_steps: int = 10
    output_dir_prefix: str = "LightOnOCR-2-ft-"
    loss_plot_dir: str = "plot_rs"
    loss_plot_name: str = "loss_curve.png"

    @property
    def output_dir(self) -> str:
        return f"{self.output_dir_prefix}{self.finevision_subset}"

    @property
    def image_dir(self) -> Path:
        return Path(self.base_dir) / self.image_subdir

    @property
    def label_dir(self) -> Path:
        return Path(self.base_dir) / self.label_subdir


class ImageLabelDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, label_extension: str = ".json", ids: Sequence[str] | None = None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_extension = label_extension if label_extension.startswith(".") else f".{label_extension}"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")

        self.id_to_image = self._build_image_index()

        if ids is None:
            self.ids = sorted(self.id_to_image.keys())
            if not self.ids:
                raise ValueError(
                    f"No valid image/{self.label_extension} label pairs found in {self.image_dir}. "
                    f"Supported image types: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)} (PDF is not supported). "
                    f"Expected labels in {self.label_dir}."
                )
        else:
            missing = [sample_id for sample_id in ids if sample_id not in self.id_to_image]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"Missing image/{self.label_extension} label pair for {len(missing)} ids. Example: {preview}"
                )
            self.ids = list(ids)

    def _build_image_index(self) -> Dict[str, Path]:
        id_to_image: Dict[str, Path] = {}

        for file_path in sorted(self.image_dir.iterdir()):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()
            if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            sample_id = file_path.stem
            label_path = self.label_dir / f"{sample_id}{self.label_extension}"
            if not label_path.exists():
                continue

            if sample_id in id_to_image:
                existing = id_to_image[sample_id]
                raise ValueError(
                    f"Duplicate image id '{sample_id}' from {existing.name} and {file_path.name}. "
                    "Keep only one image per id."
                )
            id_to_image[sample_id] = file_path

        return id_to_image

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, List[Dict[str, str]]]:
        sample_id = self.ids[idx]
        image_path = self.id_to_image[sample_id]
        label_path = self.label_dir / f"{sample_id}{self.label_extension}"

        image = Image.open(image_path).convert("RGB")
        label_text = self._load_label_as_text(label_path)

        return {"images": [image], "texts": [{"assistant": label_text}]}

    @staticmethod
    def _load_label_as_text(label_path: Path) -> str:
        try:
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON label file: {label_path} ({error})") from error

        # Canonical compact JSON target to reduce truncation risk at fixed max_length.
        return json.dumps(label_data, ensure_ascii=False, separators=(",", ":"))


def split_ids(ids: Sequence[str], train_ratio: float, val_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    total = len(ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return list(ids[:train_end]), list(ids[train_end:val_end]), list(ids[val_end:])


def subset_or_full(dataset: Dataset, max_size: int) -> Dataset:
    if max_size <= 0 or len(dataset) <= max_size:
        return dataset
    return Subset(dataset, range(max_size))


def load_processor_and_model(config: TrainConfig, device: torch.device) -> Tuple[LightOnOcrProcessor, LightOnOcrForConditionalGeneration]:
    processor = LightOnOcrProcessor.from_pretrained(config.model_id)
    processor.tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype}
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"

    model = LightOnOcrForConditionalGeneration.from_pretrained(config.model_id, **model_kwargs).to(device)

    for param in model.model.language_model.parameters():
        param.requires_grad = False

    print(f"Using device: {device}")
    print(f"Language model frozen: {param.requires_grad}")
    return processor, model


def evaluate_model(
    model: LightOnOcrForConditionalGeneration,
    processor: LightOnOcrProcessor,
    dataset: Dataset,
    device: torch.device,
    max_length: int,
    longest_edge: int,
    max_new_tokens: int,
    num_samples: int = 50,
    batch_size: int = 8,
    description: str = "Model",
) -> Dict[str, float]:
    sample_count = min(num_samples, len(dataset))
    if sample_count == 0:
        raise ValueError(f"{description} dataset is empty.")

    model.eval()
    predictions: List[str] = []
    ground_truths: List[str] = []

    print(f"\nEvaluating {description} on {sample_count} samples...")

    for start_idx in tqdm(range(0, sample_count, batch_size), desc=f"{description} eval"):
        end_idx = min(start_idx + batch_size, sample_count)
        batch_samples = [dataset[i] for i in range(start_idx, end_idx)]

        batch_images = [[sample["images"][0]] for sample in batch_samples]
        batch_ground_truths = [sample["texts"][0]["assistant"].strip() for sample in batch_samples]

        messages = [{"role": "user", "content": [{"type": "image"}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts = [prompt] * len(batch_images)

        inputs = processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            size={"longest_edge": longest_edge},
        ).to(device)

        if "pixel_values" in inputs and device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]

        batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        batch_predictions = [prediction.strip() for prediction in batch_predictions]

        predictions.extend(batch_predictions)
        ground_truths.extend(batch_ground_truths)

    cer_score = cer(ground_truths, predictions) * 100
    wer_score = wer(ground_truths, predictions) * 100
    perfect_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)

    print(f"CER: {cer_score:.2f}% | WER: {wer_score:.2f}% | Perfect: {perfect_matches}/{sample_count}")
    for i in range(min(3, len(predictions))):
        status = "MATCH" if predictions[i] == ground_truths[i] else "DIFF"
        print(f"[{status}] Sample {i + 1}: '{predictions[i]}' vs '{ground_truths[i]}'")

    return {"cer": cer_score, "wer": wer_score, "perfect_matches": perfect_matches}


def build_collate_fn(
    processor: LightOnOcrProcessor,
    max_length: int,
    longest_edge: int,
    use_bf16: bool,
    strict_no_truncation: bool,
):
    def collate_fn(examples):
        batch_messages = []
        batch_images = []

        for example in examples:
            example_images = example["images"]
            example_texts = example["texts"]

            assert len(example_images) == 1, f"Expected 1 image per sample, got {len(example_images)}"
            assert len(example_texts) == 1, f"Expected 1 text per sample, got {len(example_texts)}"

            image = example_images[0].convert("RGB")
            batch_images.append(image)

            assistant_text = example_texts[0].get("assistant", "").strip()
            messages = [
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ]
            batch_messages.append(messages)

        if not batch_images:
            return None

        texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch_messages
        ]

        token_lengths = [
            len(input_ids)
            for input_ids in processor.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        ]
        overlength = [(idx, length) for idx, length in enumerate(token_lengths) if length > max_length]
        if overlength and strict_no_truncation:
            preview = ", ".join(f"{idx}:{length}" for idx, length in overlength[:5])
            raise ValueError(
                f"Detected {len(overlength)} overlength sample(s) in current batch for max_length={max_length}. "
                f"Example batch_index:token_len -> {preview}. "
                "Increase OCR_MAX_LENGTH or shorten labels."
            )

        inputs = processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=not strict_no_truncation,
            max_length=max_length,
            size={"longest_edge": longest_edge},
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id

        for i in range(len(labels)):
            full_ids = inputs["input_ids"][i].tolist()
            assistant_content_start = None

            for idx in range(len(full_ids) - len(ASSISTANT_START_PATTERN) + 1):
                if full_ids[idx : idx + len(ASSISTANT_START_PATTERN)] == ASSISTANT_START_PATTERN:
                    assistant_content_start = idx + len(ASSISTANT_START_PATTERN)
                    break

            labels[i, :] = -100
            if assistant_content_start is None:
                print(f"Warning: Could not find assistant marker in sample {i}")
                continue

            for idx in range(assistant_content_start, len(full_ids)):
                if full_ids[idx] == pad_token_id:
                    break
                labels[i, idx] = inputs["input_ids"][i, idx]

            labels[i, inputs["input_ids"][i] == pad_token_id] = -100

        inputs["labels"] = labels
        if use_bf16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    return collate_fn


def build_training_args(config: TrainConfig, use_bf16: bool) -> TrainingArguments:
    strategy_arg = (
        "evaluation_strategy"
        if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "eval_strategy"
    )

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.0,
        logging_steps=config.logging_steps,
        **{strategy_arg: "steps"},
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if use_bf16 else "adamw_torch",
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
    )


def plot_loss_curve(log_history: Sequence[Dict[str, float]], save_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping loss plot.")
        return False

    train_steps: List[float] = []
    train_losses: List[float] = []
    eval_steps: List[float] = []
    eval_losses: List[float] = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    if not train_losses and not eval_losses:
        print("No loss values found in trainer log history. Skipping loss plot.")
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, marker="o", linewidth=2, label="Training Loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss plot saved to: {save_path}")
    return True


def print_comparison(base_results: Dict[str, float], finetuned_results: Dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print("COMPARISON")
    print(f"{'Metric':<20} {'Base':<12} {'Finetuned':<12} {'Change':<12}")
    print("-" * 56)
    print(
        f"{'CER (%)':<20} {base_results['cer']:<12.2f} {finetuned_results['cer']:<12.2f} "
        f"{base_results['cer'] - finetuned_results['cer']:+.2f}"
    )
    print(
        f"{'WER (%)':<20} {base_results['wer']:<12.2f} {finetuned_results['wer']:<12.2f} "
        f"{base_results['wer'] - finetuned_results['wer']:+.2f}"
    )
    print(
        f"{'Perfect':<20} {base_results['perfect_matches']:<12} {finetuned_results['perfect_matches']:<12} "
        f"{finetuned_results['perfect_matches'] - base_results['perfect_matches']:+d}"
    )
    print("=" * 80)


def main():
    config = TrainConfig()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"

    processor, model = load_processor_and_model(config, device)

    full_ds = ImageLabelDataset(
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        label_extension=config.label_extension,
    )
    train_ids, val_ids, test_ids = split_ids(full_ds.ids, config.train_ratio, config.val_ratio)
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            "Dataset split produced an empty train/val/test partition. "
            "Use more samples or adjust train_ratio/val_ratio."
        )

    train_ds = ImageLabelDataset(
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        label_extension=config.label_extension,
        ids=train_ids,
    )
    val_ds = ImageLabelDataset(
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        label_extension=config.label_extension,
        ids=val_ids,
    )
    test_ds = ImageLabelDataset(
        image_dir=config.image_dir,
        label_dir=config.label_dir,
        label_extension=config.label_extension,
        ids=test_ids,
    )

    print(f"Dataset size: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Image extensions accepted: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
    print("PDF files are ignored by dataset loading.")

    print("\n" + "=" * 80)
    print("BEFORE TRAINING")
    print("=" * 80)
    if isinstance(model, PeftModel):
        with model.disable_adapter():
            base_results = evaluate_model(
                model=model,
                processor=processor,
                dataset=test_ds,
                device=device,
                max_length=config.max_length,
                longest_edge=config.longest_edge,
                max_new_tokens=config.eval_max_new_tokens,
                num_samples=config.eval_samples,
                batch_size=config.eval_batch_size,
                description="Base",
            )
    else:
        base_results = evaluate_model(
            model=model,
            processor=processor,
            dataset=test_ds,
            device=device,
            max_length=config.max_length,
            longest_edge=config.longest_edge,
            max_new_tokens=config.eval_max_new_tokens,
            num_samples=config.eval_samples,
            batch_size=config.eval_batch_size,
            description="Base",
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    collate_fn = build_collate_fn(
        processor=processor,
        max_length=config.max_length,
        longest_edge=config.longest_edge,
        use_bf16=use_bf16,
        strict_no_truncation=config.strict_no_truncation,
    )

    train_ds_for_fit = subset_or_full(train_ds, config.train_subset_size)
    val_ds_for_fit = subset_or_full(val_ds, config.val_subset_size)
    training_args = build_training_args(config, use_bf16=use_bf16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_for_fit,
        eval_dataset=val_ds_for_fit,
        data_collator=collate_fn,
    )

    print(f"Output directory: {config.output_dir}")
    print(f"Max input length: {config.max_length}")
    print(f"Strict no truncation: {config.strict_no_truncation}")
    print(
        "Effective batch size: "
        f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print("Starting training...")
    print(f"Training samples used: {len(train_ds_for_fit)}")
    print(f"Validation samples used: {len(val_ds_for_fit)}")

    trainer.train()

    plot_path = Path(config.loss_plot_dir) / config.loss_plot_name
    plot_loss_curve(trainer.state.log_history, plot_path)

    print("\n" + "=" * 80)
    print("AFTER TRAINING")
    finetuned_results = evaluate_model(
        model=model,
        processor=processor,
        dataset=test_ds,
        device=device,
        max_length=config.max_length,
        longest_edge=config.longest_edge,
        max_new_tokens=config.eval_max_new_tokens,
        num_samples=config.eval_samples,
        batch_size=config.eval_batch_size,
        description="Finetuned",
    )

    print_comparison(base_results, finetuned_results)

    trainer.save_model(config.save_model_dir)
    processor.save_pretrained(config.save_model_dir)
    print(f"Model saved to {config.save_model_dir}")


if __name__ == "__main__":
    main()
