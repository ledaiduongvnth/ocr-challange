"""
Finetune Chandra (Qwen3-VL) - Image -> JSON.

Load thang anh tu disk (da augment truoc bang augument.py).
Khong co runtime augmentation.

Cach chay:
    pip install transformers accelerate peft bitsandbytes pillow datasets deepspeed

    # Dung data goc hoac data da augment (chi can sua data_dir trong FinetuneConfig)
    # data_dir/
    #   images/   <- anh (.jpg/.png/...)
    #   label/    <- JSON label (cung ten stem voi anh)

    # 2 GPU (khuyen nghi)
    torchrun --nproc_per_node=2 finetune_train_final.py

    # 1 GPU
    python finetune_train_final.py
"""

import io
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


# ============================================================
# SYSTEM PROMPT - day du table handling rules
# ============================================================
SYSTEM_PROMPT = """
You are an OCR assistant. Your task is Extract all information from the document image and return it as a single strictly formatted valid JSON object.
### OUTPUT FORMAT:
- Return ONLY strictly formatted valid JSON object.
- Do NOT output HTML.
- Do NOT wrap in markdown or code blocks.
- Do NOT add explanations.

### CRITICAL JSON FORMATTING
Every item in a dictionary MUST be a valid key-value pair separated by a colon (e.g., `"Key": "Value"`). NEVER output a standalone string without a colon and a value inside an object. 

### STRICT COMMITTEE EXTRACTION GUIDELINES

1. **Document Title:** Always find MAIN DOCUMENT TITLE (tên tài liệu / tên chứng từ) of the WHOLE document if it exists.The title must be in Vietnamese(The title may also contain an English noun) and English translation(if present near the Vietnamese title) is optional. Assign it to the root key `"Title"`. 
   - Note: Landing AI sometimes hides the title inside logo tags (e.g., `<::logo: GIẤY LĨNH TIỀN...::>`). You MUST extract it from there. 
   - HARD RULES (reject candidates that violate any rule):
      + Reject section headings / numbering:
         * title_vi must NOT start with any digit (0-9)
         * title_vi must NOT start with: "A.", "A,", "B.", "B,", "1.", "1,", "I.", "I,", "II.", "II," (case-insensitive)
         * Also reject common section prefixes: "III.", "IV.", "V.", "(1)", "1)", "a)", "-", "•", "*"
      + Reject weird/special characters:
         * Ignore any text containing characters outside Vietnamese letters (including diacritics), spaces, and these punctuation marks only: "-", ",", ".". Title may contains "/", so "/" is allowed.
         * If text contains any of: "_ @ # $ % ^ & * = + < > { } : [ ] ?" then it is NOT a title candidate.
      + Title should be a document name, not a sentence:
         * Must not look like a paragraph (reject if very long or contains many commas)
      + Prefer near to the top-of-page + big font and uppercase:
         * Only consider blocks in the top area of the page (top 25% by position).
         * The best title is usually the block(s) with a big font_size in that top area.
         * If the title spans multiple lines, merge consecutive blocks that are close vertically and have similar font_size.

2. **Noise & Artifact Removal:** - Completely ignore visual noise (watermarks, QR codes, purely visual logos).
   - *Exception:* If any logo or attestation contains actual textual data (like a signature, a stamp stating "ĐÃ CHI TIỀN", or a title), you MUST extract that text.

3. **Sections vs. Root-Level Keys vs. Tables (CRITICAL):**
   - **Floating Key-Value Pairs (Root Level):** If key-value pairs are loosely listed on the page WITHOUT a visual bounding box/square table and WITHOUT a highlighted section name, they MUST remain as flat, root-level keys. Do NOT arbitrarily group them into "Section1".
   - **Form Grids (Sections):** You can ONLY group items into a Section if they are visually enclosed in a drawn square/box OR fall under a clear, highlighted section heading. Group these under their explicit heading. If a valid enclosed box lacks a heading, use `"Section1"`, `"Section2"`.
   - **True Tables:** Only structures with repeated row records under consistent column headers are tables. Name them using the explicit caption above them, or `"Table1"` if unnamed. 
   - **Nesting:** If a table or a free text block appears *inside* a valid form section, nest it as a child of that section.

4. **Table Formatting & Totals:**
   - Format tables as a list of dictionaries. Keys inside the dictionary MUST exactly match the column headers. NEVER hallucinate or invent headers.
   - **Merged Cells:** If a cell is visually merged across multiple columns, duplicate its value into each covered column in the JSON row.
   - **Total Rows (SEPARATE TABLE):** If a standalone "Total" or "Tổng cộng" line appears immediately below a table, DO NOT put it inside the main table. Extract it as a *separate* table (a list containing a single dictionary). Use the exact total label as the root key, and map the values to their corresponding column headers. Example: `"Tổng cộng (Total)": [{"Thành tiền": "71,900,000"}]`.

5. **Signatures & Approvals:** - Extract signature blocks. Use the explicit role/title as the key. 
   - The value MUST include ALL associated text in that block, including printed names, employee IDs (e.g., "YENNT6.NCB"), stamp text, or timestamps. Preserve line breaks (e.g., `"YENNT6.NCB\nNguyễn Thị Yến"`). If no role is found, use `"Signature1"`.

6. **Key-Value & Checkboxes:** - Extract key-value pairs exactly as written. If a value is missing, use an empty string `""`.
   - Represent checkboxes as booleans (e.g., `"Loại tiền (Currency)": {"VND": true, "EUR": false}`).
   - If there are duplicate sibling keys, append `_1`, `_2` to them.

7. **Free Text Aggregation:** - Standalone text blocks (including isolated metadata at the top or bottom of the page like Bank Names, Branch names, or Addresses) MUST be assigned to keys like `"FreeText1"`, `"FreeText2"`. DO NOT use long text blocks as keys without values.
   - Consecutive blocks of free text must be merged into a single `FreeText` string. Preserve line breaks using `\n`. 

8. **BLANK PAGE HANDLING (CRITICAL):**
   - **Blank Page:** If you encounter a completely blank page or a page containing only visual noise/page numbers, ignore it entirely. Do not let it interrupt the extraction. Treat the pages before and after the blank page as perfectly continuous.
   """
USER_PROMPT = (
    "OCR this document image into structured JSON. "
    "Apply all STRICT COMMITTEE EXTRACTION GUIDELINES strictly. "
    "Return ONLY valid JSON."
)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class FinetuneConfig:
    # Dataset
    data_dir: str = "/root/khaint02/augument_multi"
    images_subdir: str = "images"
    labels_subdir: str = "labels"

    # Model
    model_name: str = "/root/khaint02/model_goc"
    output_dir: str = "/root/khaint02/chandra_lan_cuoi_win_nao"

    # Training
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_seq_length: int = 6192
    save_steps: int = 175
    logging_steps: int = 10
    # eval_split: float = 0.05

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0005

    # Save every N epochs (0 = disable)
    save_every_n_epochs: int = 3

    # LoRA - rank cao cho bang phuc tap
    use_lora: bool = True
    lora_r: int = 14
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization - False vi 2x49GB du chay bf16 DDP
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Multi-GPU: "ddp" hoac "deepspeed"
    multi_gpu_strategy: str = "ddp"

    # Image
    max_image_size: int = 1024  # resize anh lon de giam image tokens

    # save_steps ~ 1 epoch: 1394 anh / (batch=2 * grad_accum=4 * 2GPU) = ~87 steps
    # Dat 175 de eval moi ~1 epoch, tranh early stop qua som

    # Misc
    dataloader_num_workers: int = 0  # 0 tranh loi DDP worker
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

# ============================================================
# DATASET (dung cho _SubsetDataset trong training)
# ============================================================
class _SubsetDataset(Dataset):
    """Dataset load thang anh tu disk, khong augmentation runtime.
    Augmentation da duoc thuc hien truoc bang augument.py va luu vao folder rieng.
    """

    def __init__(
        self,
        image_paths: list[Path],
        config: FinetuneConfig,
        processor,
        is_train: bool,
    ):
        self.config = config
        self.processor = processor
        self.is_train = is_train

        labels_dir = Path(config.data_dir) / config.labels_subdir
        # Chi giu cac sample co file label tuong ung
        self.samples: list[tuple[Path, Path]] = [
            (p, labels_dir / (p.stem + ".json"))
            for p in image_paths
            if (labels_dir / (p.stem + ".json")).exists()
        ]
        if len(self.samples) < len(image_paths):
            missing = len(image_paths) - len(self.samples)
            import logging as _log
            _log.getLogger(__name__).warning(
                "%d image(s) bi bo qua vi khong co label tuong ung", missing
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Khong mo duoc anh: {img_path}") from e

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Label JSON loi: {label_path}") from e

        target_text = json.dumps(label_data, ensure_ascii=False)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": target_text},
        ]
        return {"messages": messages, "image": image, "target": target_text}


# ============================================================
# COLLATOR
# ============================================================
class VLMCollator:
    def __init__(self, processor, config: FinetuneConfig):
        self.processor = processor
        self.max_length = config.max_seq_length
        self.max_image_size = config.max_image_size

    def __call__(self, batch: list[dict]) -> dict:
        texts, images_list = [], []
        for item in batch:
            text = self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images_list.append(self._resize_image(item["image"]))

        # QUAN TRONG: khong dung truncation=True
        # Qwen3-VL se bi mismatch image token count neu truncate
        encoding = self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Cat thu cong neu qua dai (chi cat text, khong cat image tokens)
        if encoding["input_ids"].shape[1] > self.max_length:
            encoding["input_ids"] = encoding["input_ids"][:, : self.max_length]
            encoding["attention_mask"] = encoding["attention_mask"][:, : self.max_length]

        # Mask prompt, chi train tren assistant JSON output
        labels = encoding["input_ids"].clone()
        assistant_marker = "<|im_start|>assistant"
        marker_ids = self.processor.tokenizer.encode(
            assistant_marker, add_special_tokens=False
        )
        for i in range(labels.shape[0]):
            input_ids = encoding["input_ids"][i].tolist()
            pos = self._find_sublist(input_ids, marker_ids)
            if pos >= 0:
                labels[i, : pos + len(marker_ids)] = -100
            else:
                labels[i, :] = -100  # mask het neu khong tim thay marker

        encoding["labels"] = labels

        # Weighted loss: sample co bang phuc tap duoc train manh hon
        weights = torch.tensor(
            [self._compute_weight(item["target"]) for item in batch],
            dtype=torch.float32,
        )
        encoding["sample_weights"] = weights

        return encoding

    def _resize_image(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w > self.max_image_size or h > self.max_image_size:
            ratio = min(self.max_image_size / w, self.max_image_size / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    @staticmethod
    def _find_sublist(lst: list, sub: list) -> int:
        for i in range(len(lst) - len(sub) + 1):
            if lst[i : i + len(sub)] == sub:
                return i
        return -1

    @staticmethod
    def _compute_weight(target_text: str) -> float:
        """Tăng weight cho sample có bảng phức tạp."""
        try:
            data = json.loads(target_text)
            text = json.dumps(data)
            weight = 1.0

            num_keys = text.count('":')
            if num_keys > 30:
                weight += 0.3
            if num_keys > 60:
                weight += 0.2

            depth = VLMCollator._max_depth(data)
            if depth >= 3:
                weight += 0.3

            if VLMCollator._has_merged_cells(data):
                weight += 0.5

            return min(weight, 2.5)
        except Exception:
            return 1.0

    @staticmethod
    def _max_depth(obj, depth: int = 0) -> int:
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(VLMCollator._max_depth(v, depth + 1) for v in obj.values())
        if isinstance(obj, list):
            if not obj:
                return depth
            return max(VLMCollator._max_depth(v, depth) for v in obj)
        return depth

    @staticmethod
    def _has_merged_cells(obj) -> bool:
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for row in obj:
                vals = [str(v) for v in row.values() if v != ""]
                if len(vals) != len(set(vals)) and len(vals) > 1:
                    return True
        if isinstance(obj, dict):
            return any(VLMCollator._has_merged_cells(v) for v in obj.values())
        return False


# ============================================================
# WEIGHTED TRAINER
# ============================================================
class WeightedTrainer(Trainer):
    """Trainer ho tro sample_weights trong batch."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        outputs = model(**inputs)
        loss = outputs.loss

        if sample_weights is not None and loss is not None:
            avg_weight = sample_weights.mean().to(loss.device)
            loss = loss * avg_weight

        return (loss, outputs) if return_outputs else loss


# ============================================================
# DEEPSPEED CONFIG (ZeRO Stage-2)
# ============================================================
def get_deepspeed_config() -> dict:
    return {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }


# ============================================================
# SAVE EVERY N EPOCHS CALLBACK
# ============================================================
class SaveEveryNEpochsCallback(TrainerCallback):
    """Save model sau moi epoch vao thu muc rieng."""

    def __init__(self, save_every_n_epochs: int, output_dir: str, processor):
        self.save_every_n_epochs = save_every_n_epochs  # giữ param để tương thích
        self.output_dir = output_dir
        self.processor = processor
        self._last_saved_epoch = -1  # tránh save 2 lần cùng epoch

    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
        if not is_main:
            return

        # state.epoch là float (1.0, 2.0...), dùng round() để lấy epoch nguyên
        epoch = round(state.epoch)
        if epoch <= 0 or epoch == self._last_saved_epoch:
            return
        self._last_saved_epoch = epoch

        save_path = os.path.join(self.output_dir, f"epoch_{epoch}")
        logger.info("Saving epoch checkpoint -> %s", save_path)
        os.makedirs(save_path, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info("Epoch %d checkpoint saved -> %s", epoch, save_path)


# ============================================================
# EPOCH LOSS CALLBACK
# ============================================================
class EpochLossCallback(TrainerCallback):
    """In train loss va eval loss sau moi epoch."""

    def __init__(self):
        self.train_losses: list[float] = []
        self._step_losses: list[float] = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch)
        avg = (
            sum(self._step_losses) / len(self._step_losses)
            if self._step_losses
            else float("nan")
        )
        self.train_losses.append(avg)
        self._step_losses = []

        eval_loss = None
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break

        sep = "=" * 52
        if eval_loss is not None:
            logger.info(
                "\n%s\n  EPOCH %d / %d\n  Train Loss : %.4f\n  Eval  Loss : %.4f\n%s",
                sep, epoch, int(args.num_train_epochs), avg, eval_loss, sep,
            )
        else:
            logger.info(
                "\n%s\n  EPOCH %d / %d\n  Train Loss : %.4f\n%s",
                sep, epoch, int(args.num_train_epochs), avg, sep,
            )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("\n===== ALL EPOCHS SUMMARY =====")
        for i, loss in enumerate(self.train_losses, 1):
            logger.info("  Epoch %2d : train_loss = %.4f", i, loss)
        logger.info("=" * 30)


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    cfg = FinetuneConfig()

    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main:
        logger.info("=" * 60)
        logger.info("Detected %d GPU(s):", num_gpus)
        for i in range(num_gpus):
            p = torch.cuda.get_device_properties(i)
            logger.info("  GPU %d: %s — %.1f GB", i, p.name, p.total_memory / 1e9)
        logger.info("=" * 60)

    # ---- Quantization ----
    bnb_config = None
    if cfg.use_4bit:
        if num_gpus > 1:
            logger.warning("4-bit + multi-GPU: se dung model parallel thay vi DDP.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    # ---- Processor ----
    if is_main:
        logger.info("Loading processor: %s", cfg.model_name)
    processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- Model ----
    if is_main:
        logger.info("Loading model: %s", cfg.model_name)
    if cfg.use_4bit:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # ---- LoRA ----
    if cfg.use_lora:
        if cfg.use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "Frozen: %s params | Trainable: %s params (%.2f%%)",
                f"{frozen:,}", f"{trainable:,}",
                100 * trainable / (frozen + trainable),
            )

    # ---- Dataset split ----
    all_images = sorted([
        f for f in (Path(cfg.data_dir) / cfg.images_subdir).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
        and (Path(cfg.data_dir) / cfg.labels_subdir / (f.stem + ".json")).exists()
    ])
    random.seed(cfg.seed)
    random.shuffle(all_images)
    val_size = min(600, len(all_images) - 1)
    train_images = all_images[val_size:]
    val_images = all_images[:val_size]

    if is_main:
        logger.info(
            "Split: %d train | %d val (load thang tu disk, khong augment runtime)",
            len(train_images), len(val_images),
        )

    train_dataset = _SubsetDataset(train_images, cfg, processor, is_train=True)
    val_dataset = _SubsetDataset(val_images, cfg, processor, is_train=False)

    # ---- Collator ----
    collator = VLMCollator(processor, cfg)

    # ---- DeepSpeed ----
    deepspeed_cfg = None
    if num_gpus > 1 and cfg.multi_gpu_strategy == "deepspeed":
        deepspeed_cfg = get_deepspeed_config()
        if is_main:
            logger.info("Strategy: DeepSpeed ZeRO-2")
    elif num_gpus > 1:
        if is_main:
            logger.info("Strategy: PyTorch DDP")
    else:
        if is_main:
            logger.info("Strategy: Single GPU")

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_cfg,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ---- Trainer ----
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[
            EpochLossCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            ),
            SaveEveryNEpochsCallback(
                save_every_n_epochs=cfg.save_every_n_epochs,
                output_dir=cfg.output_dir,
                processor=processor,
            ),
        ],
    )

    # ---- Train ----
    if is_main:
        logger.info("=" * 60)
        logger.info("Starting training...")
        effective_batch = (
            cfg.per_device_train_batch_size * num_gpus * cfg.gradient_accumulation_steps
        )
        logger.info("Effective batch size: %d", effective_batch)
        logger.info("=" * 60)

    trainer.train()

    # ---- Save final ----
    if is_main:
        logger.info("Saving model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    if is_main:
        processor.save_pretrained(cfg.output_dir)
        logger.info("Done! Model saved.")


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()
