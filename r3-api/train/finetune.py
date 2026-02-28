"""
Finetune Chandra (Qwen3-VL) - Image -> JSON voi table handling day du.

Ho tro:
  - 2x NVIDIA RTX 6000 Ada (2x 49GB) voi DDP
  - Table handling: form grid vs true table, merged cells, nested tables
  - Image augmentation cho table/form images
  - Early stopping, epoch loss logging
  - Weighted loss cho samples co bang phuc tap

Cach chay:
    pip install transformers accelerate peft bitsandbytes pillow tqdm datasets deepspeed opencv-python

    # 2 GPU (khuyen nghi)
    torchrun --nproc_per_node=2 finetune_chandra.py

    # 1 GPU
    python finetune_chandra.py
"""

import io
import os
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT - day du table handling rules
# ============================================================
SYSTEM_PROMPT = """You are an OCR assistant. Extract all information from the document image and return it as a single valid JSON object.

OUTPUT FORMAT:
- Return ONLY valid JSON.
- Do NOT output HTML.
- Do NOT wrap in markdown or code blocks.
- Do NOT add explanations.
- Preserve exact original text (%, currency symbols, thousand separators, parentheses).
- Maintain correct reading order (top-to-bottom, left-to-right).
- If a value is visually empty, use "".

========================
TABLE HANDLING (STRICT)
========================

1) FIRST distinguish FORM GRID vs TRUE TABLE:
   TRUE TABLE:
     - Has consistent column headers.
     - Contains repeated row records aligned under those headers.
     - Multiple rows share the same column structure.
   FORM GRID / BOXED LAYOUT:
     - Is still key-value structure.
     - Even if visually drawn with borders or boxes.
     - Includes signature blocks, approval sections, checkbox areas.
     - DO NOT convert these into tables.
     - Represent them as key-value JSON structure.
   Only TRUE TABLES should be converted into JSON tables.

2) TABLE INSIDE FORM SECTION:
   If a TRUE TABLE appears inside a form section
   (e.g., inside a field like "So tien (With amount)"):
   - Nest that table inside that section object.
   - Do NOT output it as a top-level table.
   Example:
   {
     "So tien (With amount)": {
       "Table1": [...]
     }
   }

3) TABLE NAMING:
   - Use the table explicit title/caption/name as the JSON key.
   - If no explicit title exists, assign sequential fallback names: Table1, Table2, ...

4) TABLE FORMAT:
   Each TRUE TABLE must be formatted as:
   "Table Name": [
     {
       "Column Header 1": "value",
       "Column Header 2": "value"
     }
   ]
   STRICT RULES:
   - Each row = one dictionary.
   - Keys MUST be flat.
   - Keys MUST exactly match column headers.
   - Do NOT nest columns.

5) COLUMN ALIGNMENT:
   - Derive canonical column structure from header row.
   - Every row MUST follow the same structure.
   - If missing value -> use "".
   - Do NOT shift values between columns.

6) MERGED CELL RULE:
   If a cell visually spans multiple columns:
   - DUPLICATE (repeat) the merged value into EACH covered column in that JSON row.
   - Do NOT use colspan metadata.
   - Do NOT drop columns.
   Example: If "100" spans Q1, Q2, Q3:
   {"Q1": "100", "Q2": "100", "Q3": "100"}

7) TOTAL ROW RULE:
   If a standalone line like "Total" or "Tong cong" appears immediately below a table:
   - Treat it as the FINAL ROW of that table.
   - Append it inside the same table array.
   - Do NOT output it separately.

8) MULTI-SEGMENT TABLE:
   If a table is visually split into stacked segments but shares aligned columns:
   - Merge them into ONE logical table.

Return a single valid JSON object."""

USER_PROMPT = (
    "OCR this document image into structured JSON. "
    "Apply all TABLE HANDLING rules strictly. "
    "Return ONLY valid JSON."
)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class FinetuneConfig:
    # Dataset
    data_dir: str = "/root/khaint02/final"
    images_subdir: str = "images"
    labels_subdir: str = "label"

    # Model
    model_name: str = "/root/khaint02/model_goc"
    output_dir: str = "/root/khaint02/chandra_finetuned"

    # Training
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_seq_length: int = 6192
    save_steps: int = 50
    logging_steps: int = 10
    eval_split: float = 0.05

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Save every N epochs (0 = disable)
    save_every_n_epochs: int = 3

    # LoRA - rank cao cho bang phuc tap
    use_lora: bool = True
    lora_r: int = 14
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization - False vi 2x49GB du chay bf16 DDP
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Multi-GPU: "ddp" hoac "deepspeed"
    multi_gpu_strategy: str = "ddp"

    # Augmentation
    use_augmentation: bool = True
    augment_factor: int = 2  # moi anh tao them 1 ban augmented -> x2 train data

    # Image
    max_image_size: int = 1024  # resize anh lon de giam image tokens

    # Misc
    dataloader_num_workers: int = 0  # 0 tranh loi DDP worker
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


# ============================================================
# TABLE AUGMENTATION
# ============================================================
class TableAugmentation:
    """Augmentation dac biet cho anh bang/form."""

    def __call__(self, image: Image.Image) -> Image.Image:
        augments = [
            (0.4, self._rotate),
            (0.3, self._add_noise),
            (0.3, self._simulate_shadow),
            (0.2, self._perspective_warp),
            (0.3, self._compress_artifact),
            (0.4, self._brightness),
            (0.4, self._contrast),
            (0.2, self._blur),
        ]
        for prob, fn in augments:
            if random.random() < prob:
                try:
                    image = fn(image)
                except Exception:
                    pass  # giu nguyen neu augment loi
        return image

    @staticmethod
    def _rotate(img):
        angle = random.uniform(-3, 3)
        return img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    @staticmethod
    def _add_noise(img):
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 6), arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    @staticmethod
    def _simulate_shadow(img):
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        strength = random.uniform(0.65, 0.92)
        split = random.randint(w // 4, 3 * w // 4)
        mask = np.ones((h, w, 1), dtype=np.float32)
        if random.random() < 0.5:
            mask[:, :split] = strength
        else:
            mask[:, split:] = strength
        return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

    @staticmethod
    def _perspective_warp(img):
        try:
            import cv2
            arr = np.array(img)
            h, w = arr.shape[:2]
            m = int(min(h, w) * 0.015)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [random.randint(0, m), random.randint(0, m)],
                [w - random.randint(0, m), random.randint(0, m)],
                [w - random.randint(0, m), h - random.randint(0, m)],
                [random.randint(0, m), h - random.randint(0, m)],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(arr, M, (w, h), borderValue=(255, 255, 255))
            return Image.fromarray(warped)
        except ImportError:
            return img

    @staticmethod
    def _compress_artifact(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(60, 88))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def _brightness(img):
        return ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.25))

    @staticmethod
    def _contrast(img):
        return ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.25))

    @staticmethod
    def _blur(img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))


# ============================================================
# DATASET
# ============================================================
class ImageJsonDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    def __init__(
        self,
        data_dir: str,
        images_subdir: str,
        labels_subdir: str,
        processor,
        config: FinetuneConfig,
        is_train: bool = True,
    ):
        self.processor = processor
        self.config = config
        self.is_train = is_train
        self.augmentor = TableAugmentation() if (is_train and config.use_augmentation) else None
        self.samples = []

        images_dir = Path(data_dir) / images_subdir
        labels_dir = Path(data_dir) / labels_subdir

        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        ])
        logger.info("Found %d images in %s", len(image_files), images_dir)

        missing = 0
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".json")
            if not label_path.exists():
                missing += 1
                continue
            self.samples.append((img_path, label_path))

        if missing:
            logger.warning("%d images skipped (no matching label)", missing)
        logger.info("Valid samples: %d (is_train=%s)", len(self.samples), is_train)

        # Augment factor: nhan ban train set
        self._base_len = len(self.samples)
        self._total_len = (
            self._base_len * config.augment_factor if is_train else self._base_len
        )

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        real_idx = idx % self._base_len
        is_augmented_copy = idx >= self._base_len

        img_path, label_path = self.samples[real_idx]
        image = Image.open(img_path).convert("RGB")

        # Chi augment ban sao, giu nguyen ban goc
        if is_augmented_copy and self.augmentor is not None:
            image = self.augmentor(image)

        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
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
        return {
            "messages": messages,
            "image": image,
            "target": target_text,
        }


# ============================================================
# COLLATOR
# ============================================================
class VLMCollator:
    def __init__(self, processor, config: FinetuneConfig):
        self.processor = processor
        self.max_length = config.max_seq_length
        self.max_image_size = config.max_image_size

    def __call__(self, batch):
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
    def _find_sublist(lst, sub):
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

            # Co nhieu key -> form/table phuc tap
            num_keys = text.count('":')
            if num_keys > 30:
                weight += 0.3
            if num_keys > 60:
                weight += 0.2

            # Co nested structure (table trong form)
            depth = VLMCollator._max_depth(data)
            if depth >= 3:
                weight += 0.3

            # Co merged cells (gia tri trung lap trong cung row)
            if VLMCollator._has_merged_cells(data):
                weight += 0.5

            return min(weight, 2.5)
        except Exception:
            return 1.0

    @staticmethod
    def _max_depth(obj, depth=0):
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
    def _has_merged_cells(obj):
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
            # Scale loss theo weight trung binh cua batch
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
    """Save model sau moi N epoch vao thu muc rieng.
    Dung de dam bao khong mat progress khi train epoch lon.
    """

    def __init__(self, save_every_n_epochs: int, output_dir: str, processor):
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir
        self.processor = processor

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        epoch = int(state.epoch)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main = local_rank == 0

        if epoch > 0 and epoch % self.save_every_n_epochs == 0:
            save_path = os.path.join(self.output_dir, f"epoch_{epoch}")
            if is_main:
                logger.info("Saving epoch checkpoint -> %s", save_path)
                os.makedirs(save_path, exist_ok=True)
                # Unwrap model (DDP / PEFT)
                unwrapped = model
                if hasattr(model, "module"):
                    unwrapped = model.module
                unwrapped.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                logger.info("Epoch %d checkpoint saved -> %s", epoch, save_path)


# ============================================================
# EPOCH LOSS CALLBACK
# ============================================================
class EpochLossCallback(TrainerCallback):
    """In train loss va eval loss sau moi epoch."""

    def __init__(self):
        self.train_losses = []
        self._step_losses = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch)
        avg = (
            sum(self._step_losses) / len(self._step_losses)
            if self._step_losses else float("nan")
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
# MAIN
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
            torch_dtype=torch.bfloat16,
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

            # Log frozen vs trainable
            frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                "Frozen: %s params | Trainable: %s params (%.2f%%)",
                f"{frozen:,}", f"{trainable:,}",
                100 * trainable / (frozen + trainable),
            )

    # ---- Dataset (train/val rieng biet) ----
    all_images = sorted([
        f for f in (Path(cfg.data_dir) / cfg.images_subdir).iterdir()
        if f.suffix.lower() in ImageJsonDataset.IMAGE_EXTENSIONS
        and (Path(cfg.data_dir) / cfg.labels_subdir / (f.stem + ".json")).exists()
    ])
    random.seed(cfg.seed)
    random.shuffle(all_images)
    val_size = max(1, int(len(all_images) * cfg.eval_split))
    train_images = all_images[val_size:]
    val_images = all_images[:val_size]

    if is_main:
        logger.info(
            "Split: %d train (x%d aug = %d effective) | %d val",
            len(train_images), cfg.augment_factor,
            len(train_images) * cfg.augment_factor, len(val_images),
        )

    # Tao dataset voi is_train flag rieng
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

    # ---- Save ----
    if is_main:
        logger.info("Saving model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    if is_main:
        processor.save_pretrained(cfg.output_dir)
        logger.info("Done! Model saved.")


# ============================================================
# HELPER: SubsetDataset (train va val dung is_train rieng)
# ============================================================
class _SubsetDataset(Dataset):
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

    def __init__(self, image_paths, config: FinetuneConfig, processor, is_train: bool):
        self.config = config
        self.processor = processor
        self.is_train = is_train
        self.augmentor = TableAugmentation() if (is_train and config.use_augmentation) else None

        labels_dir = Path(config.data_dir) / config.labels_subdir
        self.samples = [
            (p, labels_dir / (p.stem + ".json"))
            for p in image_paths
        ]
        self._base_len = len(self.samples)
        self._total_len = (
            self._base_len * config.augment_factor if is_train else self._base_len
        )

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        real_idx = idx % self._base_len
        is_aug_copy = idx >= self._base_len

        img_path, label_path = self.samples[real_idx]
        image = Image.open(img_path).convert("RGB")

        if is_aug_copy and self.augmentor:
            image = self.augmentor(image)

        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
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


if __name__ == "__main__":
    main()