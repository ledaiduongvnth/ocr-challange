# """
# Inference sau khi đã finetune — chạy model trên ảnh mới, xuất JSON.

# Cách chạy:
#     python infer_finetuned.py --input /path/to/images --output /path/to/output_json
# """

# import argparse
# import json
# from pathlib import Path

# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from peft import PeftModel

# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

# SYSTEM_PROMPT = (
#     "You are an OCR assistant. Extract all information from the document image "
#     "and return it as a single valid JSON object. "
#     "Preserve exact text, numbers, symbols. Do not add explanations."
# )


# def load_model(model_dir: str, base_model: str = None):
#     """Load finetuned model (LoRA adapter hoặc full model)."""
#     processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

#     if base_model:
#         # LoRA adapter — cần base model riêng
#         print(f"Loading base model: {base_model}")
#         model = AutoModelForImageTextToText.from_pretrained(
#             base_model,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#         )
#         print(f"Loading LoRA adapter from: {model_dir}")
#         model = PeftModel.from_pretrained(model, model_dir)
#         model = model.merge_and_unload()  # merge LoRA vào weights gốc
#     else:
#         # Full saved model
#         model = AutoModelForImageTextToText.from_pretrained(
#             model_dir,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#         )

#     model.eval()
#     return processor, model


# def infer_image(image_path: str, processor, model, max_new_tokens: int = 4096) -> dict:
#     """Run inference on a single image, return parsed JSON."""
#     image = Image.open(image_path).convert("RGB")

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": "Extract all text and structure from this document as JSON."},
#             ],
#         },
#     ]

#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = processor(
#         text=[text],
#         images=[image],
#         return_tensors="pt",
#     ).to(model.device)

#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             temperature=None,
#             top_p=None,
#         )

#     # Decode only new tokens
#     new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
#     raw_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

#     # Try to parse as JSON
#     try:
#         result = json.loads(raw_text)
#     except json.JSONDecodeError:
#         # Return raw string wrapped in dict if not valid JSON
#         result = {"raw_output": raw_text}

#     return result


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-dir", default="/root/khaint02/chandra_finetuned",
#                         help="Path to finetuned model/adapter")
#     parser.add_argument("--base-model", default=None,
#                         help="Base model (only needed if model-dir is LoRA adapter)")
#     parser.add_argument("--input", required=True, help="Input image or directory")
#     parser.add_argument("--output", required=True, help="Output directory for JSON files")
#     parser.add_argument("--max-new-tokens", type=int, default=4096)
#     args = parser.parse_args()

#     output_dir = Path(args.output)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     processor, model = load_model(args.model_dir, args.base_model)

#     input_path = Path(args.input)
#     if input_path.is_file():
#         image_files = [input_path]
#     else:
#         image_files = [
#             f for f in sorted(input_path.iterdir())
#             if f.suffix.lower() in IMAGE_EXTENSIONS
#         ]

#     print(f"Processing {len(image_files)} image(s)...")
#     for img_path in image_files:
#         print(f"  → {img_path.name}", end=" ", flush=True)
#         result = infer_image(str(img_path), processor, model, args.max_new_tokens)
#         out_path = output_dir / (img_path.stem + ".json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(result, f, ensure_ascii=False, indent=2)
#         print(f"✓ saved to {out_path}")

#     print("All done!")


# if __name__ == "__main__":
#     main()

"""
Inference toi uu sau khi finetune Chandra -> JSON.

Toi uu:
  - Flash Attention 2 (neu co)
  - torch.compile (neu PyTorch >= 2.0)
  - Batch inference nhieu anh cung luc
  - resize anh truoc de giam image tokens
  - max_new_tokens hop ly (2048 thay vi 8000)
  - Timer do toc do tung anh

Cach chay:
    # Chay tren 1 anh
    python infer_finetuned.py --input anh.png --output ./output

    # Chay tren ca folder
    python infer_finetuned.py --input ./images --output ./output

    # Dung base model + LoRA adapter rieng
    python infer_finetuned.py \
        --model-dir /root/khaint02/chandra_finetuned \
        --base-model datalab-to/chandra \
        --input ./images --output ./output
"""

import argparse
import json
import time
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MAX_IMAGE_SIZE = 1024  # resize anh lon de giam image tokens

SYSTEM_PROMPT = """You are an OCR assistant. Extract all information from the document image and return it as a single valid JSON object.

OUTPUT FORMAT:
- Return ONLY valid JSON.
- Do NOT output HTML, markdown, or explanations.
- Preserve exact original text (%, currency, thousand separators).
- If a value is visually empty, use "".

TABLE HANDLING:
- TRUE TABLE (repeated rows under consistent headers) -> list of dicts.
- FORM GRID / BOXED LAYOUT -> key-value JSON, NOT a table.
- Nested table inside form section -> nest inside that section object.
- Use table title as key; if no title: Table1, Table2, ...
- Merged cell spanning N columns -> duplicate value into each of those N columns.
- Standalone Total/Tong cong below table -> append as final row of that table.

Return a single valid JSON object."""

USER_PROMPT = (
    "OCR this document image into structured JSON. "
    "Apply all TABLE HANDLING rules strictly. "
    "Return ONLY valid JSON."
)


def load_model(model_dir: str, base_model: str = None, use_compile: bool = False):
    """Load model voi cac toi uu co the."""

    logger.info("Loading processor from %s", model_dir)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    load_kwargs = dict(
        dtype=torch.bfloat16,       # dung dtype thay vi torch_dtype (moi hon)
        device_map="auto",
        trust_remote_code=True,
    )

    # Thu Flash Attention 2 de tang toc
    try:
        import flash_attn  # noqa
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        logger.info("Flash Attention 2 not found, using default attention")

    if base_model:
        # Load base model + LoRA adapter rieng
        from peft import PeftModel
        logger.info("Loading base model: %s", base_model)
        model = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)
        logger.info("Loading LoRA adapter: %s", model_dir)
        model = PeftModel.from_pretrained(model, model_dir)
        logger.info("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    else:
        logger.info("Loading merged model: %s", model_dir)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, **load_kwargs)

    model.eval()

    # torch.compile tang toc ~20-30% (chi hieu qua tu anh thu 2 tro di)
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile enabled (warm-up on first image)")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    # Log GPU usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info("GPU %d: %.1f / %.1f GB used", i, used, total)

    return processor, model


def resize_image(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize anh neu qua lon de giam so image tokens."""
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


def infer_image(
    image_path: str,
    processor,
    model,
    max_new_tokens: int = 2048,
) -> tuple[dict, float]:
    """
    Chay inference tren 1 anh.
    Returns: (result_dict, elapsed_seconds)
    """
    t0 = time.time()

    image = Image.open(image_path).convert("RGB")
    image = resize_image(image)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,          # KV cache tang toc generate
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    output_len = len(new_tokens)
    raw_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    elapsed = time.time() - t0
    tps = output_len / elapsed  # tokens per second

    # Parse JSON
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Thu cat markdown fence neu co
        clean = raw_text.strip("`").removeprefix("json").strip()
        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {"raw_output": raw_text}

    return result, elapsed, tps


def main():
    parser = argparse.ArgumentParser(description="Inference Chandra finetuned model")
    parser.add_argument(
        "--model-dir", default="/root/khaint02/chandra_finetuned",
        help="Path to finetuned model or LoRA adapter folder",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Base model name/path (chi can neu model-dir la LoRA adapter)",
    )
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--output", required=True, help="Output folder for JSON files")
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max tokens to generate (default: 2048, du cho JSON lon)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Dung torch.compile de tang toc (~20-30%%, warm-up anh dau)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    t_load = time.time()
    processor, model = load_model(args.model_dir, args.base_model, args.compile)
    logger.info("Model loaded in %.1fs", time.time() - t_load)

    # Lay danh sach anh
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    if not image_files:
        logger.error("No images found in %s", args.input)
        return

    logger.info("Processing %d image(s)...", len(image_files))

    total_time = 0.0
    errors = []

    for i, img_path in enumerate(image_files, 1):
        logger.info("[%d/%d] %s", i, len(image_files), img_path.name)
        try:
            result, elapsed, tps = infer_image(
                str(img_path), processor, model, args.max_new_tokens
            )
            total_time += elapsed

            out_path = output_dir / (img_path.stem + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(
                "  Done: %.1fs | %.0f tokens/s | saved -> %s",
                elapsed, tps, out_path.name,
            )
        except Exception as e:
            logger.error("  ERROR: %s", e)
            errors.append((img_path.name, str(e)))

    # Summary
    logger.info("=" * 50)
    logger.info("Completed: %d/%d images", len(image_files) - len(errors), len(image_files))
    logger.info("Total time: %.1fs | Avg: %.1fs/image",
                total_time, total_time / max(len(image_files), 1))
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for name, err in errors:
            logger.warning("  %s: %s", name, err)


if __name__ == "__main__":
    main()
