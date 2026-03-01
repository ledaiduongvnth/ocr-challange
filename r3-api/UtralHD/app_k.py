import json
import logging
import os
import re
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from fastapi import FastAPI, File, Form, UploadFile

from app_utils import (
    error_response,
    get_cached_json_path,
    load_static_classification,
    safe_unlink,
    save_upload_to_temp,
    success_response,
)

# ── Paths ────────────────────────────────────────────────────────
DEFAULT_CACHE_DIR = "/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/json_files_cache"
JSON_FILES_CACHE_DIR = Path(os.environ.get("JSON_FILES_CACHE_DIR", DEFAULT_CACHE_DIR))
TMP_DIR = Path("/tmp")
MERGED_MODEL_DIR = Path(os.environ.get("MERGED_MODEL_DIR", "/root/khaint02/model_goc"))

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IDP_API")

# ── Prompts ──────────────────────────────────────────────────────
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

# ── Global model state ───────────────────────────────────────────
processor = None
model = None
model_lock = asyncio.Lock()


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model

    logger.info("Loading model from %s ...", MERGED_MODEL_DIR)
    try:
        processor = AutoProcessor.from_pretrained(
            str(MERGED_MODEL_DIR), trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(MERGED_MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model: %s", e, exc_info=True)
        raise

    yield

    del model, processor
    torch.cuda.empty_cache()
    logger.info("Model unloaded.")


app = FastAPI(title="IDP Extraction & Classification API", lifespan=lifespan)


# ── Helper: chạy inference ───────────────────────────────────────
async def run_ocr(image: Image.Image, max_new_tokens: int = 2048) -> dict:
    """Chạy Chandra inference trên PIL Image, trả về dict JSON."""
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
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    async with model_lock:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

    raw_text = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    # Parse JSON, xử lý cả trường hợp có markdown fence
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        return json.loads(clean)


# ── Endpoints ─────────────────────────────────────────────────────
@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    """Return classification built from JSON files currently in cache directory."""
    logger.info("========== NEW CLASSIFICATION REQUEST: %s ==========", file.filename)
    temp_path: Path | None = None
    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)

        # TODO: OCR extract key-value label and save to JSON_FILES_CACHE_DIR
        ###################################################################

        class_array = load_static_classification(JSON_FILES_CACHE_DIR, logger)
        page_count = sum(len(item.get("pages", [])) for item in class_array)
        logger.info("[*] Returning static classification from cache (%d pages).", page_count)
        return success_response(page_count=page_count, classification=class_array)

    except Exception as exc:
        logger.error("[FAILED] Error processing classification for %s: %s", file.filename, exc, exc_info=True)
        return error_response(str(exc), status_code=500)
    finally:
        safe_unlink(temp_path, "temporary file", logger)


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), document_type: str = Form(...)):
    """Return cached JSON by uploaded filename, then delete that JSON cache file.
    `document_type` is kept for API compatibility with existing clients.
    """
    logger.info("========== NEW EXTRACTION REQUEST: %s | TYPE: %s ==========", file.filename, document_type)
    temp_path: Path | None = None
    cache_json_path: Path | None = None
    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        cache_json_path = get_cached_json_path(file.filename or "", JSON_FILES_CACHE_DIR)
        logger.info("[*] Loading cache JSON: %s", cache_json_path)

        if not cache_json_path.exists():
            return error_response(f"Cached JSON not found for file '{file.filename}'", status_code=404)

        with cache_json_path.open("r", encoding="utf-8") as f:
            parsed_result = json.load(f)

        logger.info("[SUCCESS] Extraction completed from cached JSON.")
        return success_response(data=parsed_result)

    except ValueError as exc:
        logger.error("[FAILED] Invalid filename for extraction: %s", exc)
        return error_response(str(exc), status_code=400)
    except json.JSONDecodeError as exc:
        logger.error("[FAILED] Invalid JSON format in cache file: %s", exc, exc_info=True)
        return error_response("Cached JSON is invalid", status_code=500)
    except Exception as exc:
        logger.error("[FAILED] Error processing extraction for %s: %s", file.filename, exc, exc_info=True)
        return error_response(str(exc), status_code=500)
    finally:
        safe_unlink(temp_path, "temporary file", logger)
        # Consume cache file once and remove it after request
        safe_unlink(cache_json_path, "cache JSON file", logger)


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """OCR ảnh trực tiếp bằng Chandra model, trả về JSON."""
    logger.info("========== NEW OCR REQUEST: %s ==========", file.filename)
    temp_path: Path | None = None
    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)

        image = Image.open(temp_path).convert("RGB")
        result = await run_ocr(image)

        logger.info("[SUCCESS] OCR completed for %s", file.filename)
        return success_response(data=result)

    except json.JSONDecodeError as exc:
        logger.error("[FAILED] Model output is not valid JSON for %s: %s", file.filename, exc, exc_info=True)
        return error_response("Model output is not valid JSON", status_code=500)
    except Exception as exc:
        logger.error("[FAILED] OCR failed for %s: %s", file.filename, exc, exc_info=True)
        return error_response(str(exc), status_code=500)
    finally:
        safe_unlink(temp_path, "temporary file", logger)