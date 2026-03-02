import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

import fitz
from fastapi import FastAPI, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app_utils import (
    error_response,
    get_cached_json_path,
    load_static_classification,
    safe_unlink,
    save_upload_to_temp,
    success_response,
)

# API returns JSON cached by filename in this directory.
DEFAULT_CACHE_DIR = "/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/json_files_cache"
JSON_FILES_CACHE_DIR = Path(os.environ.get("JSON_FILES_CACHE_DIR", DEFAULT_CACHE_DIR))
TMP_DIR = Path("/tmp")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

# vLLM OpenAI-compatible endpoint.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:7888/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "chandra")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
VLLM_TIMEOUT_SEC = int(os.environ.get("VLLM_TIMEOUT_SEC", "300"))
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "1024"))
VLLM_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", "0.0"))
PDF_RENDER_SCALE = float(os.environ.get("PDF_RENDER_SCALE", "1.5"))

SYSTEM_PROMPT = (
    "You are an OCR assistant. Extract key-value information from the document image and "
    "return only one valid JSON object. No markdown. No explanation."
)

USER_PROMPT = (
    "Extract all visible key-value pairs and table content from this page into JSON. "
    "Return only valid JSON."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IDP_API")

app = FastAPI(title="IDP Extraction & Classification API")


def _extract_content(resp_json: dict[str, Any]) -> str:
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()

    return str(content).strip()


def _extract_json_block(text: str) -> str:
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        return text.strip()
    return text[min(starts):].strip()


def _safe_json_parse(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except json.JSONDecodeError:
        candidate = _extract_json_block(text)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            return {"raw_output": text}


def _build_vllm_payload(image_data_url: str) -> dict[str, Any]:
    return {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": VLLM_TEMPERATURE,
        "max_tokens": VLLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def _post_vllm(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = VLLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with request.urlopen(req, timeout=VLLM_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _infer_page_json(image_data_url: str) -> dict[str, Any]:
    payload = _build_vllm_payload(image_data_url)

    try:
        resp_json = _post_vllm(payload)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Fallback for servers that do not support response_format.
        if exc.code == 400 and "response_format" in body:
            payload.pop("response_format", None)
            try:
                resp_json = _post_vllm(payload)
            except Exception as retry_exc:
                raise RuntimeError(f"vLLM request failed after fallback: {retry_exc}") from retry_exc
        else:
            raise RuntimeError(f"vLLM HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"vLLM request failed: {exc}") from exc

    raw_text = _extract_content(resp_json)
    return _safe_json_parse(raw_text)


def _image_path_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _iter_pdf_page_data_urls(pdf_path: Path) -> Iterable[tuple[int, str]]:
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
            encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            yield page_idx, f"data:image/png;base64,{encoded}"


def _iter_page_inputs(temp_path: Path) -> Iterable[tuple[int, str]]:
    suffix = temp_path.suffix.lower()
    if suffix == ".pdf":
        return _iter_pdf_page_data_urls(temp_path)

    if suffix in IMAGE_EXTENSIONS:
        return [(0, _image_path_to_data_url(temp_path))]

    raise ValueError(f"Unsupported upload format: '{suffix}'. Use PDF or image file.")


def _clear_cache_dir() -> None:
    JSON_FILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old_json in JSON_FILES_CACHE_DIR.glob("*.json"):
        safe_unlink(old_json, "stale cache JSON file", logger)


def _cache_inference_results(temp_path: Path) -> None:
    _clear_cache_dir()

    page_inputs = _iter_page_inputs(temp_path)
    page_count = 0
    success_count = 0
    logger.info("[*] Running vLLM inference and caching page outputs...")

    for page_idx, data_url in page_inputs:
        page_count += 1
        try:
            parsed_json = _infer_page_json(data_url)
            if "error" not in parsed_json:
                success_count += 1
        except Exception as exc:
            logger.error(f"[-] Page {page_idx} inference failed: {exc}")
            parsed_json = {"error": str(exc)}

        out_path = JSON_FILES_CACHE_DIR / f"split_document_page_{page_idx}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)

        logger.info(f"[*] Cached page JSON: {out_path.name}")

    if page_count == 0:
        raise RuntimeError("No pages found for inference")
    if success_count == 0:
        raise RuntimeError("Inference failed for all pages")


@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    """Run OCR inference on uploaded PDF/image and return grouped classification."""
    logger.info(f"========== NEW CLASSIFICATION REQUEST: {file.filename} ==========")
    temp_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        # vLLM calls and PDF rendering are blocking; run in a worker thread.
        await run_in_threadpool(_cache_inference_results, temp_path)

        class_array = load_static_classification(JSON_FILES_CACHE_DIR, logger)
        page_count = sum(len(item.get("pages", [])) for item in class_array)

        logger.info(f"[*] Classification ready ({page_count} pages).")
        return success_response(page_count=page_count, classification=class_array)

    except ValueError as exc:
        logger.error(f"[FAILED] Invalid input for classification: {exc}")
        return error_response(str(exc), status_code=400)

    except Exception as exc:
        logger.error(f"[FAILED] Error processing classification for {file.filename}: {exc}", exc_info=True)
        return error_response(str(exc), status_code=500)

    finally:
        safe_unlink(temp_path, "temporary file", logger)


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), document_type: str = Form(...)):
    """Return cached JSON by uploaded filename, then delete that JSON cache file.

    `document_type` is kept for API compatibility with existing clients.
    """
    logger.info(f"========== NEW EXTRACTION REQUEST: {file.filename} | TYPE: {document_type} ==========")
    temp_path: Path | None = None
    cache_json_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        cache_json_path = get_cached_json_path(file.filename or "", JSON_FILES_CACHE_DIR)
        logger.info(f"[*] Loading cache JSON: {cache_json_path}")

        if not cache_json_path.exists():
            return error_response(f"Cached JSON not found for file '{file.filename}'", status_code=404)

        with cache_json_path.open("r", encoding="utf-8") as f:
            parsed_result = json.load(f)

        logger.info("[SUCCESS] Extraction completed from cached JSON.")
        return success_response(data=parsed_result)

    except ValueError as exc:
        logger.error(f"[FAILED] Invalid filename for extraction: {exc}")
        return error_response(str(exc), status_code=400)

    except json.JSONDecodeError as exc:
        logger.error(f"[FAILED] Invalid JSON format in cache file: {exc}", exc_info=True)
        return error_response("Cached JSON is invalid", status_code=500)

    except Exception as exc:
        logger.error(f"[FAILED] Error processing extraction for {file.filename}: {exc}", exc_info=True)
        return error_response(str(exc), status_code=500)

    finally:
        safe_unlink(temp_path, "temporary file", logger)
        # Requested behavior: consume cache file once and remove it after request.
        safe_unlink(cache_json_path, "cache JSON file", logger)
