import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IDP_API")

app = FastAPI(title="IDP Extraction & Classification API")


@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    """Return classification built from JSON files currently in cache directory."""
    logger.info(f"========== NEW CLASSIFICATION REQUEST: {file.filename} ==========")
    temp_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        class_array = load_static_classification(JSON_FILES_CACHE_DIR, logger)
        page_count = sum(len(item.get("pages", [])) for item in class_array)

        logger.info(f"[*] Returning static classification from cache ({page_count} pages).")
        return success_response(page_count=page_count, classification=class_array)

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
