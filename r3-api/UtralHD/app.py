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
VLLM_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "4096"))
VLLM_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", "0.0"))
VLLM_FORMAT_RETRY_ATTEMPTS = int(os.environ.get("VLLM_FORMAT_RETRY_ATTEMPTS", "3"))
PDF_RENDER_SCALE = float(os.environ.get("PDF_RENDER_SCALE", "1"))

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
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty model output")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_json_block(cleaned)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON output format") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Output JSON must be an object")
    if not parsed:
        raise ValueError("Output JSON is empty")

    return parsed


def _build_vllm_payload(image_data_url: str, max_tokens: int) -> dict[str, Any]:
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
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def _validate_json_node(node: Any, path: str = "$") -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str) or not key.strip():
                return f"Invalid key at {path}"
            child_error = _validate_json_node(value, f"{path}.{key}")
            if child_error:
                return child_error
        return None

    if isinstance(node, list):
        for idx, item in enumerate(node):
            child_error = _validate_json_node(item, f"{path}[{idx}]")
            if child_error:
                return child_error
        return None

    if isinstance(node, (str, int, float, bool)) or node is None:
        return None

    return f"Unsupported value type at {path}: {type(node).__name__}"


def _has_meaningful_value(node: Any) -> bool:
    if isinstance(node, str):
        return bool(node.strip())
    if isinstance(node, dict):
        return any(_has_meaningful_value(value) for value in node.values())
    if isinstance(node, list):
        return any(_has_meaningful_value(item) for item in node)
    return node is not None


def _validate_output_schema(parsed_json: dict[str, Any]) -> str | None:
    if not isinstance(parsed_json, dict):
        return "Output JSON must be an object"
    if not parsed_json:
        return "Output JSON is empty"
    if "error" in parsed_json and len(parsed_json) == 1:
        return f"Output contains error only: {parsed_json.get('error')}"

    structural_error = _validate_json_node(parsed_json)
    if structural_error:
        return structural_error

    if not _has_meaningful_value(parsed_json):
        return "Output JSON has no meaningful values"

    return None


def _next_max_tokens(current_max_tokens: int) -> int:
    return max(current_max_tokens + 512, int(current_max_tokens * 1.5))


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
    attempts = max(1, VLLM_FORMAT_RETRY_ATTEMPTS)
    max_tokens = VLLM_MAX_TOKENS
    last_error = "Invalid JSON output format"

    for attempt in range(1, attempts + 1):
        payload = _build_vllm_payload(image_data_url, max_tokens)

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
        try:
            parsed_json = _safe_json_parse(raw_text)
            schema_error = _validate_output_schema(parsed_json)
            if schema_error is None:
                return parsed_json
            last_error = schema_error
        except ValueError as exc:
            last_error = str(exc)

        if attempt < attempts:
            max_tokens = _next_max_tokens(max_tokens)
            logger.warning(
                "[-] Wrong output format on attempt %d/%d (%s). Retrying with max_tokens=%d",
                attempt,
                attempts,
                last_error,
                max_tokens,
            )
            logger.warning("[-] Raw model response on attempt %d: %s", attempt, raw_text or "<empty>")

    return {"error": f"{last_error} after {attempts} attempts"}


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


def _cache_inference_results(temp_path: Path, original_filename: str | None) -> None:
    _clear_cache_dir()

    page_inputs = _iter_page_inputs(temp_path)
    original_stem = Path((original_filename or "document")).stem
    safe_stem = Path(original_stem).name
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

        out_path = JSON_FILES_CACHE_DIR / f"{safe_stem}_{page_idx}.json"
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
        await run_in_threadpool(_cache_inference_results, temp_path, file.filename)

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
