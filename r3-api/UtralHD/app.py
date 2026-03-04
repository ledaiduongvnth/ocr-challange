import base64
import io
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set, Tuple
from urllib import error, request

import fitz
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile

from app_utils import (
    error_response,
    get_document_level_json_path,
    load_static_classification,
    safe_unlink,
    save_upload_to_temp,
    success_response,
)


def _get_positive_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return default

    return parsed if parsed > 0 else default


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default

    return parsed if parsed > 0 else default


# API returns JSON cached by filename in this directory.
DEFAULT_CACHE_DIR = "/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/json_files_cache"
DEFAULT_LABELS_DIR = "/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/labels"
DEFAULT_DOCUMENT_LEVEL_DIR = "/media/drive-2t/hoangnv83/code/ocr/ocr-challange-duong/r3-api/UtralHD/document_level_json"
JSON_FILES_CACHE_DIR = Path(os.environ.get("JSON_FILES_CACHE_DIR", DEFAULT_CACHE_DIR))
LABELS_DIR = Path(os.environ.get("LABELS_DIR", DEFAULT_LABELS_DIR))
DOCUMENT_LEVEL_JSON_DIR = Path(os.environ.get("DOCUMENT_LEVEL_JSON_DIR", DEFAULT_DOCUMENT_LEVEL_DIR))
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
# 1.0 scale ~= 72 DPI in PDF user-space units.
# 2.0 scale (~144 DPI) is a stronger default for OCR while still practical for vLLM.
PDF_RENDER_SCALE = _get_positive_float_env("PDF_RENDER_SCALE", 2.2)
PDF_RENDER_MAX_SIDE_PX = _get_positive_int_env("PDF_RENDER_MAX_SIDE_PX", 2500)
PDF_RENDER_MAX_PIXELS = _get_positive_int_env("PDF_RENDER_MAX_PIXELS", 4_800_000)

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
logger.setLevel(logging.INFO)

app = FastAPI(title="IDP Extraction & Classification API")

_PADDLE_ORIENTATION_MODEL = None
_PADDLE_ORIENTATION_ERROR: Optional[Exception] = None


def _get_paddle_orientation_model():
    global _PADDLE_ORIENTATION_MODEL, _PADDLE_ORIENTATION_ERROR

    if _PADDLE_ORIENTATION_ERROR is not None:
        return None

    if _PADDLE_ORIENTATION_MODEL is None:
        try:
            from paddleocr import DocImgOrientationClassification

            _PADDLE_ORIENTATION_MODEL = DocImgOrientationClassification(
                model_name="PP-LCNet_x1_0_doc_ori",
                device="cpu",
            )
            logger.info("[*] Loaded Paddle DocImgOrientationClassification on CPU")
        except Exception as exc:  # pragma: no cover - optional dependency
            _PADDLE_ORIENTATION_ERROR = exc
            logger.warning("[-] Paddle orientation classifier unavailable: %s", exc)
            return None

    return _PADDLE_ORIENTATION_MODEL


def _parse_angle_like(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            angle = _parse_angle_like(item)
            if angle is not None:
                return angle
        return None

    parsed: Optional[int]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None

        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            parsed = int(stripped)
        else:
            digits = "".join(ch for ch in stripped if ch.isdigit() or ch == "-")
            if not digits or digits == "-":
                return None
            try:
                parsed = int(digits)
            except ValueError:
                return None
    else:
        try:
            parsed = int(round(float(value)))
        except (TypeError, ValueError):
            return None

    normalized = parsed % 360
    if normalized in (0, 90, 180, 270):
        return normalized

    return None


def _parse_orientation_class_id(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            angle = _parse_orientation_class_id(item)
            if angle is not None:
                return angle
        return None

    try:
        class_id = int(round(float(value)))
    except (TypeError, ValueError):
        return None

    if 0 <= class_id <= 3:
        return class_id * 90

    return None


def _extract_angle_from_paddle_result(result: Any) -> Optional[int]:
    if result is None:
        return None

    if isinstance(result, list):
        for item in result:
            angle = _extract_angle_from_paddle_result(item)
            if angle is not None:
                return angle
        return None

    payload = result.to_dict() if hasattr(result, "to_dict") else result

    if isinstance(payload, dict):
        legacy_info = payload.get("doc_preprocessor_res")
        if legacy_info is not None:
            info = legacy_info.to_dict() if hasattr(legacy_info, "to_dict") else legacy_info
            if isinstance(info, dict):
                angle = _parse_angle_like(info.get("angle"))
            else:
                angle = _parse_angle_like(getattr(info, "angle", None))
            if angle is not None:
                return angle

        for key in ("angle", "label_name", "label", "label_names", "labels"):
            angle = _parse_angle_like(payload.get(key))
            if angle is not None:
                return angle

        for key in ("class_id", "class_ids"):
            angle = _parse_orientation_class_id(payload.get(key))
            if angle is not None:
                return angle

        for key in ("result", "results", "res", "output", "pred", "prediction"):
            if key in payload:
                angle = _extract_angle_from_paddle_result(payload[key])
                if angle is not None:
                    return angle

    angle = _parse_angle_like(getattr(result, "angle", None))
    if angle is not None:
        return angle

    angle = _parse_angle_like(getattr(result, "label_name", None))
    if angle is not None:
        return angle

    angle = _parse_angle_like(getattr(result, "label_names", None))
    if angle is not None:
        return angle

    angle = _parse_orientation_class_id(getattr(result, "class_id", None))
    if angle is not None:
        return angle

    angle = _parse_orientation_class_id(getattr(result, "class_ids", None))
    if angle is not None:
        return angle

    return None


def detect_orientation_with_paddle(image: Image.Image) -> Optional[int]:
    model = _get_paddle_orientation_model()
    if model is None:
        return None

    temp_file_path = None
    started = time.perf_counter()
    logger.info("[*] Running orientation detection on image %dx%d", image.width, image.height)

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            temp_file_path = temp_file.name

        raw_predictions = model.predict(input=temp_file_path)
        predictions = list(raw_predictions) if raw_predictions is not None else []
    except Exception as exc:  # pragma: no cover - external dependency
        logger.warning("[-] Paddle orientation detection failed: %s", exc)
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

    elapsed = time.perf_counter() - started
    if not predictions:
        logger.info("[*] Orientation detection completed in %.2fs with no prediction", elapsed)
        return None

    for prediction in predictions:
        angle = _extract_angle_from_paddle_result(prediction)
        if angle is not None:
            logger.info("[*] Orientation detection completed in %.2fs, angle=%d", elapsed, angle)
            return angle

    logger.info("[*] Orientation detection completed in %.2fs, angle unresolved", elapsed)
    return None


def _compute_projection_metrics(image: Image.Image) -> Tuple[float, float, float, float]:
    if np is None or cv2 is None:
        return 0.0, 0.0, 0.0, 0.0

    array = np.array(image.convert("L"))
    if array.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    array = cv2.GaussianBlur(array, (5, 5), 0)
    _, thresh = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - thresh

    horiz = ink.sum(axis=1).astype(np.float64)
    vert = ink.sum(axis=0).astype(np.float64)
    height, width = ink.shape

    horiz_var = float(np.var(horiz) / (height * height + 1e-6))
    vert_var = float(np.var(vert) / (width * width + 1e-6))

    band_h = max(1, height // 4)
    top = float(horiz[:band_h].sum())
    bottom = float(horiz[-band_h:].sum())

    band_w = max(1, width // 4)
    left = float(vert[:band_w].sum())
    right = float(vert[-band_w:].sum())

    return horiz_var, vert_var, top - bottom, left - right


def _select_best_rotation(
    image: Image.Image,
    angles: Sequence[int],
    preferred: Set[int],
) -> Tuple[Image.Image, int]:
    best_angle = 0
    best_image = image
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    seen: set[int] = set()

    for angle in angles:
        normalized = int(angle) % 360
        if normalized in seen:
            continue
        seen.add(normalized)

        rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
        horiz_var, vert_var, top_bottom, left_right = _compute_projection_metrics(rotated)

        alignment_score = horiz_var - vert_var
        rotation_penalty = -abs(normalized if normalized <= 180 else 360 - normalized)
        key = (
            alignment_score,
            1 if normalized in preferred else 0,
            left_right,
            top_bottom,
            rotation_penalty,
        )

        if key > best_key:
            best_key = key
            best_angle = normalized
            best_image = rotated

    return best_image, best_angle


def _normalize_orientation(image: Image.Image) -> Tuple[Image.Image, int]:
    paddle_angle = detect_orientation_with_paddle(image)
    if paddle_angle is not None:
        normalized = paddle_angle % 360
        rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
        return rotated, normalized

    # Fallback heuristic when Paddle orientation classifier is unavailable.
    if np is not None and cv2 is not None:
        return _select_best_rotation(image, (0, 90, 180, 270), set())

    return image, 0


def _pil_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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
    return max(current_max_tokens + 512, int(current_max_tokens * 1.1))


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


def _retry_scale_for_attempt(attempt: int) -> float:
    return 1.0 + 0.05 * max(0, attempt - 1)


def _scaled_image_for_attempt(base_image: Image.Image, attempt: int) -> tuple[Image.Image, float]:
    scale = _retry_scale_for_attempt(attempt)
    if abs(scale - 1.0) <= 1e-3:
        return base_image, 1.0

    new_width = max(1, int(round(base_image.width * scale)))
    new_height = max(1, int(round(base_image.height * scale)))

    if new_width == base_image.width and new_height == base_image.height:
        return base_image, 1.0

    resampling = getattr(Image, "Resampling", Image)
    resized = base_image.resize((new_width, new_height), resampling.LANCZOS)
    applied_scale_x = new_width / base_image.width
    applied_scale_y = new_height / base_image.height
    applied_scale = (applied_scale_x + applied_scale_y) / 2.0
    return resized, applied_scale


def _infer_page_json(base_image: Image.Image, page_label: str) -> dict[str, Any]:
    attempts = max(1, VLLM_FORMAT_RETRY_ATTEMPTS)
    max_tokens = VLLM_MAX_TOKENS
    last_error = "Invalid JSON output format"

    for attempt in range(1, attempts + 1):
        image_for_attempt, image_scale = _scaled_image_for_attempt(base_image, attempt)
        image_data_url = _pil_image_to_data_url(image_for_attempt)
        payload = _build_vllm_payload(image_data_url, max_tokens)

        logger.info(
            "[*] Sending %s to vLLM (attempt=%d/%d, size=%dx%d, scale=%.1f, max_tokens=%d)",
            page_label,
            attempt,
            attempts,
            image_for_attempt.width,
            image_for_attempt.height,
            image_scale,
            max_tokens,
        )
        request_started = time.perf_counter()

        try:
            resp_json = _post_vllm(payload)
            logger.info(
                "[*] vLLM response received for %s (attempt=%d/%d) in %.2fs",
                page_label,
                attempt,
                attempts,
                time.perf_counter() - request_started,
            )
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
            next_scale = _retry_scale_for_attempt(attempt + 1)
            logger.warning(
                "[-] Wrong output format on attempt %d/%d (%s). Retrying with max_tokens=%d, image_scale=%.1f",
                attempt,
                attempts,
                last_error,
                max_tokens,
                next_scale,
            )
            logger.warning(
                "[-] Raw model response on attempt %d (image_scale=%.1f): %s",
                attempt,
                image_scale,
                raw_text or "<empty>",
            )

    return {"error": f"{last_error} after {attempts} attempts"}


def _load_image_for_inference(image_path: Path) -> Image.Image:
    logger.info("[*] Preparing image %s for inference", image_path.name)
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        normalized_image, angle = _normalize_orientation(image)
        if angle:
            logger.info("[*] Rotated image %s by %d degrees before inference", image_path.name, angle)
        return normalized_image.copy()


def _iter_pdf_page_images(pdf_path: Path) -> Iterable[tuple[int, Image.Image]]:
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            logger.info("[*] Preparing PDF page %d for inference", page_idx)
            requested_scale = PDF_RENDER_SCALE
            page_width = max(1.0, float(page.rect.width))
            page_height = max(1.0, float(page.rect.height))
            effective_scale = requested_scale

            side_scale_limit = PDF_RENDER_MAX_SIDE_PX / max(page_width, page_height)
            pixel_scale_limit = (PDF_RENDER_MAX_PIXELS / (page_width * page_height)) ** 0.5
            effective_scale = min(effective_scale, side_scale_limit, pixel_scale_limit)
            effective_scale = max(0.1, effective_scale)

            effective_dpi = 72.0 * effective_scale
            if effective_scale + 1e-6 < requested_scale:
                logger.info(
                    "[*] PDF page %d render scale capped from %.2f to %.2f (dpi=%.0f, max_side=%d, max_pixels=%d)",
                    page_idx,
                    requested_scale,
                    effective_scale,
                    effective_dpi,
                    PDF_RENDER_MAX_SIDE_PX,
                    PDF_RENDER_MAX_PIXELS,
                )
            else:
                logger.info(
                    "[*] PDF page %d render scale=%.2f (dpi=%.0f)",
                    page_idx,
                    effective_scale,
                    effective_dpi,
                )

            pix = page.get_pixmap(matrix=fitz.Matrix(effective_scale, effective_scale), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            normalized_image, angle = _normalize_orientation(image)
            if angle:
                logger.info("[*] Rotated PDF page %d by %d degrees before inference", page_idx, angle)
            yield page_idx, normalized_image


def _iter_page_inputs(temp_path: Path) -> Iterable[tuple[int, Image.Image]]:
    suffix = temp_path.suffix.lower()
    if suffix == ".pdf":
        return _iter_pdf_page_images(temp_path)

    if suffix in IMAGE_EXTENSIONS:
        return [(0, _load_image_for_inference(temp_path))]

    raise ValueError(f"Unsupported upload format: '{suffix}'. Use PDF or image file.")


def _extract_page_number_from_stem(stem: str) -> int | None:
    page_pattern = re.search(r"(?:^|[_-])page[_-]?(\d+)$", stem, flags=re.IGNORECASE)
    if page_pattern:
        return int(page_pattern.group(1))

    trailing_number = re.search(r"(?:[_-])(\d+)$", stem)
    if trailing_number:
        return int(trailing_number.group(1))

    return None


def _page_json_sort_key(path: Path) -> tuple[int, int, str]:
    page_number = _extract_page_number_from_stem(path.stem)
    if page_number is None:
        return (1, 10**9, path.name.lower())
    return (0, page_number, path.name.lower())


def _is_freetext_like_key(key: str) -> bool:
    # Accept FreeText1, FreeText1_1, FreeText1_1_1, ...
    return bool(re.fullmatch(r"FreeText(?:\d+)?(?:_\d+)*", key))


def _merge_page_jsons(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = value
            continue

        existing = target[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_page_jsons(existing, value)
            continue

        if isinstance(existing, list) and isinstance(value, list):
            existing.extend(value)
            continue

        # Keep every FreeText fragment even if repeated (important for long-table page splits).
        if existing == value and not _is_freetext_like_key(key):
            continue

        duplicate_idx = 1
        duplicate_key = f"{key}_{duplicate_idx}"
        while duplicate_key in target:
            duplicate_idx += 1
            duplicate_key = f"{key}_{duplicate_idx}"
        target[duplicate_key] = value


def _is_table_rows(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, dict) for row in value)


def _normalize_header_name(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.lower())


def _is_implicit_column_headers(headers: list[str]) -> bool:
    return all(re.fullmatch(r"column\d+", header.lower()) for header in headers)


def _build_table_merge_plan(
    track_headers: list[str],
    new_headers: list[str],
) -> tuple[float, dict[str, str]] | None:
    if not track_headers or not new_headers:
        return None

    if track_headers == new_headers:
        return 1.0, {header: header for header in new_headers}

    track_implicit = _is_implicit_column_headers(track_headers)
    new_implicit = _is_implicit_column_headers(new_headers)

    if len(track_headers) == len(new_headers):
        if track_implicit and new_implicit:
            return 0.90, {src: dst for src, dst in zip(new_headers, track_headers)}

        if (not track_implicit) and new_implicit:
            return 0.88, {src: dst for src, dst in zip(new_headers, track_headers)}

        if track_implicit and (not new_implicit):
            return 0.82, {src: dst for src, dst in zip(new_headers, track_headers)}

    # Some OCR pages lose one trailing table column on continuation pages.
    # If one side is implicit ColumnN headers and the structures are near-equal,
    # align by column order and merge as the same running table.
    if abs(len(track_headers) - len(new_headers)) <= 1:
        min_cols = min(len(track_headers), len(new_headers))
        if min_cols >= 3:
            if track_implicit and new_implicit:
                return 0.86, {
                    src: track_headers[idx]
                    for idx, src in enumerate(new_headers[:min_cols])
                }

            if (not track_implicit) and new_implicit:
                return 0.84, {
                    src: track_headers[idx]
                    for idx, src in enumerate(new_headers[:min_cols])
                }

            if track_implicit and (not new_implicit):
                return 0.80, {
                    src: track_headers[idx]
                    for idx, src in enumerate(new_headers[:min_cols])
                }

    normalized_track = {_normalize_header_name(header): header for header in track_headers}
    mapping: dict[str, str] = {}
    for src in new_headers:
        normalized_src = _normalize_header_name(src)
        dst = normalized_track.get(normalized_src)
        if dst:
            mapping[src] = dst

    overlap = len(mapping)
    if overlap == 0:
        return None

    coverage = overlap / max(1, len(track_headers))
    if coverage < 0.60:
        return None

    score = 0.70 + 0.20 * min(1.0, coverage)
    return score, mapping


def _append_table_rows(
    table_rows: list[dict[str, Any]],
    incoming_rows: list[dict[str, Any]],
    mapping: dict[str, str],
    ordered_headers: list[str],
) -> None:
    for row in incoming_rows:
        if not isinstance(row, dict):
            continue

        remapped: dict[str, Any] = {header: "" for header in ordered_headers}
        for src_key, src_value in row.items():
            dst_key = mapping.get(src_key)
            if dst_key is None:
                continue
            remapped[dst_key] = src_value

        table_rows.append(remapped)


def _allocate_unique_object_key(merged_result: dict[str, Any], base_key: str) -> str:
    if base_key not in merged_result:
        return base_key

    duplicate_idx = 1
    duplicate_key = f"{base_key}_{duplicate_idx}"
    while duplicate_key in merged_result:
        duplicate_idx += 1
        duplicate_key = f"{base_key}_{duplicate_idx}"
    return duplicate_key


def _merge_page_with_table_tracks(
    merged_result: dict[str, Any],
    table_tracks: list[dict[str, Any]],
    page_json: dict[str, Any],
    page_idx: int,
) -> None:
    non_table_payload: dict[str, Any] = {}
    continuation_track_idx: int | None = None

    for key, value in page_json.items():
        if not _is_table_rows(value):
            non_table_payload[key] = value
            continue

        headers = list(value[0].keys())
        best_track_idx: int | None = None
        best_plan: tuple[float, dict[str, str]] | None = None
        best_score = -1.0

        for track_idx, track in enumerate(table_tracks):
            if page_idx - track["last_page"] not in (0, 1):
                continue

            plan = _build_table_merge_plan(track["headers"], headers)
            if plan is None:
                continue

            score = plan[0]
            if page_idx - track["last_page"] == 1:
                score += 0.05
            if track["key"] == key:
                score += 0.03

            if score > best_score:
                best_score = score
                best_track_idx = track_idx
                best_plan = plan

        if best_track_idx is None or best_plan is None:
            initial_rows = [dict(row) for row in value if isinstance(row, dict)]
            output_key = _allocate_unique_object_key(merged_result, key)
            merged_result[output_key] = initial_rows
            table_tracks.append(
                {
                    "key": key,
                    "output_key": output_key,
                    "headers": headers,
                    "rows": initial_rows,
                    "last_page": page_idx,
                    "first_page": page_idx,
                }
            )
            continue

        track = table_tracks[best_track_idx]
        _, mapping = best_plan
        _append_table_rows(track["rows"], value, mapping, track["headers"])

        # Mark this page as table-continuation if the track already existed before.
        if page_idx > track.get("first_page", page_idx):
            continuation_track_idx = best_track_idx

        track["last_page"] = page_idx

    if non_table_payload:
        freetext_payload = {k: v for k, v in non_table_payload.items() if _is_freetext_like_key(k)}
        other_payload = {k: v for k, v in non_table_payload.items() if not _is_freetext_like_key(k)}

        if continuation_track_idx is not None and freetext_payload:
            logger.info("[*] Dropped FreeText on page %d because it is between merged table parts", page_idx)
        elif freetext_payload:
            _merge_page_jsons(merged_result, freetext_payload)

        if other_payload:
            _merge_page_jsons(merged_result, other_payload)


def _flush_table_tracks_into_result(merged_result: dict[str, Any], table_tracks: list[dict[str, Any]]) -> None:
    # Tables are inserted at first-seen position during merge to preserve object order.
    _ = (merged_result, table_tracks)



def _indexed_group_name(key: str) -> str | None:
    for group in ("FreeText", "Table", "Section"):
        # Accept keys like FreeText, FreeText1, FreeText1_1, Table2, Section3_2.
        if re.fullmatch(rf"{group}(?:\d+)?(?:_\d+)*", key):
            return group
    return None


def _normalize_group_index_keys(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_group_index_keys(item) for item in node]

    if not isinstance(node, dict):
        return node

    counters = {"FreeText": 0, "Table": 0, "Section": 0}
    normalized: dict[str, Any] = {}

    for key, value in node.items():
        normalized_value = _normalize_group_index_keys(value)
        group = _indexed_group_name(key)

        if group is None:
            new_key = key
        else:
            counters[group] += 1
            new_key = f"{group}{counters[group]}"

        if new_key in normalized:
            duplicate_idx = 1
            candidate = f"{new_key}_{duplicate_idx}"
            while candidate in normalized:
                duplicate_idx += 1
                candidate = f"{new_key}_{duplicate_idx}"
            new_key = candidate

        normalized[new_key] = normalized_value

    return normalized


def _reorder_document_level_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep FreeText after table blocks in final document-level JSON."""
    other_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Any]] = []
    freetext_items: list[tuple[str, Any]] = []

    for key, value in payload.items():
        if _is_freetext_like_key(key):
            freetext_items.append((key, value))
            continue

        if _indexed_group_name(key) == "Table" or _is_table_rows(value):
            table_items.append((key, value))
            continue

        other_items.append((key, value))

    reordered: dict[str, Any] = {}
    for key, value in [*other_items, *table_items, *freetext_items]:
        reordered[key] = value

    return reordered


def _save_document_level_results(
    request_cache_dir: Path,
    original_filename: str | None,
    class_array: list[dict[str, Any]],
) -> None:
    DOCUMENT_LEVEL_JSON_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(Path(original_filename or "document").stem).name
    json_paths = sorted(request_cache_dir.glob("*.json"), key=_page_json_sort_key)

    page_data_by_index: dict[int, dict[str, Any]] = {}
    for fallback_index, json_path in enumerate(json_paths):
        page_number = _extract_page_number_from_stem(json_path.stem)
        page_index = fallback_index if page_number is None else page_number

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("[-] Failed to load page JSON for document-level merge (%s): %s", json_path.name, exc)
            continue

        if isinstance(data, dict):
            page_data_by_index[page_index] = data

    valid_doc_items = [
        doc_item
        for doc_item in class_array
        if isinstance(doc_item.get("index"), int) and isinstance(doc_item.get("pages"), list)
    ]
    single_document_mode = len(valid_doc_items) == 1

    for doc_item in valid_doc_items:
        doc_index = doc_item.get("index")
        pages = doc_item.get("pages", [])

        merged_result: dict[str, Any] = {}
        table_tracks: list[dict[str, Any]] = []
        for page_idx in pages:
            page_json = page_data_by_index.get(page_idx)
            if page_json is None:
                logger.warning("[-] Missing page JSON for merge: doc_index=%s page=%s", doc_index, page_idx)
                continue
            _merge_page_with_table_tracks(merged_result, table_tracks, page_json, page_idx)

        _flush_table_tracks_into_result(merged_result, table_tracks)
        normalized_result = _normalize_group_index_keys(merged_result)
        out_name = f"{safe_stem}.json" if single_document_mode else f"{safe_stem}-{doc_index}.json"
        out_path = DOCUMENT_LEVEL_JSON_DIR / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(normalized_result, f, ensure_ascii=False, indent=2)
        logger.info("[*] Saved document-level JSON: %s", out_path.name)


def _ensure_output_dirs() -> None:
    JSON_FILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENT_LEVEL_JSON_DIR.mkdir(parents=True, exist_ok=True)


def _cache_inference_results(temp_path: Path, original_filename: str | None) -> Path:
    _ensure_output_dirs()

    page_inputs = _iter_page_inputs(temp_path)
    original_stem = Path((original_filename or "document")).stem
    safe_stem = Path(original_stem).name
    request_cache_dir = JSON_FILES_CACHE_DIR / safe_stem
    request_cache_dir.mkdir(parents=True, exist_ok=True)
    for old_json in request_cache_dir.glob("*.json"):
        safe_unlink(old_json, "stale per-file cache JSON file", logger)

    is_pdf_input = temp_path.suffix.lower() == ".pdf"
    total_pdf_pages = 0
    if is_pdf_input:
        with fitz.open(temp_path) as doc:
            total_pdf_pages = doc.page_count

    page_count = 0
    success_count = 0
    logger.info("[*] Running vLLM inference and caching page outputs...")

    for page_idx, page_image in page_inputs:
        page_count += 1
        page_label = f"{safe_stem}_page_{page_idx}"
        logger.info("[*] Starting inference for %s", page_label)

        try:
            parsed_json = _infer_page_json(page_image, page_label)
            if "error" not in parsed_json:
                success_count += 1
        except Exception as exc:
            logger.error(f"[-] Page {page_idx} inference failed: {exc}")
            parsed_json = {"error": str(exc)}

        out_name = f"{safe_stem}-{page_idx}.json" if (is_pdf_input and total_pdf_pages > 1) else f"{safe_stem}.json"
        out_path = request_cache_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)

        labels_out_path = LABELS_DIR / out_name
        if labels_out_path != out_path:
            with labels_out_path.open("w", encoding="utf-8") as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            logger.info(f"[*] Saved label JSON: {labels_out_path.name}")

        logger.info(f"[*] Cached page JSON: {out_path.name}")

    if page_count == 0:
        raise RuntimeError("No pages found for inference")
    if success_count == 0:
        raise RuntimeError("Inference failed for all pages")

    return request_cache_dir


@app.post("/classification")
async def classification_endpoint(file: UploadFile = File(...)):
    """Run OCR inference on uploaded PDF/image and return grouped classification."""
    logger.info(f"========== NEW CLASSIFICATION REQUEST: {file.filename} ==========")
    temp_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        request_cache_dir = _cache_inference_results(temp_path, file.filename)

        class_array = load_static_classification(request_cache_dir, logger)
        page_count = sum(len(item.get("pages", [])) for item in class_array)
        _save_document_level_results(request_cache_dir, file.filename, class_array)

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
    """Return cached JSON by uploaded filename.

    `document_type` is kept for API compatibility with existing clients.
    """
    logger.info(f"========== NEW EXTRACTION REQUEST: {file.filename} | TYPE: {document_type} ==========")
    temp_path: Path | None = None
    cache_json_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp(file, TMP_DIR, logger)
        cache_json_path = get_document_level_json_path(file.filename or "", DOCUMENT_LEVEL_JSON_DIR)
        logger.info(f"[*] Loading document-level JSON: {cache_json_path}")

        if not cache_json_path.exists():
            stem = Path(Path(file.filename or "").name).stem
            candidates = sorted(DOCUMENT_LEVEL_JSON_DIR.glob(f"{stem}-*.json"))
            if len(candidates) == 1:
                cache_json_path = candidates[0]
                logger.info(f"[*] Fallback to single matched document-level JSON: {cache_json_path}")
            else:
                return error_response(
                    f"Document-level JSON not found for file '{file.filename}'",
                    status_code=404,
                )

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
