import json
import re
import uuid
from logging import Logger
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from document_types import DEFAULT_DOCUMENT_TYPE_CODE, match_document_type_code


def success_response(**payload: Any) -> JSONResponse:
    """Create a consistent success response payload."""
    return JSONResponse(content={"status": "success", **payload})


def error_response(message: str, status_code: int = 500) -> JSONResponse:
    """Create a consistent error response payload."""
    return JSONResponse(content={"status": "error", "message": message}, status_code=status_code)


def safe_unlink(path: Path | None, description: str, logger: Logger) -> None:
    """Delete a file if it exists. Used in finally blocks for cleanup."""
    if not path:
        return

    try:
        if path.exists():
            path.unlink()
            logger.info(f"[*] Cleaned up {description}: {path}")
    except Exception as exc:
        logger.warning(f"[-] Failed to clean up {description}: {path} | {exc}")


async def save_upload_to_temp(file: UploadFile, tmp_dir: Path, logger: Logger) -> Path:
    """Save uploaded file to temp directory so request flow is traceable."""
    safe_filename = Path(file.filename or "upload.bin").name
    temp_path = tmp_dir / f"{uuid.uuid4()}_{safe_filename}"

    with temp_path.open("wb") as buffer:
        buffer.write(await file.read())

    logger.info(f"[*] Saved temporary file to {temp_path}")
    return temp_path


def get_cached_json_path(upload_filename: str, cache_dir: Path) -> Path:
    """Map uploaded filename to nested cache JSON path using filename stem."""
    safe_name = Path(upload_filename or "").name
    if not safe_name:
        raise ValueError("Uploaded file must include a valid filename")
    stem = Path(safe_name).stem
    base_stem, sep, tail = stem.rpartition("-")
    parent_dir = base_stem if sep and tail.isdigit() and base_stem else stem
    return cache_dir / parent_dir / f"{stem}.json"

def get_document_level_json_path(upload_filename: str, document_level_dir: Path) -> Path:
    """Map uploaded filename to document-level JSON path (flat directory)."""
    safe_name = Path(upload_filename or "").name
    if not safe_name:
        raise ValueError("Uploaded file must include a valid filename")
    stem = Path(safe_name).stem
    return document_level_dir / f"{stem}.json"



def _extract_raw_document_type(data: Any) -> str | None:
    if isinstance(data, dict):
        for key in ("document_type", "DocumentType", "Title", "title"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        # Handle wrapped outputs like {"result": {...}} from inference pipelines.
        for nested_key in ("result", "data"):
            nested_value = data.get(nested_key)
            nested_match = _extract_raw_document_type(nested_value)
            if nested_match:
                return nested_match

        return None

    if isinstance(data, list):
        for item in data:
            nested_match = _extract_raw_document_type(item)
            if nested_match:
                return nested_match

    return None


def extract_raw_document_type(data: Any) -> str | None:
    """Read document type candidates from cached JSON data."""
    return _extract_raw_document_type(data)


def resolve_document_type_code(raw_value: str | None) -> str:
    """Convert label/code text to canonical code (fallback: OTHER)."""
    return match_document_type_code(raw_value)


def _extract_page_number(stem: str) -> int | None:
    """Extract page number from filename stem when available.

    Examples:
      split_document_page_12 -> 12
      page-3 -> 3
      abc_7 -> 7
    """
    page_pattern = re.search(r"(?:^|[_-])page[_-]?(\d+)$", stem, flags=re.IGNORECASE)
    if page_pattern:
        return int(page_pattern.group(1))

    trailing_number = re.search(r"(?:[_-])(\d+)$", stem)
    if trailing_number:
        return int(trailing_number.group(1))

    return None


def _json_sort_key(path: Path) -> tuple[int, int, str]:
    page_number = _extract_page_number(path.stem)
    if page_number is None:
        return (1, 10**9, path.name.lower())
    return (0, page_number, path.name.lower())


def load_static_classification(cache_dir: Path, logger: Logger) -> list[dict[str, Any]]:
    """Build grouped classification from JSON files in cache directory.

    Rules:
    1) A page with explicit title/document_type starts a new segment (separator).
    2) A page without title inherits document_type from nearest previous titled page.
    3) Matching uses hard + soft rules via `match_document_type_code`.
    """
    if not cache_dir.exists():
        logger.warning(f"[-] Cache directory does not exist: {cache_dir}")
        return []

    json_paths = sorted(cache_dir.glob("*.json"), key=_json_sort_key)
    if not json_paths:
        return []

    page_rows: list[dict[str, Any]] = []
    for fallback_index, json_path in enumerate(json_paths):
        page_number = _extract_page_number(json_path.stem)
        page_index = fallback_index if page_number is None else page_number

        has_explicit_title = False
        document_type = DEFAULT_DOCUMENT_TYPE_CODE

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            raw_type = extract_raw_document_type(data)
            has_explicit_title = bool(raw_type and raw_type.strip())
            if has_explicit_title:
                document_type = resolve_document_type_code(raw_type)
        except Exception as exc:
            logger.warning(f"[-] Failed to read {json_path.name}: {exc}")

        page_rows.append(
            {
                "page_index": page_index,
                "document_type": document_type,
                "has_explicit_title": has_explicit_title,
            }
        )

    # Build segments: titled pages are separators; untitled pages inherit previous segment type.
    segments: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None

    for row in page_rows:
        page_index = row["page_index"]
        has_explicit_title = row["has_explicit_title"]
        row_type = row["document_type"]

        if current_segment is None:
            current_segment = {
                "index": len(segments),
                "document_type": row_type,
                "pages": [page_index],
            }
            segments.append(current_segment)
            continue

        if has_explicit_title:
            current_segment = {
                "index": len(segments),
                "document_type": row_type,
                "pages": [page_index],
            }
            segments.append(current_segment)
            continue

        # No title -> inherit type from nearest previous titled/active segment.
        current_segment["pages"].append(page_index)

    return segments
