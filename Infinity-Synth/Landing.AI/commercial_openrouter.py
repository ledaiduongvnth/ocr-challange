import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from landingai_ade import LandingAIADE
from openai import OpenAI

# --- CONFIGURATION ---
# The SDK uses VISION_AGENT_API_KEY by default, but we pass it explicitly.
LANDING_AI_API_KEY = "OGZ3c201Nm1odjcxd3I3OHg4cWN5OmlFdnRlVXFZNk5maURMMk5KbHZPcWhxa0VsclpLdjZW"
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "landing-ai-ocr-pipeline")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LANDING_RAW_MD_SUBDIR = "md"
LANDING_RAW_JSON_SUBDIR = "json"

# Initialize clients.
landing_client = LandingAIADE(apikey=LANDING_AI_API_KEY)
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


class FatalPipelineError(RuntimeError):
    """Raised when the pipeline should stop immediately."""


def _flatten_error_chain(error: Exception) -> list[Exception]:
    """Collect exception + chained exceptions for better classification."""
    chain: list[Exception] = []
    seen: set[int] = set()
    current: Exception | None = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def is_fatal_api_error(error: Exception) -> bool:
    """Detect API/auth errors where retrying next files is pointless."""
    chain = _flatten_error_chain(error)
    message_blob = " | ".join(str(err) for err in chain).upper()

    # Common explicit status names.
    if any(
        token in message_blob
        for token in (
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "ACCESS_DENIED",
            "FORBIDDEN",
            "INVALID_AUTH",
            "UNAVAILABLE",
        )
    ):
        return True

    # API key/token problems from different providers.
    if any(token in message_blob for token in ("API KEY", "API_KEY", "TOKEN")) and any(
        keyword in message_blob for keyword in ("LEAK", "INVALID", "REVOK", "EXPIRE", "DISABLE", "MISSING", "NOT VALID")
    ):
        return True

    # Numeric auth status hints.
    if any(
        code in message_blob
        for code in (
            " 401 ",
            " 403 ",
            " 405 ",
            " 429 ",
            " 500 ",
            " 502 ",
            " 503 ",
            " 504 ",
            "CODE': 401",
            "CODE': 403",
            "CODE': 405",
            "CODE': 429",
            "CODE': 500",
            "CODE': 502",
            "CODE': 503",
            "CODE': 504",
            '"CODE": 401',
            '"CODE": 403',
            '"CODE": 405',
            '"CODE": 429',
            '"CODE": 500',
            '"CODE": 502',
            '"CODE": 503',
            '"CODE": 504',
        )
    ):
        return True

    # Structured status code hints on exception objects.
    for err in chain:
        for attr in ("status_code", "code", "http_status"):
            value = getattr(err, attr, None)
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric in (401, 403, 405, 429, 500, 502, 503, 504):
                return True

    return False


def _get_mime_type(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")


def upload_to_gemini(file_path: Path) -> dict[str, Any]:
    """
    Prepare a local file as a multimodal content block for OpenRouter.
    Kept with the same function name to keep this file close to commercial.py.
    """
    print(f"[*] Uploading {file_path} to OpenRouter...")
    file_bytes = file_path.read_bytes()
    encoded = base64.b64encode(file_bytes).decode("ascii")
    mime_type = _get_mime_type(file_path)
    data_url = f"data:{mime_type};base64,{encoded}"
    print("[+] Uploaded successfully.")
    if mime_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": data_url}}
    return {"type": "file", "file": {"filename": file_path.name, "file_data": data_url}}


def extract_markdown_with_landing_ai(file_path: Path) -> tuple[str, Any]:
    """Run OCR on a document using Landing AI and return markdown + full response."""
    print(f"[*] Sending {file_path} to Landing AI (dpt-2-latest) for OCR...")
    response = landing_client.parse(document=file_path, model="dpt-2-latest")
    print("[+] OCR extraction complete.")
    return response.markdown, response


def _extract_openrouter_text(response: Any) -> str:
    """Extract text content robustly from OpenRouter chat completion response."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


def _parse_json_text(response_text: str) -> dict:
    """Parse JSON even when the model wraps output in markdown fences."""
    candidate = response_text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for marker in ("{", "["):
            idx = candidate.find(marker)
            if idx == -1:
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise

    return parsed


def convert_to_json_multimodal(
    target_file_path: Path,
    target_ocr_text: str,
    example_file_path: Path,
    prompts_dir: Path,
) -> dict:
    """Generate strict JSON from OCR text + source document using OpenRouter Gemini."""
    print("[*] Preparing multimodal prompt for Gemini...")

    system_rules = (prompts_dir / "system_rules.txt").read_text(encoding="utf-8")
    example_ocr = (prompts_dir / "pvcombank_example_ocr.txt").read_text(encoding="utf-8")
    example_json = (prompts_dir / "pvcombank_example_json.json").read_text(encoding="utf-8")

    gemini_example_file = upload_to_gemini(example_file_path)
    gemini_target_file = upload_to_gemini(target_file_path)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "### ONE-SHOT EXAMPLE ###\nHere is the reference document image/PDF:"},
        gemini_example_file,
        {"type": "text", "text": f"\nHere is the Landing AI OCR output for the reference document:\n{example_ocr}"},
        {"type": "text", "text": f"\nHere is the exact Target JSON Output you must generate based on the rules:\n{example_json}"},
        {
            "type": "text",
            "text": "\n### YOUR TURN ###\nNow apply the exact same logic. Here is the target document image/PDF you need to process:",
        },
        gemini_target_file,
        {"type": "text", "text": f"\nHere is the Landing AI OCR output for the target document:\n{target_ocr_text}"},
        {
            "type": "text",
            "text": "\nBased on the rules, the example, the target image, and the target OCR, generate the strict JSON output.",
        },
    ]

    print("[*] Generating JSON structure with Gemini 2.5 Flash...")
    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": user_content},
    ]
    request_headers = {
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_NAME,
    }
    response = openrouter_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.1,
        extra_headers=request_headers,
    )

    response_text = _extract_openrouter_text(response)
    try:
        parsed_json = _parse_json_text(response_text)
        print("[+] JSON structuring complete.")
        return parsed_json
    except json.JSONDecodeError as error:
        print(f"[-] Gemini failed to return valid JSON: {response_text}")
        raise error


def _to_jsonable(value: Any) -> Any:
    """Best-effort converter for arbitrary SDK objects into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "dict"):
        try:
            return _to_jsonable(value.dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        return {key: _to_jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}

    return repr(value)


def get_landing_raw_dirs(landing_raw_dir: Path) -> tuple[Path, Path]:
    """Return subfolders for markdown and JSON Landing AI raw artifacts."""
    markdown_dir = landing_raw_dir / LANDING_RAW_MD_SUBDIR
    response_json_dir = landing_raw_dir / LANDING_RAW_JSON_SUBDIR
    return markdown_dir, response_json_dir


def get_landing_raw_paths(target_file: Path, landing_raw_dir: Path) -> tuple[Path, Path]:
    """Return markdown and full-response artifact paths using subfolder layout."""
    markdown_dir, response_json_dir = get_landing_raw_dirs(landing_raw_dir)
    raw_markdown_path = markdown_dir / f"{target_file.stem}.md"
    raw_response_path = response_json_dir / f"{target_file.stem}.landing.json"
    return raw_markdown_path, raw_response_path


def save_landing_raw_output(
    target_file: Path,
    landing_markdown: str,
    landing_response: Any,
    landing_raw_dir: Path,
) -> tuple[Path, Path]:
    """Persist markdown and full Landing AI response for later inspection/debug."""
    raw_markdown_path, raw_response_path = get_landing_raw_paths(target_file=target_file, landing_raw_dir=landing_raw_dir)
    raw_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
    raw_markdown_path.write_text(landing_markdown, encoding="utf-8")

    serializable_response = _to_jsonable(landing_response)
    raw_response_path.write_text(json.dumps(serializable_response, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVED] Landing AI markdown: {raw_markdown_path}")
    print(f"[SAVED] Landing AI full response: {raw_response_path}")
    return raw_markdown_path, raw_response_path


def get_output_paths(target_file: Path, output_dir: Path, landing_raw_dir: Path) -> tuple[Path, Path, Path]:
    """Return all generated artifact paths for one target input."""
    prediction_json_path = output_dir / f"{target_file.stem}.json"
    raw_markdown_path, raw_response_path = get_landing_raw_paths(target_file=target_file, landing_raw_dir=landing_raw_dir)
    return prediction_json_path, raw_markdown_path, raw_response_path


def should_skip_target(target_file: Path, output_dir: Path, landing_raw_dir: Path) -> bool:
    """Skip work for this input if label output already exists."""
    prediction_json_path, raw_markdown_path, raw_response_path = get_output_paths(
        target_file=target_file,
        output_dir=output_dir,
        landing_raw_dir=landing_raw_dir,
    )
    # Legacy flat files are still accepted for resume behavior after layout changes.
    legacy_markdown_path = landing_raw_dir / f"{target_file.stem}.md"
    legacy_response_path = landing_raw_dir / f"{target_file.stem}.landing.json"
    has_current_md = raw_markdown_path.exists()
    has_current_json = raw_response_path.exists()
    has_legacy_md = legacy_markdown_path.exists()
    has_legacy_json = legacy_response_path.exists()

    if prediction_json_path.exists():
        md_status = (
            str(raw_markdown_path.relative_to(landing_raw_dir))
            if has_current_md
            else (f"{legacy_markdown_path.name} (legacy)" if has_legacy_md else "missing")
        )
        json_status = (
            str(raw_response_path.relative_to(landing_raw_dir))
            if has_current_json
            else (f"{legacy_response_path.name} (legacy)" if has_legacy_json else "missing")
        )
        print(
            "[SKIP] Label already exists for "
            f"{target_file.name}: {prediction_json_path.name} | raw_md={md_status} | raw_json={json_status}"
        )
        return True

    return False


def process_document(
    target_file: Path,
    example_file: Path,
    output_dir: Path,
    prompts_dir: Path,
    landing_raw_dir: Path,
) -> bool:
    """Process one target document and write JSON + raw Landing AI OCR output."""
    try:
        if should_skip_target(target_file=target_file, output_dir=output_dir, landing_raw_dir=landing_raw_dir):
            return True

        target_ocr, landing_response = extract_markdown_with_landing_ai(target_file)
        save_landing_raw_output(
            target_file=target_file,
            landing_markdown=target_ocr,
            landing_response=landing_response,
            landing_raw_dir=landing_raw_dir,
        )

        final_json = convert_to_json_multimodal(
            target_file_path=target_file,
            target_ocr_text=target_ocr,
            example_file_path=example_file,
            prompts_dir=prompts_dir,
        )

        output_path, _, _ = get_output_paths(target_file=target_file, output_dir=output_dir, landing_raw_dir=landing_raw_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_json, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[SUCCESS] Saved prediction to {output_path}")
        return True
    except Exception as error:  # noqa: BLE001
        if is_fatal_api_error(error):
            raise FatalPipelineError(f"Fatal API/auth error while processing {target_file}: {error}") from error
        print(f"[FAILED] Error processing {target_file}: {error}")
        return False


def list_target_files(target_dir: Path, recursive: bool) -> list[Path]:
    """Return supported document files in a folder."""
    iterator = target_dir.rglob("*") if recursive else target_dir.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def process_folder(
    target_dir: Path,
    example_file: Path,
    output_dir: Path,
    prompts_dir: Path,
    landing_raw_dir: Path,
    recursive: bool,
) -> tuple[int, int, int]:
    """Process every supported document in a folder."""
    all_targets = list_target_files(target_dir=target_dir, recursive=recursive)

    if not all_targets:
        print(f"[WARN] No supported files found in {target_dir}")
        return 0, 0, 0

    succeeded = 0
    failed = 0
    skipped = 0
    example_resolved = example_file.resolve()

    for target in all_targets:
        if target.resolve() == example_resolved:
            print(f"[SKIP] Skipping reference file: {target}")
            skipped += 1
            continue

        if process_document(target, example_file, output_dir, prompts_dir, landing_raw_dir):
            succeeded += 1
        else:
            failed += 1

    print(f"[DONE] Total: {len(all_targets)} | Success: {succeeded} | Failed: {failed} | Skipped: {skipped}")
    return succeeded, failed, skipped


def build_parser(default_base_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Landing AI OCR + Gemini JSON pipeline")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target-file", type=Path, help="Path to a single PDF/image file to process.")
    target_group.add_argument("--target-dir", type=Path, help="Path to a directory containing PDFs/images to process.")
    parser.add_argument(
        "--example-file",
        type=Path,
        default=default_base_dir / "images" / "pvcombank_example.pdf",
        help="Reference example document used for one-shot prompting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_base_dir / "pred",
        help="Directory where JSON output files will be saved.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=default_base_dir / "prompts",
        help="Directory containing system_rules and one-shot example prompt files.",
    )
    parser.add_argument(
        "--landing-raw-dir",
        type=Path,
        default=default_base_dir / "pred" / "landing_raw",
        help="Directory where raw Landing AI OCR markdown files will be saved.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively process subfolders of --target-dir.")
    return parser


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(default_base_dir=script_dir)
    args = parser.parse_args()

    example_file = args.example_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompts_dir = args.prompts_dir.expanduser().resolve()
    landing_raw_dir = args.landing_raw_dir.expanduser().resolve()

    if not example_file.exists():
        parser.error(f"Example file not found: {example_file}")
    if not prompts_dir.exists():
        parser.error(f"Prompts directory not found: {prompts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    landing_raw_dir.mkdir(parents=True, exist_ok=True)
    landing_markdown_dir, landing_response_json_dir = get_landing_raw_dirs(landing_raw_dir=landing_raw_dir)
    landing_markdown_dir.mkdir(parents=True, exist_ok=True)
    landing_response_json_dir.mkdir(parents=True, exist_ok=True)

    if args.target_file:
        target_file = args.target_file.expanduser().resolve()
        if not target_file.exists():
            parser.error(f"Target file not found: {target_file}")
        try:
            process_document(target_file, example_file, output_dir, prompts_dir, landing_raw_dir)
        except FatalPipelineError as fatal_error:
            print(f"[FATAL] {fatal_error}")
            raise SystemExit(1) from fatal_error
    else:
        target_dir = (args.target_dir or (script_dir / "images")).expanduser().resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            parser.error(f"Target directory not found: {target_dir}")
        try:
            process_folder(
                target_dir,
                example_file,
                output_dir,
                prompts_dir,
                landing_raw_dir,
                recursive=args.recursive,
            )
        except FatalPipelineError as fatal_error:
            print(f"[FATAL] {fatal_error}")
            raise SystemExit(1) from fatal_error
