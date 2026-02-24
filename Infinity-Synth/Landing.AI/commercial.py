import argparse
import json
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from landingai_ade import LandingAIADE

# --- CONFIGURATION ---
# The SDK uses VISION_AGENT_API_KEY by default, but we pass it explicitly.
LANDING_AI_API_KEY = ""
GEMINI_API_KEY = ""
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Initialize clients.
landing_client = LandingAIADE(apikey=LANDING_AI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def upload_to_gemini(file_path: Path):
    """Upload a file to Gemini and wait until processing is complete."""
    print(f"[*] Uploading {file_path} to Gemini...")
    uploaded_file = gemini_client.files.upload(file=str(file_path))

    while uploaded_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        uploaded_file = gemini_client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed for {file_path}.")

    print("\n[+] Uploaded successfully.")
    return uploaded_file


def extract_markdown_with_landing_ai(file_path: Path) -> tuple[str, Any]:
    """Run OCR on a document using Landing AI and return markdown + full response."""
    print(f"[*] Sending {file_path} to Landing AI (dpt-2-latest) for OCR...")
    response = landing_client.parse(document=file_path, model="dpt-2-latest")
    print("[+] OCR extraction complete.")
    return response.markdown, response


def convert_to_json_multimodal(
    target_file_path: Path,
    target_ocr_text: str,
    example_file_path: Path,
    prompts_dir: Path,
) -> dict:
    """Generate strict JSON from OCR text + source document using Gemini."""
    print("[*] Preparing multimodal prompt for Gemini...")

    system_rules = (prompts_dir / "system_rules.txt").read_text(encoding="utf-8")
    example_ocr = (prompts_dir / "pvcombank_example_ocr.txt").read_text(encoding="utf-8")
    example_json = (prompts_dir / "pvcombank_example_json.json").read_text(encoding="utf-8")

    gemini_example_file = upload_to_gemini(example_file_path)
    gemini_target_file = upload_to_gemini(target_file_path)

    prompt_contents = [
        "### ONE-SHOT EXAMPLE ###\nHere is the reference document image/PDF:",
        gemini_example_file,
        f"\nHere is the Landing AI OCR output for the reference document:\n{example_ocr}",
        f"\nHere is the exact Target JSON Output you must generate based on the rules:\n{example_json}",
        "\n### YOUR TURN ###\nNow apply the exact same logic. Here is the target document image/PDF you need to process:",
        gemini_target_file,
        f"\nHere is the Landing AI OCR output for the target document:\n{target_ocr_text}",
        "\nBased on the rules, the example, the target image, and the target OCR, generate the strict JSON output.",
    ]

    print("[*] Generating JSON structure with Gemini 2.5 Flash...")
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_rules,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    try:
        parsed_json = json.loads(response.text)
        print("[+] JSON structuring complete.")
        return parsed_json
    except json.JSONDecodeError as error:
        print(f"[-] Gemini failed to return valid JSON: {response.text}")
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


def save_landing_raw_output(
    target_file: Path,
    landing_markdown: str,
    landing_response: Any,
    landing_raw_dir: Path,
) -> tuple[Path, Path]:
    """Persist markdown and full Landing AI response for later inspection/debug."""
    landing_raw_dir.mkdir(parents=True, exist_ok=True)

    raw_markdown_path = landing_raw_dir / f"{target_file.stem}.md"
    raw_markdown_path.write_text(landing_markdown, encoding="utf-8")

    raw_response_path = landing_raw_dir / f"{target_file.stem}.landing.json"
    serializable_response = _to_jsonable(landing_response)
    raw_response_path.write_text(json.dumps(serializable_response, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVED] Landing AI markdown: {raw_markdown_path}")
    print(f"[SAVED] Landing AI full response: {raw_response_path}")
    return raw_markdown_path, raw_response_path


def process_document(
    target_file: Path,
    example_file: Path,
    output_dir: Path,
    prompts_dir: Path,
    landing_raw_dir: Path,
) -> bool:
    """Process one target document and write JSON + raw Landing AI OCR output."""
    try:
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

        output_path = output_dir / f"{target_file.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_json, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[SUCCESS] Saved prediction to {output_path}")
        return True
    except Exception as error:  # noqa: BLE001
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

    if args.target_file:
        target_file = args.target_file.expanduser().resolve()
        if not target_file.exists():
            parser.error(f"Target file not found: {target_file}")
        process_document(target_file, example_file, output_dir, prompts_dir, landing_raw_dir)
    else:
        target_dir = (args.target_dir or (script_dir / "images")).expanduser().resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            parser.error(f"Target directory not found: {target_dir}")
        process_folder(
            target_dir,
            example_file,
            output_dir,
            prompts_dir,
            landing_raw_dir,
            recursive=args.recursive,
        )
