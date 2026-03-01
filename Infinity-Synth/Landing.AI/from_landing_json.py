import argparse
import json
from pathlib import Path

from commercial import FatalPipelineError
from commercial import SUPPORTED_EXTENSIONS
from commercial import convert_to_json_multimodal
from commercial import is_fatal_api_error


def derive_source_stem(landing_json_file: Path) -> str:
    """Map '<name>_parse_output.json' -> '<name>' for output/image lookup."""
    stem = landing_json_file.stem
    suffix = "_parse_output"
    suffix2 = ".landing"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem[: -len(suffix2)] if stem.endswith(suffix2) else stem


def extract_markdown_from_landing_json(landing_payload: dict) -> str:
    """Extract markdown text from Landing AI full response JSON."""
    top_markdown = landing_payload.get("markdown")
    if isinstance(top_markdown, str) and top_markdown.strip():
        return top_markdown

    result = landing_payload.get("result")
    if isinstance(result, dict):
        result_markdown = result.get("markdown")
        if isinstance(result_markdown, str) and result_markdown.strip():
            return result_markdown

        pages = result.get("pages")
        if isinstance(pages, list):
            page_markdowns = [
                page.get("markdown", "")
                for page in pages
                if isinstance(page, dict) and isinstance(page.get("markdown"), str)
            ]
            merged = "\n\n".join(chunk for chunk in page_markdowns if chunk.strip())
            if merged.strip():
                return merged

    chunks = landing_payload.get("chunks")
    if isinstance(chunks, list):
        chunk_markdowns = [
            chunk.get("markdown", "")
            for chunk in chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("markdown"), str)
        ]
        merged = "\n\n".join(chunk for chunk in chunk_markdowns if chunk.strip())
        if merged.strip():
            return merged

    # Generic fallback: search the whole payload for markdown-looking fields.
    fallback_markdowns = list(iter_markdown_values(landing_payload))
    if fallback_markdowns:
        merged = "\n\n".join(fallback_markdowns)
        if merged.strip():
            return merged

    raise ValueError("Could not find markdown content in Landing AI JSON response.")


def iter_markdown_values(value: object):
    """Yield non-empty markdown strings found in nested dict/list payloads."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key.lower() == "markdown" and item.strip():
                yield item
            else:
                yield from iter_markdown_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_markdown_values(item)


def find_target_image(images_dir: Path, source_stem: str) -> Path:
    """Resolve image path matching the source stem."""
    for extension in sorted(SUPPORTED_EXTENSIONS):
        candidate = images_dir / f"{source_stem}{extension}"
        if candidate.exists():
            return candidate

    matches = sorted(
        path
        for path in images_dir.glob(f"{source_stem}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"[WARN] Multiple image matches for {source_stem}; using first: {matches[0].name}")
        return matches[0]

    raise FileNotFoundError(f"No image found for '{source_stem}' under {images_dir}")


def process_one(
    landing_json_file: Path,
    images_dir: Path,
    markdown_dir: Path,
    label_dir: Path,
    example_file: Path,
    prompts_dir: Path,
) -> bool:
    """Build markdown + key-value label for one Landing JSON file."""
    source_stem = derive_source_stem(landing_json_file)
    markdown_output_path = markdown_dir / f"{source_stem}.md"
    label_output_path = label_dir / f"{source_stem}.json"

    if label_output_path.exists():
        print(f"[SKIP] Label already exists: {label_output_path}")
        return True

    try:
        payload = json.loads(landing_json_file.read_text(encoding="utf-8"))
        markdown_text = extract_markdown_from_landing_json(payload)
    except Exception as error:  # noqa: BLE001
        print(f"[FAILED] Unable to extract markdown from {landing_json_file}: {error}")
        return False

    if not markdown_output_path.exists():
        markdown_output_path.write_text(markdown_text, encoding="utf-8")
        print(f"[SAVED] Markdown: {markdown_output_path}")

    try:
        target_image = find_target_image(images_dir=images_dir, source_stem=source_stem)
        final_json = convert_to_json_multimodal(
            target_file_path=target_image,
            target_ocr_text=markdown_text,
            example_file_path=example_file,
            prompts_dir=prompts_dir,
        )
        label_output_path.write_text(json.dumps(final_json, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[SAVED] Label: {label_output_path}")
        return True
    except Exception as error:  # noqa: BLE001
        if is_fatal_api_error(error):
            raise FatalPipelineError(f"Fatal API/auth error while processing {landing_json_file}: {error}") from error
        print(f"[FAILED] Error processing {landing_json_file}: {error}")
        return False


def build_parser(default_base_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate markdown + key-value labels from Landing AI full-response JSON files."
    )
    parser.add_argument(
        "--landing-json-dir",
        type=Path,
        default=Path("/media/hdd01/PycharmProjects/ocr-challange/data/1030/landing/json"),
        help="Directory containing Landing AI full JSON responses (*_parse_output.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Directory containing source images; defaults to sibling path '<dataset>/images'.",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        help="Directory to save markdown labels; defaults to '<dataset>/landing/md'.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        help="Directory to save key-value labels; defaults to '<dataset>/label'.",
    )
    parser.add_argument(
        "--example-file",
        type=Path,
        default=default_base_dir / "images" / "pvcombank_example.pdf",
        help="Reference example document used for one-shot prompting.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=default_base_dir / "prompts",
        help="Directory containing system_rules and one-shot prompt files.",
    )
    return parser


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = build_parser(default_base_dir=script_dir)
    args = parser.parse_args()

    landing_json_dir = args.landing_json_dir.expanduser().resolve()
    if not landing_json_dir.exists() or not landing_json_dir.is_dir():
        parser.error(f"Landing JSON directory not found: {landing_json_dir}")

    dataset_root = landing_json_dir.parent.parent
    images_dir = (args.images_dir or (dataset_root / "images")).expanduser().resolve()
    markdown_dir = (args.markdown_dir or (dataset_root / "landing" / "md")).expanduser().resolve()
    label_dir = (args.label_dir or (dataset_root / "label")).expanduser().resolve()
    example_file = args.example_file.expanduser().resolve()
    prompts_dir = args.prompts_dir.expanduser().resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        parser.error(f"Images directory not found: {images_dir}")
    if not example_file.exists():
        parser.error(f"Example file not found: {example_file}")
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        parser.error(f"Prompts directory not found: {prompts_dir}")

    markdown_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    landing_json_files = sorted(path for path in landing_json_dir.glob("*.json") if path.is_file())
    if not landing_json_files:
        print(f"[WARN] No JSON files found in {landing_json_dir}")
        return 0

    success = 0
    failed = 0
    try:
        for landing_json_file in landing_json_files:
            if process_one(
                landing_json_file=landing_json_file,
                images_dir=images_dir,
                markdown_dir=markdown_dir,
                label_dir=label_dir,
                example_file=example_file,
                prompts_dir=prompts_dir,
            ):
                success += 1
            else:
                failed += 1
    except FatalPipelineError as fatal_error:
        print(f"[FATAL] {fatal_error}")
        return 1

    print(f"[DONE] Total: {len(landing_json_files)} | Success: {success} | Failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
