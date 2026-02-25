#!/usr/bin/env python3
"""Convert all PDFs in a folder into page images with sanitized file names."""

from __future__ import annotations

import argparse
import re
import subprocess
import unicodedata
import uuid
from pathlib import Path


def sanitize_name(name: str) -> str:
    """Keep only Ubuntu-safe ASCII characters and replace the rest with '_'."""
    # Normalize Vietnamese D-stroke explicitly before ASCII folding.
    normalized = unicodedata.normalize("NFKD", name.replace("Đ", "D").replace("đ", "d"))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "file"


def make_unique_base(base: str, used: dict[str, int]) -> str:
    """Avoid collisions after sanitization."""
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count + 1}"


def convert_pdf(pdf_path: Path, output_dir: Path, safe_base: str, dpi: int) -> int:
    """Convert one PDF to PNG files and return page count."""
    temp_prefix = output_dir / f"__tmp__{uuid.uuid4().hex}"
    command = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(temp_prefix),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    pattern = re.compile(rf"^{re.escape(temp_prefix.name)}-(\d+)\.png$")
    temp_images: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            temp_images.append((int(match.group(1)), path))
    temp_images.sort(key=lambda item: item[0])

    for page_no, temp_image in temp_images:
        final_name = f"{safe_base}_page_{page_no:04d}.png"
        final_path = output_dir / final_name
        temp_image.replace(final_path)

    return len(temp_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split all PDFs in a folder into per-page images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/pdf"),
        help="Folder that contains PDF files (default: data/pdf)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pdf_images"),
        help="Folder to write output images (default: data/pdf_images)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Image resolution in DPI (default: 200)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return 0

    used_names: dict[str, int] = {}
    total_pages = 0
    failures = 0

    for pdf_path in pdf_files:
        safe_base = make_unique_base(sanitize_name(pdf_path.stem), used_names)
        try:
            pages = convert_pdf(pdf_path, output_dir, safe_base, args.dpi)
            total_pages += pages
            print(f"[OK] {pdf_path.name} -> {pages} pages (prefix: {safe_base})")
        except subprocess.CalledProcessError as exc:
            failures += 1
            stderr = exc.stderr.strip() if exc.stderr else "No error output."
            print(f"[ERROR] {pdf_path.name}: {stderr}")
        except Exception as exc:  # noqa: BLE001 - keep going for other files
            failures += 1
            print(f"[ERROR] {pdf_path.name}: {exc}")

    print(
        f"Done. PDFs: {len(pdf_files)}, Pages: {total_pages}, "
        f"Success: {len(pdf_files) - failures}, Failed: {failures}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
