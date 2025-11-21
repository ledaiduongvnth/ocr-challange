#!/usr/bin/env python3
"""Run Chandra OCR with a lightweight orientation normalization pre-pass."""

from __future__ import annotations

import argparse
import os
import tempfile
from textwrap import dedent
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import pdfplumber
import numpy as np
from PIL import Image

BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chandra OCR runner with orientation detection.")
    parser.add_argument("input_path", type=Path, help="File or directory containing PDFs/images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./chandra_out"),
        help="Directory to store markdown/html/json outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="datalab-to/chandra",
        help="Hugging Face repo id to load (default: datalab-to/chandra).",
    )
    parser.add_argument(
        "--method",
        choices=("hf", "vllm"),
        default="vllm",
        help="Inference backend. Use 'hf' for local GPU, 'vllm' for server.",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Subset of pages for PDFs, e.g. '1-3,5'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Pages per batch (default: 1).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override max generated tokens per page.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers when using vllm backend.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retry count for vllm backend.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        default=False,
        help="Store cropped images referenced in output markdown.",
    )
    parser.add_argument(
        "--include-headers-footers",
        action="store_true",
        default=False,
        help="Keep detected headers/footers in the output.",
    )
    parser.add_argument("--no-html", action="store_true", help="Skip writing HTML alongside markdown.")
    parser.add_argument(
        "--paginate-output",
        action="store_true",
        help="Insert page separators into markdown/html.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. 'cuda:0'.")
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Optional attention implementation (flash_attention_2, etc.).",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["MODEL_CHECKPOINT"] = args.checkpoint
    if args.device:
        os.environ["TORCH_DEVICE"] = args.device
    if args.attn_impl:
        os.environ["TORCH_ATTN"] = args.attn_impl


def build_generate_kwargs(args: argparse.Namespace) -> dict:
    generate_kwargs = {
        "include_images": args.include_images,
        "include_headers_footers": args.include_headers_footers,
    }
    if args.max_output_tokens is not None:
        generate_kwargs["max_output_tokens"] = args.max_output_tokens
    if args.method == "vllm":
        if args.max_workers is not None:
            generate_kwargs["max_workers"] = args.max_workers
        if args.max_retries is not None:
            generate_kwargs["max_retries"] = args.max_retries
    return generate_kwargs


def is_pdf_native(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".pdf":
        return False
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            if not pdf.pages:
                return False
            text = (pdf.pages[0].extract_text() or "").strip()
            return bool(text)
    except Exception:
        return False


_PADDLE_ORIENTATION_MODEL = None
_PADDLE_ORIENTATION_ERROR: Optional[Exception] = None


def _get_paddle_orientation_model():
    global _PADDLE_ORIENTATION_MODEL, _PADDLE_ORIENTATION_ERROR
    if _PADDLE_ORIENTATION_ERROR is not None:
        return None
    if _PADDLE_ORIENTATION_MODEL is None:
        try:
            from paddleocr import PPStructureV3

            _PADDLE_ORIENTATION_MODEL = PPStructureV3(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
            )
            print("Loaded Paddle document orientation classifier.")
        except Exception as exc:  # pragma: no cover - optional dependency
            _PADDLE_ORIENTATION_ERROR = exc
            print(f"     Paddle orientation classifier unavailable: {exc}")
            return None
    return _PADDLE_ORIENTATION_MODEL


def _compute_projection_metrics(image: Image.Image) -> Tuple[float, float, float, float]:
    """Return (horizontal_variance, vertical_variance, top-bottom balance, left-right balance)."""
    arr = np.array(image.convert("L"))
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = cv2.GaussianBlur(arr, (5, 5), 0)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - thresh  # make text areas high-valued
    horiz = ink.sum(axis=1).astype(np.float64)
    vert = ink.sum(axis=0).astype(np.float64)
    height, width = ink.shape
    horiz_var = float(np.var(horiz) / (height * height + 1e-6))
    vert_var = float(np.var(vert) / (width * width + 1e-6))
    band = max(1, height // 4)
    top = float(horiz[:band].sum())
    bottom = float(horiz[-band:].sum())
    side_band = max(1, width // 4)
    left = float(vert[:side_band].sum())
    right = float(vert[-side_band:].sum())
    return horiz_var, vert_var, top - bottom, left - right


def _extract_angle_from_result(res) -> Optional[int]:
    info = getattr(res, "doc_preprocessor_res", None)
    if info is None and hasattr(res, "to_dict"):
        info = res.to_dict().get("doc_preprocessor_res")
    if info is None and isinstance(res, dict):
        info = res.get("doc_preprocessor_res")
    if info is None:
        return None
    if hasattr(info, "to_dict"):
        info = info.to_dict()
    angle = None
    if isinstance(info, dict):
        angle = info.get("angle")
    else:
        angle = getattr(info, "angle", None)
    if angle is None:
        return None
    try:
        return int(round(float(angle)))
    except (TypeError, ValueError):
        return None


def detect_orientation_with_paddle(image: Image.Image) -> Optional[int]:
    model = _get_paddle_orientation_model()
    if model is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        preds = model.predict(input=tmp_path)
    except Exception as exc:  # pragma: no cover - external dependency
        print(f"     Paddle orientation detection failed: {exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if not preds:
        return None
    angle = _extract_angle_from_result(preds[0])
    return angle


def _select_best_rotation(
    image: Image.Image, angles: Sequence[int], preferred: Set[int]
) -> Tuple[Image.Image, int]:
    best_angle = 0
    best_image = image
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    seen = set()
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


def normalize_orientation(image: Image.Image) -> Tuple[Image.Image, int]:
    """Rotate the image (0/90/180/270 CCW) so that text lines run horizontally."""
    paddle_angle = detect_orientation_with_paddle(image)
    if paddle_angle is not None:
        normalized = paddle_angle % 360
        print(f"     Paddle doc orientation angle: {normalized}°")
        rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
        return rotated, normalized
    return _select_best_rotation(image, (0, 90, 180, 270), set())


def normalize_pages(
    images: Iterable[Image.Image],
    save_dir: Path | None = None,
    prefix: str = "page",
) -> List[Image.Image]:
    normalized = []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images, 1):
        fixed, angle = normalize_orientation(image)
        if angle:
            print(f"     page {idx}: rotated {angle}° CCW to fix orientation")
        else:
            print(f"     page {idx}: orientation OK")
        if save_dir:
            out_path = save_dir / f"{prefix}_{idx:03d}.png"
            fixed.save(out_path)
            print(f"        saved rotated page -> {out_path}")
        normalized.append(fixed)
    return normalized


def print_component_bboxes(file_name: str, results: List) -> None:
    """Log component bounding boxes (tables, text blocks, etc.) per page."""
    print(f"  components for {file_name}:")
    for page_idx, result in enumerate(results, 1):
        chunks = getattr(result, "chunks", None)
        if not chunks:
            continue
        for comp_idx, chunk in enumerate(chunks, 1):
            comp_type = (
                chunk.get("type")
                or chunk.get("label")
                or chunk.get("category")
                or "unknown"
            )
            bbox = chunk.get("bbox") or chunk.get("box") or chunk.get("page_box")
            print(f"    page {page_idx} #{comp_idx}: {comp_type} bbox={bbox}")


CUSTOM_PROMPT_SUFFIX = dedent(
    """\
    Note:
    - If a cell in the table is empty (no text or empty string), keep the cell in the markdown.
    - Header or footer lines can stick together; remember to separate them.
    """
)


def run():
    args = parse_args()
    configure_environment(args)

    from chandra.input import load_file
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.scripts.cli import get_supported_files, save_merged_output
    from chandra.prompts import PROMPT_MAPPING

    files: List[Path] = get_supported_files(args.input_path)
    if not files:
        raise SystemExit(f"No supported files found under {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running Chandra ({args.method}) with checkpoint '{args.checkpoint}' on {len(files)} file(s)...")

    inference = InferenceManager(method=args.method)
    generate_kwargs = build_generate_kwargs(args)
    base_prompt = f"{PROMPT_MAPPING['ocr_layout']}{CUSTOM_PROMPT_SUFFIX}"

    def process_file(file_path: Path) -> None:
        if file_path.suffix.lower() == ".pdf":
            pdf_type = "native (digital)" if is_pdf_native(file_path) else "scanned (image-based)"
            print(f"  PDF type: {pdf_type}")
        else:
            print("  Input type: image (treated as scanned)")

        config = {"page_range": args.page_range} if args.page_range else {}
        images = load_file(str(file_path), config)
        print(f"  -> {len(images)} page(s)")
        rotated_dir = args.output_dir / "rotated_pages" / file_path.stem
        images = normalize_pages(images, save_dir=rotated_dir, prefix=file_path.stem)

        all_results = []
        for start in range(0, len(images), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(images))
            print(f"     batching pages {start + 1}-{end}")
            batch_items = [
                BatchInputItem(
                    image=image,
                    prompt_type="ocr_layout",
                    prompt=base_prompt,
                )
                for image in images[start:end]
            ]
            results = inference.generate(batch_items, **generate_kwargs)
            print(results)
            all_results.extend(results)

        print_component_bboxes(file_path.name, all_results)
        save_merged_output(
            args.output_dir,
            file_path.name,
            all_results,
            save_images=args.include_images,
            save_html=not args.no_html,
            paginate_output=args.paginate_output,
        )

    for idx, file_path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {file_path.name}")
        process_file(file_path)


if __name__ == "__main__":
    run()
