from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from chandra.model.schema import BatchInputItem
from PIL import Image, ImageDraw

from orientation import normalize_page_images
from utils import log_component_bboxes


def chandra_analyze_layout(
    file_path: Path,
    images: Sequence[Image.Image],
    infer_fn: Callable[[Sequence[BatchInputItem]], Sequence],
    prompt: str | None,
    batch_size: int,
    debug_dir: Path | None = None,
) -> Tuple[List[Image.Image], List]:
    """
    Run a layout pass to get component bounding boxes.

    Inputs:
        file_path: source file path (for logging/debug names)
        images: page images to analyze (pre-rendered)
        infer_fn: callable that takes a sequence of BatchInputItem and returns layout results
        prompt: prompt string to use for layout model
        batch_size: batch size for inference
        debug_dir: optional directory to save annotated debug images

    Outputs:
        (images, layout_results) where layout_results mirrors the input pages and contains chunk bboxes.
    """
    if prompt is None:
        from chandra.prompts import PROMPT_MAPPING

        prompt = PROMPT_MAPPING["ocr_layout"]
    print(f"  [layout] -> {len(images)} page(s)")
    layout_results: List = []
    assert images, "layout analysis requires at least one page image"
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        print(f"     [layout] batching pages {start + 1}-{end}")
        batch_items = [
            BatchInputItem(
                image=image,
                prompt_type="ocr_layout",
                prompt=prompt,
            )
            for image in images[start:end]
        ]
        results = infer_fn(batch_items)
        layout_results.extend(results)

    assert len(layout_results) == len(images), "layout results must match number of pages"
    log_component_bboxes(file_path.name, layout_results)

    if debug_dir:
        for page_idx, (img, layout) in enumerate(zip(images, layout_results), 1):
            annotated = img.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            chunks = getattr(layout, "chunks", None) or []
            for chunk in chunks:
                bbox = chunk.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[:4]
                draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
                label = chunk.get("label") or chunk.get("type") or "unknown"
                draw.text((x0 + 2, y0 + 2), label, fill="red")
            page_dir = debug_dir / f"{page_idx:03d}" / "debug_layout"
            page_dir.mkdir(parents=True, exist_ok=True)
            out_path = page_dir / f"{file_path.stem}_layout.png"
            annotated.save(out_path)
            print(f"     [layout] saved debug image -> {out_path}")

    return list(images), layout_results
