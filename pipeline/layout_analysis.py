from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

from chandra.model.schema import BatchInputItem

from components import log_component_bboxes
from orientation import normalize_page_images
from PIL import ImageDraw


def analyze_layout(
    file_path: Path,
    args,
    inference,
    generate_kwargs: dict,
    base_prompt: str,
    batch_size: int,
    loader: Callable,
) -> Tuple[List, List]:
    """Run a lightweight layout pass to get component bounding boxes."""
    config = {"page_range": args.page_range} if args.page_range else {}
    images = loader(str(file_path), config)
    print(f"  [layout] -> {len(images)} page(s)")
    rotated_dir = args.output_dir / "rotated_pages" / file_path.stem
    # TODO
    # images = normalize_page_images(images, save_dir=rotated_dir, prefix=f"{file_path.stem}_layout")

    layout_results = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        print(f"     [layout] batching pages {start + 1}-{end}")
        batch_items = [
            BatchInputItem(
                image=image,
                prompt_type="ocr_layout",
                prompt=base_prompt,
            )
            for image in images[start:end]
        ]
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(batch_items, **batch_kwargs)
        layout_results.extend(results)

    log_component_bboxes(file_path.name, layout_results)

    # Optional: save images with overlaid layout boxes for debugging.
    debug_dir = args.output_dir / "debug_layout"
    debug_dir.mkdir(parents=True, exist_ok=True)
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
        out_path = debug_dir / f"{file_path.stem}_layout_{page_idx:03d}.png"
        annotated.save(out_path)
        print(f"     [layout] saved debug image -> {out_path}")

    return images, layout_results
