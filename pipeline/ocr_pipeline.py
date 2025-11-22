from __future__ import annotations

from pathlib import Path
from typing import List, Callable

from chandra.model.schema import BatchInputItem

from orientation import normalize_page_images
from components import log_component_bboxes


def run_ocr_pipeline(
    file_path: Path,
    args,
    inference,
    generate_kwargs: dict,
    base_prompt: str,
    batch_size: int,
    loader: Callable,
    batch_input_cls=BatchInputItem,
    images: List | None = None,
    layout_results: List | None = None,
    ) -> List:
    if images is None:
        config = {"page_range": args.page_range} if args.page_range else {}
        images = loader(str(file_path), config)
        print(f"  -> {len(images)} page(s)")
        rotated_dir = args.output_dir / "rotated_pages" / file_path.stem
        # TODO
        # images = normalize_page_images(images, save_dir=rotated_dir, prefix=file_path.stem)
    else:
        print(f"  -> using preloaded images ({len(images)} page(s))")

    all_results = []
    assert layout_results is not None and len(layout_results) == len(
        images
    ), "layout_results must be provided and match number of pages"

    # Recognize each detected component individually using cropped regions.
    component_items = []
    for page_idx, layout in enumerate(layout_results, 0):
        chunks = getattr(layout, "chunks", None) or []
        page_image = images[page_idx]
        for chunk in chunks:
            bbox = chunk.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                x0, y0, x1, y1 = [max(0, float(v)) for v in bbox[:4]]
            except Exception:
                continue
            x0, y0 = int(x0), int(y0)
            x1, y1 = int(x1), int(y1)
            # Clamp bounds
            x0 = max(0, min(x0, page_image.width))
            y0 = max(0, min(y0, page_image.height))
            x1 = max(x0 + 1, min(x1, page_image.width))
            y1 = max(y0 + 1, min(y1, page_image.height))
            cropped = page_image.crop((x0, y0, x1, y1))
            component_items.append(
                batch_input_cls(
                    image=cropped,
                    prompt_type="ocr_layout",
                    prompt=base_prompt,
                )
            )

    print(f"     batching {len(component_items)} detected components for OCR")
    for start in range(0, len(component_items), batch_size):
        end = min(start + batch_size, len(component_items))
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(component_items[start:end], **batch_kwargs)
        all_results.extend(results)

    return all_results
