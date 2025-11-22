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
) -> List:
    config = {"page_range": args.page_range} if args.page_range else {}
    images = loader(str(file_path), config)
    print(f"  -> {len(images)} page(s)")
    rotated_dir = args.output_dir / "rotated_pages" / file_path.stem
    images = normalize_page_images(images, save_dir=rotated_dir, prefix=file_path.stem)

    all_results = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        print(f"     batching pages {start + 1}-{end}")
        batch_items = [
            batch_input_cls(
                image=image,
                prompt_type="ocr_layout",
                prompt=base_prompt,
            )
            for image in images[start:end]
        ]
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(batch_items, **batch_kwargs)
        all_results.extend(results)

    log_component_bboxes(file_path.name, all_results)

    return all_results
