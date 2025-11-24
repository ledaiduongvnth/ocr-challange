from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

def _to_processed_image(result, fallback_image: Image.Image) -> Image.Image:
    """Extract processed image from doc_preprocessor output; fallback to original."""
    if result is None:
        return fallback_image
    # Custom object with attribute
    processed = getattr(result, "processed_image", None) or getattr(
        result, "image", None
    )
    if processed is None and isinstance(result, dict):
        processed = (
            result.get("processed_image")
            or result.get("image")
            or result.get("img")
        )
    if processed is None:
        return fallback_image
    try:
        if isinstance(processed, Image.Image):
            return processed
        arr = np.array(processed)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr)
    except Exception:
        pass
    return fallback_image


def preprocess_with_ppstructure(
    images: Sequence[Image.Image],
    use_orientation: bool = True,
    use_unwarp: bool = True,
    debug_dir: Path | None = None,
) -> List[Image.Image]:
    """Best-effort preprocessing via PP-StructureV3 doc_preprocessor (orientation/unwarp)."""
    try:
        from paddleocr._pipelines.doc_preprocessor import DocPreprocessor
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"DocPreprocessor unavailable ({exc}); skipping preprocessing.")
        return list(images)

    preprocessor = DocPreprocessor(
        use_doc_orientation_classify=use_orientation,
        use_doc_unwarping=use_unwarp,
    )

    processed_images: List[Image.Image] = []
    for idx, img in enumerate(images, 1):
        np_img = np.array(img.convert("RGB"))
        try:
            results = preprocessor.predict([np_img])
        except Exception as exc:
            print(f"DocPreprocessor failed on page {idx} ({exc}); using original.")
            processed_images.append(img)
            continue

        if not results:
            processed_images.append(img)
            continue

        processed = _to_processed_image(results[0], img)
        processed_images.append(processed)

        if debug_dir:
            try:
                page_dir = debug_dir / f"{idx:03d}" / "debug_layout"
                page_dir.mkdir(parents=True, exist_ok=True)
                out_path = page_dir / "preprocessed.png"
                processed.save(out_path)
                print(f"  [pp-structure] saved preprocessed page -> {out_path}")
            except Exception:
                pass

    return processed_images
