from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from components import log_component_bboxes

try:
    from paddlex import create_model
except Exception:  # pragma: no cover - optional dependency
    create_model = None


def _load_pp_doclayout(model_name: str = "PP-DocLayout-L"):
    if create_model is None:
        raise ImportError(
            "paddlex is not installed. Install PaddleX 3.0rc1+ to use PP-DocLayout."
        )
    return create_model(model_name)


def _predict_layout(model, images: Sequence[Image.Image]) -> List:
    results = []
    for img in images:
        np_img = np.array(img.convert("RGB"))
        preds = model.predict(np_img)
        # model.predict may yield generator/custom objects; coerce to list of raw dicts
        page_entries = []
        try:
            iterable_preds = list(preds)
        except TypeError:
            iterable_preds = [preds]

        for p in iterable_preds:
            raw = p.res if hasattr(p, "res") else p
            if not raw:
                continue
            if isinstance(raw, dict) and "boxes" in raw:
                page_entries.extend(raw["boxes"])
            elif isinstance(raw, dict):
                page_entries.append(raw)
        results.append(page_entries)
    return results


def _to_chunks(pred) -> List[dict]:
    chunks = []
    if not pred:
        return chunks
    # pred is a list of dict entries from PP-DocLayout standard_data
    for box_info in pred:
        box = (
            box_info.get("coordinate")
            or box_info.get("bbox")
            or box_info.get("points")
        )
        label = (
            box_info.get("label")
            or box_info.get("transcription")
            or box_info.get("type")
            or "unknown"
        )
        if not box or len(box) < 4:
            continue
        x0, y0, x1, y1 = map(float, box[:4])
        chunks.append({"bbox": [x0, y0, x1, y1], "label": str(label)})
    return chunks


def analyze_layout_pp_doclayout(
    file_path: Path,
    images: Sequence[Image.Image],
    model_name: str = "PP-DocLayout-L",
    debug_dir: Path | None = None,
) -> Tuple[List[Image.Image], List]:
    """
    Run layout detection using PP-DocLayout.

    Inputs:
        file_path: source file path (for logging/debug names)
        images: page images to analyze (pre-rendered)
        model_name: PP-DocLayout model to load (default: PP-DocLayout-L)
        debug_dir: optional directory to save annotated debug images

    Outputs:
        (images, layout_results) where layout_results mirrors the input pages and contains chunk bboxes.
    """
    assert images, "pp-doclayout analysis requires at least one page image"
    print(f"  [pp-doclayout] -> {len(images)} page(s)")
    model = _load_pp_doclayout(model_name)
    preds = _predict_layout(model, images)

    layout_results = []
    for pred in preds:
        layout_results.append(type("LayoutResult", (object,), {"chunks": _to_chunks(pred)})())

    assert len(layout_results) == len(images), "layout results must match number of pages"
    log_component_bboxes(file_path.name, layout_results)

    if debug_dir:
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
                draw.rectangle((x0, y0, x1, y1), outline="blue", width=2)
                label = chunk.get("label") or "unknown"
                draw.text((x0 + 2, y0 + 2), label, fill="blue")
            out_path = debug_dir / f"{file_path.stem}_ppdoc_layout_{page_idx:03d}.png"
            annotated.save(out_path)
            print(f"     [pp-doclayout] saved debug image -> {out_path}")

    return list(images), layout_results
