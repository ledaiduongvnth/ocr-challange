from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from PIL import Image, ImageDraw

from utils import log_component_bboxes


def _load_surya_predictor():
    try:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "surya is not installed. Install surya to use the surya layout backend."
        ) from exc
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )
    return layout_predictor


def _to_chunks(preds) -> List[dict]:
    chunks: List[dict] = []
    # If Surya returns an object with .bboxes, unwrap it
    if hasattr(preds, "bboxes"):
        preds = getattr(preds, "bboxes") or []
    if preds is None:
        return chunks
    seq = preds if isinstance(preds, (list, tuple)) else [preds]
    for pred in seq:
        bbox = None
        label = "unknown"
        score = None

        if hasattr(pred, "__dict__"):
            bbox = (
                getattr(pred, "bbox", None)
                or getattr(pred, "box", None)
                or getattr(pred, "coordinate", None)
                or getattr(pred, "points", None)
            )
            label = (
                getattr(pred, "label", None)
                or getattr(pred, "type", None)
                or getattr(pred, "category", None)
                or "unknown"
            )
            score = getattr(pred, "confidence", None) or getattr(pred, "score", None)
        elif isinstance(pred, dict):
            bbox = (
                pred.get("bbox")
                or pred.get("box")
                or pred.get("coordinate")
                or pred.get("points")
            )
            label = pred.get("label") or pred.get("type") or pred.get("category") or "unknown"
            score = pred.get("score") or pred.get("confidence")
        elif isinstance(pred, (list, tuple)) and len(pred) >= 4:
            bbox = pred[:4]

        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = map(float, bbox[:4])
        chunk: dict[str, Any] = {"bbox": [x0, y0, x1, y1], "label": str(label)}
        if score is not None:
            chunk["score"] = float(score)
        chunks.append(chunk)
    return chunks


def analyze_layout_surya(
    file_path: Path,
    images: Sequence[Image.Image],
    debug_dir: Path | None = None,
) -> Tuple[List[Image.Image], List]:
    """
    Run layout detection using surya LayoutPredictor.

    Inputs:
        file_path: source file path
        images: pre-rendered page images
        debug_dir: optional directory to save annotated debug images
    """
    assert images, "surya layout analysis requires at least one page image"
    print(f"  [layout] backend: surya -> {len(images)} page(s)")
    predictor = _load_surya_predictor()

    raw_preds = predictor(list(images)) or []
    layout_results = []
    for idx in range(len(images)):
        page_pred = raw_preds[idx] if idx < len(raw_preds) else []
        chunks = _to_chunks(page_pred)
        layout_results.append(type("LayoutResult", (object,), {"chunks": chunks})())

    log_component_bboxes(file_path.name, layout_results)

    if debug_dir:
        for page_idx, (img, layout) in enumerate(zip(images, layout_results), 1):
            annotated = img.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for chunk in getattr(layout, "chunks", None) or []:
                bbox = chunk.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[:4]
                draw.rectangle((x0, y0, x1, y1), outline="green", width=2)
                label = chunk.get("label") or "unknown"
                draw.text((x0 + 2, y0 + 2), label, fill="green")
            page_dir = debug_dir / f"{page_idx:03d}" / "debug_layout"
            page_dir.mkdir(parents=True, exist_ok=True)
            out_path = page_dir / f"{file_path.stem}_surya_layout.png"
            annotated.save(out_path)
            print(f"     [surya] saved debug image -> {out_path}")

    return list(images), layout_results
