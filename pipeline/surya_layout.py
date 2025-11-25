from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from PIL import Image, ImageDraw

from utils import log_component_bboxes


def _load_marker_converter():
    try:
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        from marker.logger import configure_logging, get_logger
    except Exception as exc:
        raise ImportError(
            "marker is not available. Ensure marker_chandra-main is present and dependencies are installed."
        ) from exc

    configure_logging()
    get_logger()  # initialize logger
    models = create_model_dict()
    parser = ConfigParser({"output_format": "chunks"})
    converter_cls = parser.get_converter_cls()
    converter = converter_cls(
        config=parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )
    return converter


def _blocks_to_layouts(
    blocks, num_pages: int, images: Sequence[Image.Image], page_info: dict | None
) -> List:
    """Group marker blocks by page into simple LayoutResult-like objects."""
    layouts: List = [type("LayoutResult", (object,), {"chunks": []})() for _ in range(num_pages)]

    for blk in blocks or []:
        raw_id = getattr(blk, "id", "") or ""
        page_idx = getattr(blk, "page", 0) or 0
        block_idx = None
        try:
            if raw_id.startswith("/page/"):
                parts = raw_id.split("/")
                if len(parts) >= 4:
                    page_idx = int(parts[2])
                    block_idx = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else None
        except Exception:
            page_idx = getattr(blk, "page", 0) or 0
        if 0 <= page_idx < num_pages:
            img_width = images[page_idx].width if images else None
            img_height = images[page_idx].height if images else None
            page_dims = None
            if page_info and isinstance(page_info, dict):
                info = page_info.get(page_idx) or page_info.get(str(page_idx))
                if info:
                    bbox_info = info.get("bbox") or info.get("image_bbox")
                    if bbox_info and len(bbox_info) >= 4:
                        page_dims = (
                            float(bbox_info[2]) - float(bbox_info[0]),
                            float(bbox_info[3]) - float(bbox_info[1]),
                        )
            bbox = getattr(blk, "bbox", None)
            label = getattr(blk, "block_type", "") or "unknown"
            score = getattr(blk, "confidence", None) or getattr(blk, "score", None)
            if bbox and len(bbox) >= 4:
                x0, y0, x1, y1 = map(float, bbox[:4])
                if img_width and img_height and page_dims:
                    page_w, page_h = page_dims
                    scale_x = img_width / max(1.0, page_w)
                    scale_y = img_height / max(1.0, page_h)
                    x0 *= scale_x
                    x1 *= scale_x
                    y0 *= scale_y
                    y1 *= scale_y
                chunk: dict[str, Any] = {
                    "bbox": [x0, y0, x1, y1],
                    "label": str(label),
                    "page_index": page_idx,
                }
                if score is not None:
                    chunk["score"] = float(score)
                if block_idx is not None:
                    chunk["block_index"] = block_idx
                layouts[page_idx].chunks.append(chunk)
    for layout in layouts:
        layout.chunks.sort(key=lambda chunk: chunk.get("block_index", float("inf")))
    return layouts


def analyze_layout_surya(
    file_path: Path,
    images: Sequence[Image.Image],
    debug_dir: Path | None = None,
) -> Tuple[List[Image.Image], List]:
    """
    Run layout detection using marker (Surya-backed) converter to get block layout.

    Inputs:
        file_path: source file path
        images: pre-rendered page images
        debug_dir: optional directory to save annotated debug images
    """
    assert images, "marker layout analysis requires at least one page image"
    print(f"  [layout] backend: marker (converter) -> {len(images)} page(s)")

    converter = _load_marker_converter()
    rendered = converter(str(file_path))
    blocks = getattr(rendered, "blocks", []) or []
    page_info = getattr(rendered, "page_info", {}) or {}

    layout_results = _blocks_to_layouts(blocks, len(images), images, page_info)
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
            out_path = page_dir / f"{file_path.stem}_marker_layout.png"
            annotated.save(out_path)
            print(f"     [marker] saved debug image -> {out_path}")

    return list(images), layout_results
