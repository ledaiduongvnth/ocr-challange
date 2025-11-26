from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils import filter_non_text_chunks, log_component_bboxes


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


def merge_overlapping_non_table_chunks(
    layout_results: Sequence, images: Sequence[Image.Image] | None = None
) -> List:
    """Merge overlapping non-table chunks on each page into single combined chunks."""

    def overlaps(b1, b2) -> bool:
        if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
            return False
        x0, y0, x1, y1 = b1[:4]
        x0b, y0b, x1b, y1b = b2[:4]
        return not (x1 <= x0b or x0 >= x1b or y1 <= y0b or y0 >= y1b)

    merged_layouts: List = []
    for layout in layout_results or []:
        chunks = getattr(layout, "chunks", None) or []
        tables = []
        for c in chunks:
            if (c.get("label") or "").lower() == "table":
                if images:
                    page_idx = int(c.get("page_index", 0) or 0)
                    if 0 <= page_idx < len(images):
                        img = images[page_idx]
                        bbox = c.get("bbox") or []
                        if len(bbox) >= 4:
                            x0, y0, x1, y1 = bbox[:4]
                            left = max(0, int(x0))
                            top = max(0, int(y0))
                            right = min(img.width, int(x1))
                            bottom = min(img.height, int(y1))
                            if right > left and bottom > top:
                                cropped = img.crop((left, top, right, bottom))
                                # Mask out overlaps with any chunk to white on the crop.
                                draw = ImageDraw.Draw(cropped)
                                for other in chunks:
                                    if other is c:
                                        continue
                                    obbox = other.get("bbox") or []
                                    if len(obbox) < 4:
                                        continue
                                    ox0, oy0, ox1, oy1 = obbox[:4]
                                    ix0 = max(left, int(ox0))
                                    iy0 = max(top, int(oy0))
                                    ix1 = min(right, int(ox1))
                                    iy1 = min(bottom, int(oy1))
                                    if ix1 <= ix0 or iy1 <= iy0:
                                        continue
                                    # Translate intersection into crop coordinates.
                                    draw.rectangle(
                                        (
                                            ix0 - left,
                                            iy0 - top,
                                            ix1 - left,
                                            iy1 - top,
                                        ),
                                        fill="white",
                                    )
                                c["crop_image"] = cropped
                tables.append(c)
        
        
        non_tables = [c for c in chunks if (c.get("label") or "").lower() != "table"]

        visited = set()
        grouped: List[List[dict[str, Any]]] = []
        for idx, chunk in enumerate(non_tables):
            if idx in visited:
                continue
            stack = [idx]
            visited.add(idx)
            component: List[dict[str, Any]] = []
            while stack:
                cur = stack.pop()
                component.append(non_tables[cur])
                for nbr, other in enumerate(non_tables):
                    if nbr in visited:
                        continue
                    if overlaps(non_tables[cur].get("bbox"), other.get("bbox")):
                        visited.add(nbr)
                        stack.append(nbr)
            grouped.append(component)

        merged_chunks: List[dict[str, Any]] = []
        for group in grouped:
            if not group:
                continue
            valid_boxes = [g for g in group if g.get("bbox") and len(g["bbox"]) >= 4]
            if not valid_boxes:
                continue
            # pick the largest by area to inherit properties
            def area(ch):
                x0, y0, x1, y1 = ch["bbox"][:4]
                return max(0.0, x1 - x0) * max(0.0, y1 - y0)

            base_chunk = max(valid_boxes, key=area)
            xs0, ys0, xs1, ys1 = zip(*(g["bbox"][:4] for g in valid_boxes))
            merged_bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
            merged_chunk = base_chunk.copy()
            merged_chunk["bbox"] = merged_bbox

            if images:
                page_idx = int(base_chunk.get("page_index", 0) or 0)
                if 0 <= page_idx < len(images):
                    img = images[page_idx]
                    left = max(0, int(merged_bbox[0]))
                    top = max(0, int(merged_bbox[1]))
                    right = min(img.width, int(merged_bbox[2]))
                    bottom = min(img.height, int(merged_bbox[3]))
                    if right > left and bottom > top:
                        # Start with white canvas
                        canvas = Image.new(img.mode, (right - left, bottom - top), "white")
                        for g in valid_boxes:
                            gx0, gy0, gx1, gy1 = g["bbox"][:4]
                            ix0 = max(left, int(gx0))
                            iy0 = max(top, int(gy0))
                            ix1 = min(right, int(gx1))
                            iy1 = min(bottom, int(gy1))
                            if ix1 <= ix0 or iy1 <= iy0:
                                continue
                            region = img.crop((ix0, iy0, ix1, iy1))
                            canvas.paste(region, (ix0 - left, iy0 - top))
                        merged_chunk["crop_image"] = canvas

            merged_chunks.append(merged_chunk)

        new_chunks = tables + merged_chunks
        new_chunks.sort(key=lambda chunk: chunk.get("block_index", float("inf")))
        layout.chunks = new_chunks
        merged_layouts.append(layout)

    return merged_layouts


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
    # layout_results = filter_non_text_chunks(layout_results)

    if debug_dir:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
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
                block_idx = chunk.get("block_index")
                text = f"{label}"
                if block_idx is not None:
                    text = f"{text}#{block_idx}"
                draw.text((x0 + 2, y0 + 2), text, fill="red", font=font)
            page_dir = debug_dir / f"{page_idx:03d}" / "debug_layout"
            page_dir.mkdir(parents=True, exist_ok=True)
            out_path = page_dir / f"{file_path.stem}_marker_layout.png"
            annotated.save(out_path)
            print(f"     [marker] saved debug image -> {out_path}")
    
    layout_results = merge_overlapping_non_table_chunks(layout_results, images)

    if debug_dir:
        for page_idx, (img, layout) in enumerate(zip(images, layout_results), 1):
            for chunk in getattr(layout, "chunks", None) or []:
                label = chunk.get("label") or "unknown"
                block_idx = chunk.get("block_index")
                crop_img = chunk.get("crop_image")
                if crop_img is not None:
                    crop_dir = debug_dir / f"{page_idx:03d}" / "debug_crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{file_path.stem}_block{block_idx if block_idx is not None else 'na'}_{label}.png"
                    crop_path = crop_dir / crop_name
                    crop_img.save(crop_path)
    return list(images), layout_results
