from __future__ import annotations

from typing import Any, List, Sequence

from PIL import Image, ImageDraw


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
            if "table" in (c.get("label") or "").lower():
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

        non_tables = [c for c in chunks if "table" not in (c.get("label") or "").lower()]

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

            def area(ch):
                x0, y0, x1, y1 = ch["bbox"][:4]
                return max(0.0, x1 - x0) * max(0.0, y1 - y0)

            base_chunk = max(valid_boxes, key=area)
            xs0, ys0, xs1, ys1 = zip(*(g["bbox"][:4] for g in valid_boxes))
            merged_bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
            merged_chunk = base_chunk.copy()
            merged_chunk["bbox"] = merged_bbox
            block_indices = [
                b.get("block_index") for b in valid_boxes if b.get("block_index") is not None
            ]
            if block_indices:
                merged_chunk["block_index"] = min(block_indices)
            merged_chunk["page_index"] = base_chunk.get("page_index", 0)

            if images:
                page_idx = int(base_chunk.get("page_index", 0) or 0)
                if 0 <= page_idx < len(images):
                    img = images[page_idx]
                    left = max(0, int(merged_bbox[0]))
                    top = max(0, int(merged_bbox[1]))
                    right = min(img.width, int(merged_bbox[2]))
                    bottom = min(img.height, int(merged_bbox[3]))
                    if right > left and bottom > top:
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
