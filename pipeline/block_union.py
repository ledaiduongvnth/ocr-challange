from __future__ import annotations

from typing import Any, List, Sequence

from PIL import Image, ImageDraw


def merge_overlapping_non_table_chunks(
    layout_results: Sequence, images: Sequence[Image.Image] | None = None
) -> List:
    """Merge overlapping table/non-table chunks per page and attach cropped images."""

    def overlaps(b1, b2) -> bool:
        if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
            return False
        x0, y0, x1, y1 = b1[:4]
        x0b, y0b, x1b, y1b = b2[:4]
        return not (x1 <= x0b or x0 >= x1b or y1 <= y0b or y0 >= y1b)

    def area(box) -> float:
        if not box or len(box) < 4:
            return 0.0
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    def label_of(chunk: dict[str, Any]) -> str:
        return str(
            chunk.get("label")
            or chunk.get("type")
            or chunk.get("category")
            or ""
        ).lower()

    def merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunks:
            return []
        merged: list[dict[str, Any]] = []
        used = [False] * len(chunks)
        for idx, chunk in enumerate(chunks):
            if used[idx]:
                continue
            used[idx] = True
            group_indices = [idx]
            queue = [idx]
            while queue:
                current = queue.pop()
                current_bbox = chunks[current].get("bbox")
                for other_idx in range(len(chunks)):
                    if used[other_idx]:
                        continue
                    if overlaps(current_bbox, chunks[other_idx].get("bbox")):
                        used[other_idx] = True
                        queue.append(other_idx)
                        group_indices.append(other_idx)
            if len(group_indices) == 1:
                merged.append(chunks[group_indices[0]])
                continue
            group_chunks = [chunks[i] for i in group_indices]
            base_chunk = max(group_chunks, key=lambda c: area(c.get("bbox")))
            merged_bbox = [
                min(c["bbox"][0] for c in group_chunks),
                min(c["bbox"][1] for c in group_chunks),
                max(c["bbox"][2] for c in group_chunks),
                max(c["bbox"][3] for c in group_chunks),
            ]
            combined = dict(base_chunk)
            combined["bbox"] = merged_bbox
            merged.append(combined)
        return merged

    merged_layouts: List = []
    for layout in layout_results or []:
        chunks = getattr(layout, "chunks", None) or []
        table_chunks: list[dict[str, Any]] = []
        other_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            bbox = chunk.get("bbox") or chunk.get("box") or chunk.get("page_box")
            if not bbox or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = map(float, bbox[:4])
            normalized_chunk = dict(chunk)
            normalized_chunk["bbox"] = [x0, y0, x1, y1]
            if "table" in label_of(normalized_chunk):
                table_chunks.append(normalized_chunk)
            else:
                other_chunks.append(normalized_chunk)

        merged_tables = merge_chunks(table_chunks)
        merged_non_tables = merge_chunks(other_chunks)
        final_chunks = merged_tables + merged_non_tables
        final_chunks.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
        layout.chunks = final_chunks
        merged_layouts.append(layout)

    if images:
        for page_idx, layout in enumerate(merged_layouts):
            if page_idx >= len(images):
                break
            page_img = images[page_idx]
            src_img = page_img if page_img.mode == "RGB" else page_img.convert("RGB")
            page_chunks = getattr(layout, "chunks", None) or []
            non_table_bboxes = [
                chunk["bbox"]
                for chunk in page_chunks
                if "table" not in label_of(chunk)
            ]
            for chunk in page_chunks:
                bbox = chunk.get("bbox")
                if not bbox or len(bbox) < 4:
                    chunk["crop_image"] = None
                    continue
                crop_box = tuple(int(round(v)) for v in bbox[:4])
                crop_img = src_img.crop(crop_box)
                if "table" in label_of(chunk) and non_table_bboxes:
                    draw = None
                    for nt_bbox in non_table_bboxes:
                        inter_x0 = max(bbox[0], nt_bbox[0])
                        inter_y0 = max(bbox[1], nt_bbox[1])
                        inter_x1 = min(bbox[2], nt_bbox[2])
                        inter_y1 = min(bbox[3], nt_bbox[3])
                        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                            continue
                        if draw is None:
                            draw = ImageDraw.Draw(crop_img)
                        draw.rectangle(
                            (
                                inter_x0 - bbox[0],
                                inter_y0 - bbox[1],
                                inter_x1 - bbox[0],
                                inter_y1 - bbox[1],
                            ),
                            fill="white",
                        )
                chunk["crop_image"] = crop_img

    return merged_layouts
