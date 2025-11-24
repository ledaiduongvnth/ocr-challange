from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image


def _sort_chunks_reading_order(chunks: List[dict]) -> List[dict]:
    """Fallback reading-order heuristic: top-to-bottom, then left-to-right."""
    return sorted(
        chunks,
        key=lambda c: (
            (c.get("bbox") or [0, 0])[1]
            if (c.get("bbox") and len(c["bbox"]) >= 2)
            else 0,
            (c.get("bbox") or [0, 0])[0]
            if (c.get("bbox") and len(c["bbox"]) >= 1)
            else 0,
        ),
    )


def _to_layout_result(chunks: List[dict]):
    return type("LayoutResult", (object,), {"chunks": chunks})()


def _paddlex_reading_order(
    layout_results: List, images: Sequence[Image.Image] | None
) -> List | None:
    """Use paddlex layout_parsing sorted_layout_boxes if available for better ordering."""
    if not images:
        return None
    try:
        from paddlex.inference.pipelines.layout_parsing.utils import (
            sorted_layout_boxes,
        )
    except Exception:
        return None

    ordered = []
    for idx, layout in enumerate(layout_results or []):
        chunks = getattr(layout, "chunks", None) or []
        img = images[idx] if idx < len(images) else None
        if not img:
            ordered.append(_to_layout_result(chunks))
            continue
        prepared = []
        for chunk in chunks:
            bbox = chunk.get("bbox") or chunk.get("coordinate")
            if not bbox or len(bbox) < 4:
                continue
            prepared.append({"block_bbox": bbox, "chunk": chunk})
        if not prepared:
            ordered.append(_to_layout_result(chunks))
            continue
        sorted_chunks = sorted_layout_boxes(prepared, w=img.width)
        ordered.append(_to_layout_result([entry["chunk"] for entry in sorted_chunks]))
    return ordered


def postprocess_with_ppstructure(
    layout_results: List, images: Sequence[Image.Image] | None = None
) -> List:
    """
    Apply PP-StructureV3 postprocess when available; fallback to reading-order sort.

    Inputs:
        layout_results: list of layout result objects; each must have a 'chunks' attribute/list.
        images: optional list of page images (PIL) matching layout_results, used for paddlex heuristic.
    Outputs:
        layout_results with chunks post-processed.
    """
    ordered = _paddlex_reading_order(layout_results, images)
    if ordered is not None:
        print("Using paddlex reading-order heuristic.")
        return ordered

    print("Using simple reading-order sort.")
    return [
        _to_layout_result(
            _sort_chunks_reading_order(getattr(l, "chunks", None) or [])
        )
        for l in layout_results or []
    ]
