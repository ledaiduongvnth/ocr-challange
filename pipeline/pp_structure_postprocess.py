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


def postprocess_with_ppstructure(
    layout_results: List, images: Sequence[Image.Image] | None = None
) -> List:
    """
    Apply PP-StructureV3 postprocess when available; fallback to reading-order sort.

    Inputs:
        layout_results: list of layout result objects; each must have a 'chunks' attribute/list.
        images: optional list of page images (PIL) matching layout_results, needed for PP-Structure postprocess.
    Outputs:
        layout_results with chunks post-processed.
    """
    PostProcess = None
    import_error = None
    try:
        from paddlex.inference.pipelines.pp_structurev3.postprocess import (  # type: ignore
            PostProcess as PPPost,
        )
        PostProcess = PPPost
    except Exception as exc:  # pragma: no cover - optional dependency
        import_error = exc
        try:
            from paddlex.inference.serving.basic_serving._pipeline_apps.pp_structurev3 import (  # type: ignore
                PostProcess as PPPost,
            )
            PostProcess = PPPost
            import_error = None
        except Exception as exc2:
            import_error = exc2

    if PostProcess is None:
        print(f"PP-Structure postprocess unavailable ({import_error}); using fallback ordering.")
        return [
            _to_layout_result(
                _sort_chunks_reading_order(getattr(l, "chunks", None) or [])
            )
            for l in layout_results or []
        ]

    processed = []
    for idx, layout in enumerate(layout_results or []):
        chunks = getattr(layout, "chunks", None) or []
        payload = {"boxes": chunks}
        if images and idx < len(images):
            payload["image"] = np.array(images[idx].convert("RGB"))
        try:
            res = PostProcess()(payload) or {}
            new_chunks = res.get("boxes") or res.get("layout") or chunks
        except Exception as exc:
            print(f"PP-Structure postprocess failed ({exc}); using fallback ordering.")
            new_chunks = _sort_chunks_reading_order(chunks)
        processed.append(_to_layout_result(new_chunks))
    return processed
