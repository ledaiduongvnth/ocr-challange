from __future__ import annotations

from typing import List


def log_component_bboxes(file_name: str, results: List) -> None:
    """Log component bounding boxes (tables, text blocks, etc.) per page."""
    print(f"  components for {file_name}:")
    for page_idx, result in enumerate(results, 1):
        chunks = getattr(result, "chunks", None)
        if not chunks:
            continue
        for comp_idx, chunk in enumerate(chunks, 1):
            comp_type = (
                chunk.get("type")
                or chunk.get("label")
                or chunk.get("category")
                or "unknown"
            )
            bbox = chunk.get("bbox") or chunk.get("box") or chunk.get("page_box")
            print(f"    page {page_idx} #{comp_idx}: {comp_type} bbox={bbox}")
