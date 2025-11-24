from __future__ import annotations

from typing import List, Sequence


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Extracted PDF Table</title>
  <style>
    table,
    th,
    td {{
      border: 1px solid black;
      border-collapse: collapse;
      padding: 4px;
    }}
  </style>
</head>
<body>
  <table class="pdf-table">
{table_rows}
  </table>
</body>
</html>
"""

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


def filter_image_chunks(layout_results: List, images: Sequence | None = None):
    """Remove layout chunks whose label contains 'image' (case-insensitive)."""
    if not layout_results:
        return layout_results
    filtered = []
    for res in layout_results:
        chunks = getattr(res, "chunks", None)
        if chunks is None:
            filtered.append(res)
            continue
        keep = [
            c
            for c in chunks
            if "image" not in str(c.get("label", "")).lower()
        ]
        res.chunks = keep
        filtered.append(res)
    return filtered
