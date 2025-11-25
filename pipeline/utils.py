from __future__ import annotations

import html
from typing import Any, List, Sequence


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

COORD_TOLERANCE = 2.0

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
            block_idx = chunk.get("block_index")
            bbox = chunk.get("bbox") or chunk.get("box") or chunk.get("page_box")
            if block_idx is not None:
                print(
                    f"    page {page_idx} #{comp_idx} (block_idx={block_idx}): {comp_type} bbox={bbox}"
                )
            else:
                print(f"    page {page_idx} #{comp_idx}: {comp_type} bbox={bbox}")


def filter_non_text_chunks(layout_results: List, images: Sequence | None = None):
    """Remove layout chunks whose label contains 'picture', 'image', or 'figure' (case-insensitive)."""
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
            if all(
                banned not in str(c.get("label", "")).lower()
                for banned in ("picture", "image", "figure")
            )
        ]
        res.chunks = keep
        filtered.append(res)
    return filtered


def _unique_sorted(values: list[float]) -> list[float]:
    """Return sorted coordinates merged with a loose tolerance."""
    values = sorted(values)
    result: list[float] = []
    for val in values:
        if not result:
            result.append(val)
            continue
        if abs(val - result[-1]) <= COORD_TOLERANCE:
            result[-1] = (result[-1] + val) / 2
        else:
            result.append(val)
    return result


def _find_index(value: float, coords: list[float]) -> int:
    for idx, coord in enumerate(coords):
        if abs(value - coord) <= COORD_TOLERANCE:
            return idx
    raise ValueError(f"Value {value} did not match any coordinate line.")


def render_table_html(rows_html: str) -> str:
    return HTML_TEMPLATE.format(table_rows=rows_html)


def render_native_cells_as_html(cells: list[dict[str, float | int | str | None]]) -> str:
    x_values: list[float] = []
    y_values: list[float] = []
    for cell in cells:
        x_values.extend([float(cell["x0"]), float(cell["x1"])])
        y_values.extend([float(cell["top"]), float(cell["bottom"])])

    x_lines = _unique_sorted(x_values)
    y_lines = _unique_sorted(y_values)
    col_count = max(1, len(x_lines) - 1)
    row_count = max(1, len(y_lines) - 1)

    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    skip_positions: set[tuple[int, int]] = set()

    for cell in cells:
        row_start = _find_index(float(cell["top"]), y_lines)
        row_end = _find_index(float(cell["bottom"]), y_lines)
        col_start = _find_index(float(cell["x0"]), x_lines)
        col_end = _find_index(float(cell["x1"]), x_lines)

        row_span = max(1, row_end - row_start)
        col_span = max(1, col_end - col_start)

        anchor_key = (row_start, col_start)
        anchors[anchor_key] = {
            "row_span": row_span,
            "col_span": col_span,
            "text": cell.get("text") or "",
            "x0": cell["x0"],
            "x1": cell["x1"],
            "top": cell["top"],
            "bottom": cell["bottom"],
        }

        for r in range(row_start, row_start + row_span):
            for c in range(col_start, col_start + col_span):
                if (r, c) == anchor_key:
                    continue
                skip_positions.add((r, c))

    rows_html: list[str] = []
    for row in range(row_count):
        cells_html: list[str] = []
        for col in range(col_count):
            key = (row, col)
            if key in anchors:
                entry = anchors[key]
                rowspan_attr = f' rowspan="{entry["row_span"]}"' if entry["row_span"] > 1 else ""
                colspan_attr = f' colspan="{entry["col_span"]}"' if entry["col_span"] > 1 else ""
                coord_attrs = (
                    f' data-x0="{entry["x0"]:.2f}"'
                    f' data-x1="{entry["x1"]:.2f}"'
                    f' data-top="{entry["top"]:.2f}"'
                    f' data-bottom="{entry["bottom"]:.2f}"'
                )
                cell_text = html.escape(str(entry["text"])) if entry["text"] else "&nbsp;"
                cells_html.append(
                    f"    <td{rowspan_attr}{colspan_attr}{coord_attrs}>{cell_text}</td>"
                )
            elif key in skip_positions:
                continue
            else:
                cells_html.append('    <td class="empty">&nbsp;</td>')
        rows_html.append("  <tr>\n" + "\n".join(cells_html) + "\n  </tr>")

    return render_table_html("\n".join(rows_html))


def render_native_cells_as_markdown(cells: list[dict[str, float | int | str | None]]) -> str:
    """Return a markdown-compatible HTML table (kept minimal for Markdown)."""
    rows_html: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    for cell in cells:
        x_values.extend([float(cell["x0"]), float(cell["x1"])])
        y_values.extend([float(cell["top"]), float(cell["bottom"])])

    x_lines = _unique_sorted(x_values)
    y_lines = _unique_sorted(y_values)
    col_count = max(1, len(x_lines) - 1)
    row_count = max(1, len(y_lines) - 1)

    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    skip_positions: set[tuple[int, int]] = set()

    for cell in cells:
        row_start = _find_index(float(cell["top"]), y_lines)
        row_end = _find_index(float(cell["bottom"]), y_lines)
        col_start = _find_index(float(cell["x0"]), x_lines)
        col_end = _find_index(float(cell["x1"]), x_lines)
        anchor_key = (row_start, col_start)
        anchors[anchor_key] = {
            "row_span": max(1, row_end - row_start),
            "col_span": max(1, col_end - col_start),
            "text": cell.get("text") or "",
        }
        for r in range(row_start, row_start + anchors[anchor_key]["row_span"]):
            for c in range(col_start, col_start + anchors[anchor_key]["col_span"]):
                if (r, c) == anchor_key:
                    continue
                skip_positions.add((r, c))

    for row in range(row_count):
        cells_html: list[str] = []
        for col in range(col_count):
            key = (row, col)
            if key in anchors:
                entry = anchors[key]
                rowspan_attr = f' rowspan="{entry["row_span"]}"' if entry["row_span"] > 1 else ""
                colspan_attr = f' colspan="{entry["col_span"]}"' if entry["col_span"] > 1 else ""
                cell_text = html.escape(str(entry["text"])) if entry["text"] else "&nbsp;"
                cells_html.append(f"    <td{rowspan_attr}{colspan_attr}>{cell_text}</td>")
            elif key in skip_positions:
                continue
            else:
                cells_html.append("    <td>&nbsp;</td>")
        rows_html.append("  <tr>\n" + "\n".join(cells_html) + "\n  </tr>")

    return "<table>\n" + "\n".join(rows_html) + "\n</table>"
