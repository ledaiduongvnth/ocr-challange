from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pdfplumber

from chandra.model.schema import BatchOutputItem

COORD_TOLERANCE = 2.0
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Extracted PDF Table</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      padding: 1rem;
      background: #f5f5f5;
    }}
    table.pdf-table {{
      border-collapse: collapse;
      min-width: 60%;
      margin: 0 auto;
      background: #fff;
    }}
    table.pdf-table td {{
      border: 1px solid #999;
      padding: 4px 6px;
      vertical-align: top;
      white-space: pre-wrap;
    }}
    table.pdf-table td.empty {{
      background: #fafafa;
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


def render_table_html(rows_html: str) -> str:
    return HTML_TEMPLATE.format(table_rows=rows_html)


def is_digital_pdf(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".pdf":
        return False
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            if not pdf.pages:
                return False
            text = (pdf.pages[0].extract_text() or "").strip()
            return bool(text)
    except Exception:
        return False


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


def extract_pdf_table_cells(
    pdf_path: Path,
    page_index: int = 0,
    table_index: int = 0,
) -> list[dict[str, float | int | str | None]]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not pdf.pages:
            raise RuntimeError(f"No pages found in {pdf_path}")
        if page_index >= len(pdf.pages):
            raise ValueError(f"PDF only has {len(pdf.pages)} pages, cannot access index {page_index}")

        page = pdf.pages[page_index]
        tables = page.find_tables()
        if not tables:
            raise RuntimeError("pdfplumber could not detect table geometry on the requested page")
        if table_index >= len(tables):
            raise ValueError(
                f"Requested table index {table_index} is unavailable; only {len(tables)} table(s) detected."
            )

        table = tables[table_index]
        cells: list[dict[str, float | int | str | None]] = []
        for bbox in getattr(table, "cells", []) or []:
            if isinstance(bbox, dict):
                box = (
                    bbox.get("x0"),
                    bbox.get("top"),
                    bbox.get("x1"),
                    bbox.get("bottom"),
                )
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                box = bbox[:4]
            else:
                box = (
                    getattr(bbox, "x0", None),
                    getattr(bbox, "top", None),
                    getattr(bbox, "x1", None),
                    getattr(bbox, "bottom", None),
                )

            if None in box:
                continue

            x0, top, x1, bottom = box
            extracted = page.within_bbox((x0, top, x1, bottom)).extract_text()
            text = (extracted or "").strip()

            cells.append(
                {
                    "text": text,
                    "x0": float(x0),
                    "x1": float(x1),
                    "top": float(top),
                    "bottom": float(bottom),
                    "row": None,
                    "col": None,
                }
            )

        if not cells:
            raise RuntimeError(
                "Matched table does not expose cell geometry. Try adjusting pdfplumber table settings."
            )

        return cells


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


def build_native_batch_output(html_text: str) -> BatchOutputItem:
    """Wrap native PDF table HTML into a BatchOutputItem for downstream saving."""
    return BatchOutputItem(
        markdown=html_text,
        html=html_text,
        chunks={},
        raw=html_text,
        page_box=[0, 0, 0, 0],
        token_count=0,
        images={},
        error=False,
    )


def build_native_outputs(file_path: Path) -> list[BatchOutputItem] | None:
    """Extract table HTML for native PDFs and return as BatchOutputItem list."""
    try:
        cells = extract_pdf_table_cells(file_path)
        html_text = render_native_cells_as_html(cells)
        return [build_native_batch_output(html_text)]
    except Exception as exc:
        print(f"  Native PDF table extraction failed ({exc}); falling back to OCR.")
        return None
