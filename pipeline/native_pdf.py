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

def build_native_outputs(
    file_path: Path,
    layout_results: list | None = None,
    layout_images: list | None = None,
    debug_dir: Path | None = None,
) -> list[BatchOutputItem] | None:
    """Extract table/text for native PDFs and return as BatchOutputItem list."""
    try:
        if layout_results:
            assert layout_images is not None and len(layout_images) == len(
                layout_results
            ), "layout_images must align with layout_results for native extraction"
            outputs: list[BatchOutputItem] = []
            page_outputs: dict[int, BatchOutputItem] = {}
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
            crop_idx = 0
            with pdfplumber.open(str(file_path)) as pdf:
                for page_idx, layout in enumerate(layout_results, 0):
                    if page_idx >= len(pdf.pages):
                        break
                    page = pdf.pages[page_idx]
                    page_width, page_height = page.width, page.height
                    img_width = img_height = None
                    if layout_images and page_idx < len(layout_images):
                        img_width, img_height = layout_images[page_idx].size
                    chunks = getattr(layout, "chunks", None) or []
                    for chunk in chunks:
                        bbox = chunk.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        x0, y0, x1, y1 = bbox[:4]
                        label = (chunk.get("label") or "").lower()
                        # If we have corresponding layout image dimensions, scale bbox to PDF coordinates.
                        if img_width and img_height:
                            scale_x = page_width / max(1, img_width)
                            scale_y = page_height / max(1, img_height)
                            x0 *= scale_x
                            x1 *= scale_x
                            y0 *= scale_y
                            y1 *= scale_y
                        # Pad and clamp to page bounds to avoid pdfplumber errors.
                        pad = 5 if label in {"table"} else 3
                        x0 = max(0, min(x0 - pad, page_width))
                        y0 = max(0, min(y0 - pad, page_height))
                        x1 = max(x0 + 1e-3, min(x1 + pad, page_width))
                        y1 = max(y0 + 1e-3, min(y1 + pad, page_height))
                        cropped_page = page.within_bbox((x0, y0, x1, y1))
                        if not cropped_page:
                            continue
                        if debug_dir:
                            try:
                                img = cropped_page.to_image(resolution=200).original
                                crop_idx += 1
                                page_dir = (
                                    debug_dir
                                    / f"{page_idx+1:03d}"
                                    / "debug_native_components"
                                )
                                page_dir.mkdir(parents=True, exist_ok=True)
                                crop_path = page_dir / f"{file_path.stem}_comp{crop_idx}.png"
                                img.save(crop_path)
                            except Exception:
                                pass
                        text = (cropped_page.extract_text() or "").strip()
                        markdown_text = text
                        html_text = text
                        # If this chunk appears to be a table, try table cell extraction within the crop.
                        if label == "table":
                            try:
                                table_cells = []
                                tables = cropped_page.find_tables()
                                cell_counts = [
                                    len(getattr(tbl, "cells", []) or []) for tbl in tables
                                ]
                                print(
                                    f"    tables found: {len(tables)}, cell_counts={cell_counts}"
                                )
                                for tbl in tables:
                                    for cell_bbox in getattr(tbl, "cells", []) or []:
                                        if isinstance(cell_bbox, dict):
                                            box = (
                                                cell_bbox.get("x0"),
                                                cell_bbox.get("top"),
                                                cell_bbox.get("x1"),
                                                cell_bbox.get("bottom"),
                                            )
                                        elif isinstance(cell_bbox, (list, tuple)) and len(cell_bbox) >= 4:
                                            box = cell_bbox[:4]
                                        else:
                                            box = (
                                                getattr(cell_bbox, "x0", None),
                                                getattr(cell_bbox, "top", None),
                                                getattr(cell_bbox, "x1", None),
                                                getattr(cell_bbox, "bottom", None),
                                            )
                                        if None in box:
                                            continue
                                        cell_text = (
                                            page.within_bbox(box).extract_text() or ""
                                        ).strip()
                                        table_cells.append(
                                            {
                                                "text": cell_text,
                                                "x0": float(box[0]),
                                                "x1": float(box[2]),
                                                "top": float(box[1]),
                                                "bottom": float(box[3]),
                                            }
                                        )
                                if table_cells:
                                    print(f"    table_cells: {len(table_cells)}")
                                    html_text = render_native_cells_as_html(table_cells)
                                    markdown_text = render_native_cells_as_markdown(table_cells)
                            except Exception:
                                pass
                        if page_idx not in page_outputs:
                            page_outputs[page_idx] = BatchOutputItem(
                                markdown=markdown_text,
                                html=html_text,
                                chunks={},
                                raw=markdown_text,
                                page_box=[x0, y0, x1, y1],
                                token_count=0,
                                images={},
                                error=False,
                            )
                        else:
                            existing = page_outputs[page_idx]
                            existing.markdown = (
                                (existing.markdown or "").rstrip()
                                + "\n\n"
                                + (markdown_text or "")
                            ).strip()
                            existing.html = (
                                (existing.html or "").rstrip()
                                + "\n\n<!-- component break -->\n\n"
                                + (html_text or "")
                            ).strip()
                            existing.raw = (
                                (existing.raw or "").rstrip()
                                + "\n\n"
                                + (markdown_text or "")
                            ).strip()
                            existing.page_box = [x0, y0, x1, y1]
            if page_outputs:
                return [page_outputs[idx] for idx in sorted(page_outputs)]
    except Exception as exc:
        print(f"  Native PDF table extraction failed ({exc}); falling back to OCR.")
        return None
