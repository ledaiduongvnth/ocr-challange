from __future__ import annotations

from pathlib import Path

import pdfplumber

from utils import (
    render_native_cells_as_html,
    render_native_cells_as_markdown,
)


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


def build_native_outputs(
    file_path: Path,
    layout_results: list | None = None,
    layout_images: list | None = None,
    debug_dir: Path | None = None,
) -> list | None:
    """Extract table/text for native PDFs and update layout_results with content/html/markdown."""
    try:
        if not layout_results:
            return None

        assert layout_images is not None and len(layout_images) == len(
            layout_results
        ), "layout_images must align with layout_results for native extraction"

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
        crop_idx = 0

        if debug_dir:
            print("Before native extraction refinement:")
            for dbg_idx, dbg_layout in enumerate(layout_results, 1):
                dbg_chunks = getattr(dbg_layout, "chunks", None) or []
                print(f"  page {dbg_idx} (idx={dbg_idx-1}) -> {len(dbg_chunks)} chunks")
                for dbg_cidx, dbg_chunk in enumerate(dbg_chunks, 1):
                    dbg_content = (
                        dbg_chunk.get("content")
                        or dbg_chunk.get("text")
                        or dbg_chunk.get("markdown")
                        or ""
                    )
                    print(f"    chunk {dbg_cidx}: {dbg_content}")

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
                    if img_width and img_height:
                        scale_x = page_width / max(1, img_width)
                        scale_y = page_height / max(1, img_height)
                        x0 *= scale_x
                        x1 *= scale_x
                        y0 *= scale_y
                        y1 *= scale_y

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

                    extracted_text = (cropped_page.extract_text() or "").strip()
                    markdown_text = extracted_text
                    html_text = extracted_text

                    if label == "table":
                        try:
                            table_cells = []
                            tables = cropped_page.find_tables()
                            print(
                                f"    tables found: {len(tables)}, cell_counts={[len(getattr(tbl, 'cells', []) or []) for tbl in tables]}"
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
                                html_text = render_native_cells_as_html(table_cells)
                                markdown_text = render_native_cells_as_markdown(table_cells)
                        except Exception:
                            pass

                    chunk["content"] = html_text if label == "table" else markdown_text

        rebuilt_pages: list = []
        for layout in layout_results:
            chunks = getattr(layout, "chunks", None) or []
            lines: list[str] = []
            html_blocks: list[str] = []
            for chunk in chunks:
                content = (
                    chunk.get("content")
                    or chunk.get("text")
                    or chunk.get("markdown")
                    or ""
                )
                if content:
                    lines.append(str(content))
                    if "<table" in content or "<p" in content or "<html" in content:
                        html_blocks.append(content)
                    else:
                        html_blocks.append(f"<p>{content}</p>")
            layout.markdown = "\n\n".join(lines) if lines else ""
            layout.html = (
                f"<html><body>{''.join(html_blocks)}</body></html>" if html_blocks else ""
            )
            rebuilt_pages.append(layout)

        if debug_dir:
            print("After native extraction refinement:")
            for dbg_idx, dbg_layout in enumerate(rebuilt_pages, 1):
                dbg_chunks = getattr(dbg_layout, "chunks", None) or []
                print(f"  page {dbg_idx} (idx={dbg_idx-1}) -> {len(dbg_chunks)} chunks")
                for dbg_cidx, dbg_chunk in enumerate(dbg_chunks, 1):
                    dbg_content = (
                        dbg_chunk.get("content")
                        or dbg_chunk.get("text")
                        or dbg_chunk.get("markdown")
                        or ""
                    )
                    print(f"    chunk {dbg_cidx}: {dbg_content}")

        return rebuilt_pages
    except Exception as exc:
        print(f"  Native PDF table extraction failed ({exc}); falling back to OCR.")
        return None
