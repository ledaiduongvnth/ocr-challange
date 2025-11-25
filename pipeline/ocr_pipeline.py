from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from chandra.model.schema import BatchInputItem, BatchOutputItem
from utils import HTML_TEMPLATE


def run_ocr_pipeline(
    file_path: Path,
    args,
    inference,
    generate_kwargs: dict,
    base_prompt: str | None,
    batch_size: int,
    batch_input_cls=BatchInputItem,
    images: List | None = None,
    layout_results: List | None = None,
    debug_dir: Path | None = None,
) -> List[BatchOutputItem]:
    prompt_source = None
    if base_prompt == "default":
        from chandra.prompts import PROMPT_MAPPING
        prompt_source = PROMPT_MAPPING
    else:
        from chandra_prompts import PROMPT_MAPPING
        prompt_source = PROMPT_MAPPING
    base_prompt = prompt_source["ocr_layout"]
    assert images is not None, "Preloaded images are required for OCR pipeline"
    print(f"  -> using preloaded images ({len(images)} page(s))")

    assert layout_results is not None and len(layout_results) == len(images), (
        "layout_results (from chandra_layout_analysis/pp_doclayout/native_pdf) must be provided "
        "and match number of pages"
    )

    if debug_dir:
        print("Before OCR refinement:")
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

    # Recognize each detected component individually using cropped regions.
    component_items: list = []
    component_index_map: list[tuple[int, int]] = []
    for page_idx, layout in enumerate(layout_results, 0):
        chunks = getattr(layout, "chunks", None) or []
        page_image = images[page_idx]
        for chunk_idx, chunk in enumerate(chunks):
            bbox = chunk.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                x0, y0, x1, y1 = [max(0, float(v)) for v in bbox[:4]]
            except Exception:
                continue
            x0, y0 = int(x0), int(y0)
            x1, y1 = int(x1), int(y1)
            label = (chunk.get("label") or chunk.get("type") or "").lower()
            pad = 0 if label in {"table"} else 0
            x0 = max(0, min(x0 - pad, page_image.width))
            y0 = max(0, min(y0 - pad, page_image.height))
            x1 = max(x0 + 1, min(x1 + pad, page_image.width))
            y1 = max(y0 + 1, min(y1 + pad, page_image.height))
            cropped = page_image.crop((x0, y0, x1, y1))
            if debug_dir:
                try:
                    page_dir = debug_dir / f"{page_idx+1:03d}" / "debug_ocr_components"
                    page_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = page_dir / f"{file_path.stem}_comp{len(component_items)+1}.png"
                    cropped.save(crop_path)
                except Exception:
                    pass
            if label == "table":
                custom_prompt = prompt_source["ocr_table"]
                component_items.append(
                batch_input_cls(
                    image=cropped,
                    prompt_type="ocr",
                    prompt=custom_prompt,
                )
            )
            else:
                custom_prompt = prompt_source["ocr"]

                component_items.append(
                    batch_input_cls(
                        image=cropped,
                        prompt_type="ocr",
                        prompt=custom_prompt,
                    )
                )
            component_index_map.append((page_idx, chunk_idx))

    print(f"     batching {len(component_items)} detected components for OCR")
    for start in range(0, len(component_items), batch_size):
        end = min(start + batch_size, len(component_items))
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(component_items[start:end], **batch_kwargs)
        for offset, res in enumerate(results or []):
            comp_idx = start + offset
            if comp_idx < len(component_index_map):
                chunk_page_idx, chunk_idx = component_index_map[comp_idx]
                if chunk_page_idx < len(layout_results):
                    target_layout = layout_results[chunk_page_idx]
                    target_chunks = getattr(target_layout, "chunks", None) or []
                    if chunk_idx < len(target_chunks):
                        target_chunk = target_chunks[chunk_idx]
                        content_value = res.markdown or getattr(res, "raw", "") or ""
                        if res.html and ("<table" in res.html or "</table>" in res.html):
                            rendered = HTML_TEMPLATE.format(table_rows=res.html)
                            target_chunk["content"] = rendered
                        else:
                            target_chunk["content"] = content_value

    updated_pages: list[BatchOutputItem] = []
    for layout in layout_results:
        chunks = getattr(layout, "chunks", None) or []
        lines = []
        html_blocks = []
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
        updated_pages.append(layout)

    if debug_dir:
        print("After OCR refinement:")
        for dbg_idx, dbg_layout in enumerate(updated_pages, 1):
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

    return updated_pages
