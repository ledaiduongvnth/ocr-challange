from __future__ import annotations

from pathlib import Path
from typing import List

from chandra.model.schema import BatchInputItem, BatchOutputItem
from utils import HTML_TEMPLATE
from hoang_prompt import OCR_PROMPT, TABLE_ONLY_PROMPT

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
                dbg_content = dbg_chunk.get("markdown") or ""
                block_idx = dbg_chunk.get("block_index")
                print(f"    chunk {dbg_cidx} (block_idx={block_idx}): {dbg_content}")

    # Recognize each detected component individually using cropped regions.
    component_items: list = []
    component_index_map: list[tuple[int, int]] = []
    for page_idx, layout in enumerate(layout_results, 0):
        chunks = getattr(layout, "chunks", None) or []
        for chunk_idx, chunk in enumerate(chunks):
            block_idx = chunk.get("block_index")
            label = (chunk.get("label") or chunk.get("type") or "").lower()

            # Use precomputed crop from layout stage; it must be present.
            cropped = chunk.get("crop_image")
            assert cropped is not None, "chunk['crop_image'] must be provided by layout stage"

            if debug_dir:
                try:
                    page_dir = debug_dir / f"{page_idx+1:03d}" / "debug_ocr_components"
                    page_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = page_dir / f"{file_path.stem}_comp{block_idx}.png"
                    cropped.save(crop_path)
                except Exception:
                    pass

            prompt = TABLE_ONLY_PROMPT if label == "table" else OCR_PROMPT
            component_items.append(
                batch_input_cls(
                    image=cropped,
                    prompt_type="ocr",
                    prompt=prompt,
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
                        target_chunk["markdown"] = res.markdown or getattr(res, "raw", "") or ""
                        target_chunk["html"] = res.html or getattr(res, "raw", "") or ""

                        # print(res.html)
                        # print("----------")
                        # # print(res.markdown)
                        # print(getattr(res, "raw", ""))
                        # print("--------------------------------")


    updated_pages: list[BatchOutputItem] = []
    for layout in layout_results:
        chunks = getattr(layout, "chunks", None) or []
        markdown_blocks = []
        html_blocks = []
        for chunk in chunks:
            markdown = chunk.get("markdown") or ""
            html = chunk.get("html") or ""
            if markdown:
                markdown_blocks.append(str(markdown))
            if html:
                if "<table" in html or "<p" in html or "<html" in html:
                    html_blocks.append(html)
                else:
                    html_blocks.append(f"<p>{html}</p>")
        layout.markdown = "\n\n".join(markdown_blocks) if markdown_blocks else ""
        layout.html = (
            f"<html><body>{''.join(html_blocks)}</body></html>" if html_blocks else ""
        )
        layout.token_count = 0
        layout.images = {}
        layout.page_box = []
        updated_pages.append(layout)

    if debug_dir:
        print("After OCR refinement:")
        for dbg_idx, dbg_layout in enumerate(updated_pages, 1):
            dbg_chunks = getattr(dbg_layout, "chunks", None) or []
            print(f"  page {dbg_idx} (idx={dbg_idx-1}) -> {len(dbg_chunks)} chunks")
            for dbg_cidx, dbg_chunk in enumerate(dbg_chunks, 1):
                dbg_content = dbg_chunk.get("markdown") or ""
                block_idx = dbg_chunk.get("block_index")
                print(f"    chunk {dbg_cidx} (block_idx={block_idx}): {dbg_content}")

    return updated_pages
