from __future__ import annotations

from pathlib import Path
from typing import List, Callable

from chandra.model.schema import BatchInputItem, BatchOutputItem
from utils import HTML_TEMPLATE

def run_ocr_pipeline(
    file_path: Path,
    args,
    inference,
    generate_kwargs: dict,
    base_prompt: str | None,
    batch_size: int,
    loader: Callable,
    batch_input_cls=BatchInputItem,
    images: List | None = None,
    layout_results: List | None = None,
    debug_dir: Path | None = None,
    ) -> List[BatchOutputItem]:
    if base_prompt == "default":    
        from chandra.prompts import PROMPT_MAPPING
        base_prompt = PROMPT_MAPPING["ocr_layout"]
    else:
        from chandra_prompts import PROMPT_MAPPING
        base_prompt = PROMPT_MAPPING["ocr_layout"]
    if images is None:
        config = {"page_range": args.page_range} if args.page_range else {}
        images = loader(str(file_path), config)
        print(f"  -> {len(images)} page(s)")
    else:
        print(f"  -> using preloaded images ({len(images)} page(s))")

    outputs: List[BatchOutputItem] = []
    assert layout_results is not None and len(layout_results) == len(
        images
    ), "layout_results (from chandra_layout_analysis/pp_doclayout/native_pdf) must be provided and match number of pages"

    # Recognize each detected component individually using cropped regions.
    component_items = []
    component_pages: list[int] = []
    for page_idx, layout in enumerate(layout_results, 0):
        chunks = getattr(layout, "chunks", None) or []
        page_image = images[page_idx]
        for chunk in chunks:
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
            pad = 5 if label in {"table"} else 3
            # Clamp bounds
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
            component_items.append(
                batch_input_cls(
                    image=cropped,
                    prompt_type="ocr_layout",
                    prompt=base_prompt,
                )
            )
            component_pages.append(page_idx)

    print(f"     batching {len(component_items)} detected components for OCR")
    page_outputs: dict[int, BatchOutputItem] = {}
    for start in range(0, len(component_items), batch_size):
        end = min(start + batch_size, len(component_items))
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(component_items[start:end], **batch_kwargs)
        for offset, res in enumerate(results or []):
            comp_idx = start + offset
            page_idx = component_pages[comp_idx] if comp_idx < len(component_pages) else 0
            if res.html:
                rows_html = res.html
                if "<table" in rows_html:
                    rows_html = rows_html.replace("<table", '<table class="pdf-table"', 1)
                else:
                    rows_html = f'<table class="pdf-table">{rows_html}</table>'
                res.html = HTML_TEMPLATE.format(table_rows=rows_html)
            if page_idx not in page_outputs:
                page_outputs[page_idx] = res
                continue
            # Merge component text into the existing page result
            existing = page_outputs[page_idx]
            if hasattr(existing, "markdown"):
                existing.markdown = (
                    (existing.markdown or "").rstrip()
                    + "\n\n"
                    + (res.markdown or "")
                ).strip()
            if hasattr(existing, "html"):
                html_parts = [existing.html or ""]
                if res.html:
                    html_parts.append(res.html)
                existing.html = "\n\n<!-- component break -->\n\n".join(
                    [part for part in html_parts if part]
                )
            if hasattr(existing, "raw"):
                existing.raw = (
                    (existing.raw or "").rstrip()
                    + "\n\n"
                    + (getattr(res, "raw", "") or "")
                ).strip()
    # Return page-ordered outputs
    outputs = [page_outputs[idx] for idx in sorted(page_outputs)]

    return outputs
