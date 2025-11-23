from __future__ import annotations

from pathlib import Path
from typing import List, Callable

from chandra.model.schema import BatchInputItem, BatchOutputItem

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
    if base_prompt is None:
        # from chandra.prompts import PROMPT_MAPPING
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
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = debug_dir / f"{file_path.stem}_p{page_idx+1}_comp{len(component_items)+1}.png"
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

    print(f"     batching {len(component_items)} detected components for OCR")
    for start in range(0, len(component_items), batch_size):
        end = min(start + batch_size, len(component_items))
        batch_kwargs = dict(generate_kwargs)
        results = inference.generate(component_items[start:end], **batch_kwargs)
        for res in results or []:
            if res.html:
                rows_html = res.html
                if "<table" in rows_html:
                    rows_html = rows_html.replace("<table", '<table class="pdf-table"', 1)
                else:
                    rows_html = f'<table class="pdf-table">{rows_html}</table>'
                res.html = HTML_TEMPLATE.format(table_rows=rows_html)
            outputs.append(res)

    return outputs
