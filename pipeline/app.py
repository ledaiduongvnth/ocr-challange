"""
FastAPI wrapper exposing the main OCR pipeline.
"""

from __future__ import annotations
import re
import os
import sys
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Default remote inference endpoint if not provided via environment.
os.environ.setdefault("VLLM_API_BASE", "https://uav-vts-chandra.hf.space/v1")
os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", "")
os.environ.setdefault("VLLM_API_KEY", "")


from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.scripts.cli import get_supported_files, save_merged_output
from cli_utils import (
    apply_env_overrides,
    build_inference_options,
    determine_batch_size,
    parse_cli_args,
)
from native_pdf import build_native_outputs, is_digital_pdf
from ocr_pipeline import run_ocr_pipeline
from orientation import normalize_page_images
from pp_structure_postprocess import postprocess_with_ppstructure
from pp_structure_preprocess import preprocess_with_ppstructure
from surya_layout import analyze_layout_surya, _load_marker_converter
from utils import log_component_bboxes

app = FastAPI()
DATE_PREFIX_REGEX = re.compile(r"ngày[\s_]*tháng[\s_]*năm", re.IGNORECASE)


def _get_cli_defaults():
    """Grab CLI defaults without requiring real argv."""
    orig_argv = sys.argv
    try:
        sys.argv = [orig_argv[0] if orig_argv else "app"]
        return parse_cli_args()
    finally:
        sys.argv = orig_argv


_CLI_DEFAULTS = _get_cli_defaults()
_INFERENCE_CACHE: dict[str, InferenceManager] = {}


@app.on_event("startup")
def _preload_models() -> None:
    """Warm up heavy models at startup to avoid first-request latency."""
    try:
        default_method = _CLI_DEFAULTS.method
        if default_method and default_method not in _INFERENCE_CACHE:
            _INFERENCE_CACHE[default_method] = InferenceManager(method=default_method)
    except Exception as exc:
        print(f"[startup] failed to preload InferenceManager: {exc}")

    try:
        _load_marker_converter()
    except Exception as exc:
        print(f"[startup] failed to preload marker converter: {exc}")


def _build_args(
    input_path: Path,
    output_dir: Path,
    *,
    checkpoint: str | None = None,
    method: str | None = None,
    page_range: str | None = None,
    batch_size: int | None = None,
    max_output_tokens: int | None = None,
    max_workers: int | None = None,
    max_retries: int | None = None,
    include_images: bool | None = None,
    include_headers_footers: bool | None = None,
    html: bool | None = None,
    paginate_output: bool | None = None,
    device: str | None = None,
    attn_impl: str | None = None,
    layout_backend: str | None = None,
    preprocess_backend: str | None = None,
    postprocess_backend: str | None = None,
    prompt: str | None = None,
    native_pdf: bool | None = None,
) -> SimpleNamespace:
    """Construct an argparse-like namespace mirroring CLI defaults."""
    return SimpleNamespace(
        input_path=input_path,
        output_dir=output_dir,
        checkpoint=checkpoint if checkpoint is not None else _CLI_DEFAULTS.checkpoint,
        method=method if method is not None else _CLI_DEFAULTS.method,
        page_range=page_range,
        batch_size=batch_size if batch_size is not None else _CLI_DEFAULTS.batch_size,
        max_output_tokens=max_output_tokens,
        max_workers=max_workers,
        max_retries=max_retries,
        include_images=include_images
        if include_images is not None
        else _CLI_DEFAULTS.include_images,
        include_headers_footers=include_headers_footers
        if include_headers_footers is not None
        else _CLI_DEFAULTS.include_headers_footers,
        html=html if html is not None else _CLI_DEFAULTS.html,
        paginate_output=paginate_output
        if paginate_output is not None
        else _CLI_DEFAULTS.paginate_output,
        device=device,
        attn_impl=attn_impl,
        layout_backend=layout_backend
        if layout_backend is not None
        else _CLI_DEFAULTS.layout_backend,
        preprocess_backend=preprocess_backend
        if preprocess_backend is not None
        else _CLI_DEFAULTS.preprocess_backend,
        postprocess_backend=postprocess_backend
        if postprocess_backend is not None
        else _CLI_DEFAULTS.postprocess_backend,
        prompt=prompt if prompt is not None else _CLI_DEFAULTS.prompt,
        native_pdf=native_pdf if native_pdf is not None else getattr(_CLI_DEFAULTS, "native_pdf", False),
    )


def _process_file(file_path: Path, args: SimpleNamespace) -> dict[str, Any]:
    apply_env_overrides(args)
    files: List[Path] = get_supported_files(args.input_path)
    if not files:
        raise RuntimeError(f"No supported files found under {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inference = _INFERENCE_CACHE.get(args.method)
    if inference is None:
        inference = InferenceManager(method=args.method)
        _INFERENCE_CACHE[args.method] = inference
    batch_size = determine_batch_size(args)
    generate_kwargs = build_inference_options(args)
    results_payload: dict[str, Any] = {"pages": []}

    for file_path in files:
        is_pdf = file_path.suffix.lower() == ".pdf"
        is_native_pdf = bool(getattr(args, "native_pdf", False)) and is_pdf and is_digital_pdf(file_path)

        file_output_root = args.output_dir / file_path.stem
        results_dir = file_output_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        load_config = {"page_range": args.page_range} if args.page_range else {}
        temp_pdf_path: Path | None = None
        layout_pdf_path: Path | None = None
        source_path = file_path
        if not is_pdf:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                tmp_handle, tmp_name = tempfile.mkstemp(suffix=".pdf")
                os.close(tmp_handle)
                temp_pdf_path = Path(tmp_name)
                rgb_img.save(temp_pdf_path)
                rgb_img.close()
                source_path = temp_pdf_path
        page_images = load_file(str(source_path), load_config)
        if page_images:
            page_images = normalize_page_images(page_images, save_dir=None, prefix="page")
        debug_dir = file_output_root if args.html else None
        if args.preprocess_backend == "ppstructure":
            page_images = preprocess_with_ppstructure(
                page_images,
                use_orientation=True,
                use_unwarp=True,
                debug_dir=debug_dir,
            )
        layout_source_path = source_path
        # if page_images:
        #     try:
        #         pdf_pages = [img if img.mode == "RGB" else img.convert("RGB") for img in page_images]
        #         tmp_handle, tmp_name = tempfile.mkstemp(suffix=".pdf")
        #         os.close(tmp_handle)
        #         layout_pdf_path = Path(tmp_name)
        #         pdf_pages[0].save(layout_pdf_path, save_all=True, append_images=pdf_pages[1:])
        #         layout_source_path = layout_pdf_path
        #     except Exception:
        #         layout_pdf_path = None

        _, layout_results = analyze_layout_surya(
            file_path=layout_source_path,
            images=page_images,
            debug_dir=debug_dir,
        )

        log_component_bboxes(file_path.name, layout_results)

        if args.postprocess_backend == "ppstructure":
            layout_results = postprocess_with_ppstructure(layout_results, images=page_images)

        match (is_native_pdf,):
            case (True,):
                print("Doing native pdf processing")
                page_outputs = build_native_outputs(
                    file_path,
                    layout_results=layout_results,
                    layout_images=page_images,
                    debug_dir=debug_dir,
                )
            case _:
                print("Doing scan pdf processing")
                page_outputs = run_ocr_pipeline(
                    file_path=file_path,
                    args=args,
                    inference=inference,
                    generate_kwargs=generate_kwargs,
                    base_prompt=args.prompt,
                    batch_size=batch_size,
                    batch_input_cls=BatchInputItem,
                    images=page_images,
                    layout_results=layout_results,
                    debug_dir=debug_dir,
                )

        if page_outputs:
            for layout in page_outputs:
                chunks = getattr(layout, "chunks", None) or []
                markdown_blocks = []
                html_blocks = []
                for chunk in chunks:
                    markdown = chunk.get("markdown") or ""
                    label = (chunk.get("label") or chunk.get("type") or "").lower()
                    crop_image = chunk.get("crop_image")
                    if (
                        is_native_pdf
                        and crop_image is not None
                        and DATE_PREFIX_REGEX.search(markdown)
                        and "table" not in label
                    ):
                        try:
                            single_layout = type("LayoutResult", (object,), {"chunks": []})()
                            new_chunk = dict(chunk)
                            new_chunk["markdown"] = ""
                            new_chunk["html"] = ""
                            single_layout.chunks = [new_chunk]
                            ocr_pages = run_ocr_pipeline(
                                file_path=file_path,
                                args=args,
                                inference=inference,
                                generate_kwargs=generate_kwargs,
                                base_prompt=args.prompt,
                                batch_size=batch_size,
                                batch_input_cls=BatchInputItem,
                                images=[crop_image],
                                layout_results=[single_layout],
                                debug_dir=debug_dir,
                            )
                            ocr_chunks: list = []
                            if ocr_pages:
                                ocr_chunks = getattr(ocr_pages[0], "chunks", None) or []
                            if ocr_chunks:
                                ocr_markdown = ocr_chunks[0].get("markdown") or ""
                                ocr_html = ocr_chunks[0].get("html") or ocr_markdown
                                if ocr_markdown:
                                    markdown = ocr_markdown
                                    chunk["markdown"] = ocr_markdown
                                    chunk["html"] = ocr_html
                        except Exception:
                            pass
                    if "logo" in markdown.lower():
                        markdown = re.sub(r'<img[^>]*?>', '', markdown, flags=re.IGNORECASE)
                    # if "math" in content.lower():
                    #     content = re.sub(r'<math.*?</math>', '', content, flags=re.IGNORECASE | re.DOTALL)
                    if not markdown.strip():
                        continue
                    
                    html = markdown
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
        #################################################################################

        all_markdown: list[str] = []
        all_html: list[str] = []
        page_entries: list[dict[str, Any]] = []
        total_tokens = 0
        total_chunks = 0

        for idx, page_res in enumerate(page_outputs or [], 1):
            chunks = getattr(page_res, "chunks", None) or []
            token_count = getattr(page_res, "token_count", 0) or 0
            page_box = getattr(page_res, "page_box", []) or []

            page_md_blocks: list[str] = []
            page_html_blocks: list[str] = []
            for chunk in chunks:
                md_chunk = (chunk.get("markdown") or "").strip()
                html_chunk = (chunk.get("html") or "").strip()
                if md_chunk and "logo" in md_chunk.lower():
                    md_chunk = re.sub(r"<img[^>]*?>", "", md_chunk, flags=re.IGNORECASE)
                if md_chunk:
                    page_md_blocks.append(md_chunk)

                html_source = html_chunk or md_chunk
                if html_source:
                    if any(tag in html_source.lower() for tag in ["<table", "<p", "<html"]):
                        page_html_blocks.append(html_source)
                    else:
                        page_html_blocks.append(f"<p>{html_source}</p>")

            page_markdown = "\n\n".join(page_md_blocks) if page_md_blocks else ""
            page_html = (
                f"<html><body>{''.join(page_html_blocks)}</body></html>"
                if page_html_blocks
                else ""
            )

            # Keep the page result objects in sync for downstream save_merged_output.
            page_res.markdown = page_markdown
            page_res.html = page_html

            all_markdown.append(page_markdown)
            all_html.append(page_html)
            total_tokens += token_count
            total_chunks += len(chunks)

            page_entries.append(
                {
                    "page": idx,
                    "markdown": page_markdown,
                    "html": page_html,
                    "page_box": page_box,
                    "token_count": token_count,
                    "num_chunks": len(chunks),
                }
            )

        save_merged_output(
            results_dir,
            file_path.name,
            page_outputs,
            save_images=False,
            save_html=args.html,
            paginate_output=args.paginate_output,
        )

        if temp_pdf_path and temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except Exception:
                pass
        if layout_pdf_path and layout_pdf_path.exists():
            try:
                layout_pdf_path.unlink()
            except Exception:
                pass
        for img in page_images or []:
            try:
                img.close()
            except Exception:
                pass

        results_payload["pages"].extend(page_entries)
        results_payload["markdown"] = "".join(all_markdown)
        results_payload["html"] = "".join(all_html)
        results_payload["file_name"] = file_path.name
        results_payload["num_pages"] = len(page_outputs or [])
        results_payload["total_token_count"] = total_tokens
        results_payload["total_chunks"] = total_chunks
        # Save merged markdown/html for debugging.
        try:
            file_output_root.mkdir(parents=True, exist_ok=True)
            if results_payload["markdown"]:
                (file_output_root / f"{file_path.stem}_api.md").write_text(
                    results_payload["markdown"], encoding="utf-8"
                )
            if results_payload["html"]:
                (file_output_root / f"{file_path.stem}_api.html").write_text(
                    results_payload["html"], encoding="utf-8"
                )
        except Exception:
            pass

    return results_payload


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    output_root = Path(tempfile.mkdtemp())
    save_root = Path("./output")
    try:
        args = _build_args(
            input_path=tmp_path,
            output_dir=save_root,
        )
        result_payload = _process_file(tmp_path, args)
        markdown_content = result_payload.get("markdown", "") if isinstance(result_payload, dict) else ""
        return {"status": "success", "markdown": markdown_content}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.rmtree(output_root, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=900)
