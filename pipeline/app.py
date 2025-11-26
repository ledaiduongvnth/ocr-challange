"""
FastAPI wrapper exposing the main OCR pipeline.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from chandra.input import load_file
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.scripts.cli import get_supported_files, save_merged_output
from chandra_layout_analysis import chandra_analyze_layout
from cli_utils import apply_env_overrides, build_inference_options, determine_batch_size
from native_pdf import build_native_outputs, is_digital_pdf
from ocr_pipeline import run_ocr_pipeline
from orientation import normalize_page_images
from pp_doclayout import analyze_layout_pp_doclayout
from pp_structure_postprocess import postprocess_with_ppstructure
from pp_structure_preprocess import preprocess_with_ppstructure
from surya_layout import analyze_layout_surya
from utils import log_component_bboxes

app = FastAPI()


def _build_args(input_path: Path, output_dir: Path) -> SimpleNamespace:
    """Construct an argparse-like namespace with main.py defaults."""
    return SimpleNamespace(
        input_path=input_path,
        output_dir=output_dir,
        checkpoint="datalab-to/chandra",
        method="vllm",
        page_range=None,
        batch_size=8,
        max_output_tokens=None,
        max_workers=None,
        max_retries=None,
        include_images=True,
        include_headers_footers=False,
        html=True,
        paginate_output=False,
        device=None,
        attn_impl=None,
        layout_backend="surya",
        preprocess_backend="ppstructure",
        postprocess_backend="ppstructure",
        prompt="default",
    )


def _process_file(file_path: Path, args: SimpleNamespace) -> dict[str, Any]:
    apply_env_overrides(args)
    files: List[Path] = get_supported_files(args.input_path)
    if not files:
        raise RuntimeError(f"No supported files found under {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inference = InferenceManager(method=args.method)
    batch_size = determine_batch_size(args)
    generate_kwargs = build_inference_options(args)
    results_payload: dict[str, Any] = {"pages": []}

    for file_path in files:
        is_pdf = file_path.suffix.lower() == ".pdf"
        is_native_pdf = is_pdf and is_digital_pdf(file_path)

        file_output_root = args.output_dir / file_path.stem
        results_dir = file_output_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        load_config = {"page_range": args.page_range} if args.page_range else {}
        temp_pdf_path: Path | None = None
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

        layout_results: list = []
        if args.layout_backend == "ppdoclayout":
            _, layout_results = analyze_layout_pp_doclayout(
                file_path=source_path,
                images=page_images,
                model_name="PP-DocLayout-L",
                debug_dir=debug_dir,
            )
        elif args.layout_backend == "ppdoclayout_plus":
            _, layout_results = analyze_layout_pp_doclayout(
                file_path=source_path,
                images=page_images,
                model_name="PP-DocLayout_plus-L",
                debug_dir=debug_dir,
            )
        elif args.layout_backend == "PicoDet_layout_1x_table":
            _, layout_results = analyze_layout_pp_doclayout(
                file_path=source_path,
                images=page_images,
                model_name="PicoDet_layout_1x_table",
                debug_dir=debug_dir,
            )
        elif args.layout_backend == "surya":
            _, layout_results = analyze_layout_surya(
                file_path=source_path,
                images=page_images,
                debug_dir=debug_dir,
            )
        else:
            _, layout_results = chandra_analyze_layout(
                file_path=source_path,
                images=page_images,
                infer_fn=lambda items: inference.generate(items, **generate_kwargs),
                prompt=None,
                batch_size=batch_size,
                debug_dir=debug_dir,
            )

        log_component_bboxes(file_path.name, layout_results)

        if args.postprocess_backend == "ppstructure":
            layout_results = postprocess_with_ppstructure(layout_results, images=page_images)

        page_outputs = (
            build_native_outputs(
                file_path,
                layout_results=layout_results,
                layout_images=page_images,
                debug_dir=debug_dir,
            )
            if is_native_pdf
            else run_ocr_pipeline(
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
        )

        merged_md = []
        merged_html = []
        for idx, page_res in enumerate(page_outputs or [], 1):
            merged_md.append(getattr(page_res, "markdown", "") or "")
            merged_html.append(getattr(page_res, "html", "") or "")
            results_payload["pages"].append(
                {
                    "page": idx,
                    "markdown": getattr(page_res, "markdown", "") or "",
                    "html": getattr(page_res, "html", "") or "",
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
        for img in page_images or []:
            try:
                img.close()
            except Exception:
                pass

        results_payload["markdown"] = "\n\n".join(md for md in merged_md if md)
        results_payload["html"] = "".join(h for h in merged_html if h)

    return results_payload


@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    output_dir = Path(tempfile.mkdtemp())
    try:
        args = _build_args(input_path=tmp_path, output_dir=output_dir)
        result_payload = _process_file(tmp_path, args)
        return {"status": "success", "data": result_payload}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.rmtree(output_dir, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
