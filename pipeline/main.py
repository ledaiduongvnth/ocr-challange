#!/usr/bin/env python3
"""Run Chandra OCR with a lightweight orientation normalization pre-pass."""

from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import dedent
from typing import List

from cli_utils import (
    apply_env_overrides,
    build_inference_options,
    determine_batch_size,
    parse_cli_args,
)
from native_pdf import build_native_outputs, is_digital_pdf
from ocr_pipeline import run_ocr_pipeline
from chandra_layout_analysis import chandra_analyze_layout
from pp_doclayout import analyze_layout_pp_doclayout
from pp_structure_preprocess import preprocess_with_ppstructure
from pp_structure_postprocess import postprocess_with_ppstructure
from utils import filter_non_text_chunks




def run():
    args = parse_cli_args()
    apply_env_overrides(args)
    print(f"CLI args: {args}")

    from chandra.input import load_file
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.scripts.cli import get_supported_files, save_merged_output

    files: List[Path] = get_supported_files(args.input_path)
    if not files:
        raise SystemExit(f"No supported files found under {args.input_path}")

    if args.output_dir.exists():
        try:
            shutil.rmtree(args.output_dir)
        except Exception as exc:
            print(f"Could not clear output dir {args.output_dir} ({exc}); continuing.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running ({args.method}) on {len(files)} file(s)...")

    inference = InferenceManager(method=args.method)
    batch_size = determine_batch_size(args)
    generate_kwargs = build_inference_options(args)

    for idx, file_path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {file_path.name}")
        is_pdf = file_path.suffix.lower() == ".pdf"
        is_native_pdf = is_pdf and is_digital_pdf(file_path)
        if is_pdf:
            print(f"  PDF type: {'native' if is_native_pdf else 'scanned'}")
        else:
            print("  Input type: image")

        page_images: list = []
        layout_results: list = []
        page_outputs = None
        merged_results = None
        file_output_root = args.output_dir / file_path.stem
        results_dir = file_output_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            load_config = {"page_range": args.page_range} if args.page_range else {}
            page_images = load_file(str(file_path), load_config)
            print(f"  [layout] loaded {len(page_images)} page(s)")
            debug_layout_dir = file_output_root if args.html else None
            debug_ocr_dir = file_output_root if args.html else None
            debug_native_dir = debug_ocr_dir
            if args.preprocess_backend == "ppstructure":
                print("  [preprocess] backend: ppstructure (orientation/unwarp)")
                page_images = preprocess_with_ppstructure(
                    page_images,
                    use_orientation=True,
                    use_unwarp=True,
                    debug_dir=debug_layout_dir,
                )

            # Run layout analysis using selected backend
            if args.layout_backend == "ppdoclayout":
                print("  [layout] backend: PP-DocLayout-L")
                _, layout_results = analyze_layout_pp_doclayout(
                    file_path=file_path,
                    images=page_images,
                    model_name="PP-DocLayout-L",
                    debug_dir=debug_layout_dir,
                )
            elif args.layout_backend == "ppdoclayout_plus":
                print("  [layout] backend: PP-DocLayout_plus-L")
                _, layout_results = analyze_layout_pp_doclayout(
                    file_path=file_path,
                    images=page_images,
                    model_name="PP-DocLayout_plus-L",
                    debug_dir=debug_layout_dir,
                )
            elif args.layout_backend == "PicoDet_layout_1x_table":
                print("  [layout] backend: PicoDet_layout_1x_table")
                _, layout_results = analyze_layout_pp_doclayout(
                    file_path=file_path,
                    images=page_images,
                    model_name="PicoDet_layout_1x_table",
                    debug_dir=debug_layout_dir,
                )
            else:
                print("  [layout] backend: chandra")
                _, layout_results = chandra_analyze_layout(
                    file_path=file_path,
                    images=page_images,
                    infer_fn=lambda items: inference.generate(items, **generate_kwargs),
                    prompt=None,
                    batch_size=batch_size,
                    debug_dir=debug_layout_dir,
                )
            if args.postprocess_backend == "ppstructure":
                layout_results = postprocess_with_ppstructure(
                    layout_results, images=page_images
                )
            layout_results = filter_non_text_chunks(layout_results)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Layout analysis failed ({exc}); continuing without layout hints.")

        match (is_native_pdf,):
            case (True,):
                page_outputs = build_native_outputs(
                    file_path,
                    layout_results=layout_results,
                    layout_images=page_images,
                    debug_dir=debug_native_dir,
                )
            case _:
                page_outputs = run_ocr_pipeline(
                    file_path=file_path,
                    args=args,
                    inference=inference,
                    generate_kwargs=generate_kwargs,
                    base_prompt=args.prompt,
                    batch_size=batch_size,
                    loader=load_file,
                    batch_input_cls=BatchInputItem,
                    images=page_images,
                    layout_results=layout_results,
                    debug_dir=debug_ocr_dir,
                )

        if args.paginate_output and page_outputs:
            for page_idx, page_res in enumerate(page_outputs, 1):
                page_dir = file_output_root / f"{page_idx:03d}" / "result"
                page_dir.mkdir(parents=True, exist_ok=True)
                save_merged_output(
                    page_dir,
                    f"{file_path.stem}_p{page_idx}",
                    [page_res],
                    save_images=args.include_images,
                    save_html=args.html,
                    paginate_output=False,
                )
        else:
            save_merged_output(
                results_dir,
                file_path.name,
                page_outputs,
                save_images=args.include_images,
                save_html=args.html,
                paginate_output=args.paginate_output,
            )


if __name__ == "__main__":
    run()
