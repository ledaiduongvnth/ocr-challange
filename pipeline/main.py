#!/usr/bin/env python3
"""Run Chandra OCR with a lightweight orientation normalization pre-pass."""

from __future__ import annotations

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


CUSTOM_PROMPT_SUFFIX = dedent(
    """\
    Note:
    - If a cell in the table is empty (no text or empty string), keep the cell in the markdown.
    - Header or footer lines can stick together; remember to separate them.
    """
)


def run():
    args = parse_cli_args()
    apply_env_overrides(args)

    from chandra.input import load_file
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.prompts import PROMPT_MAPPING
    from chandra.scripts.cli import get_supported_files, save_merged_output

    files: List[Path] = get_supported_files(args.input_path)
    if not files:
        raise SystemExit(f"No supported files found under {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running Chandra ({args.method}) with checkpoint '{args.checkpoint}' on {len(files)} file(s)...")

    inference = InferenceManager(method=args.method)
    batch_size = determine_batch_size(args)
    generate_kwargs = build_inference_options(args)
    base_prompt = f"{PROMPT_MAPPING['ocr_layout']}{CUSTOM_PROMPT_SUFFIX}"

    for idx, file_path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {file_path.name}")
        is_pdf = file_path.suffix.lower() == ".pdf"
        is_native_pdf = is_pdf and is_digital_pdf(file_path)

        results = None
        if is_native_pdf:
            results = build_native_outputs(file_path)

        if results is None:
            results = run_ocr_pipeline(
                file_path=file_path,
                args=args,
                inference=inference,
                generate_kwargs=generate_kwargs,
                base_prompt=base_prompt,
                batch_size=batch_size,
                loader=load_file,
                batch_input_cls=BatchInputItem,
            )

        save_merged_output(
            args.output_dir,
            file_path.name,
            results,
            save_images=True,
            save_html=True,
            paginate_output=False,
        )


if __name__ == "__main__":
    run()
