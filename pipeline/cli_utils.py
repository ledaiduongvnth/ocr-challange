import argparse
import os
from pathlib import Path


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chandra OCR runner with orientation detection.")
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("/home/zsv/Desktop/test-native.pdf"),
        help="File or directory containing PDFs/images. Defaults to /home/zsv/Desktop/test-native.pdf if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to store markdown/html/json outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="datalab-to/chandra",
        help="Hugging Face repo id to load (default: datalab-to/chandra).",
    )
    parser.add_argument(
        "--method",
        choices=("hf", "vllm"),
        default="vllm",
        help="Inference backend. Use 'hf' for local GPU, 'vllm' for server.",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Subset of pages for PDFs, e.g. '1-3,5'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Pages per batch (default: 1).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override max generated tokens per page.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers when using vllm backend.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retry count for vllm backend.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        default=False,
        help="Store cropped images referenced in output markdown.",
    )
    parser.add_argument(
        "--include-headers-footers",
        action="store_true",
        default=False,
        help="Keep detected headers/footers in the output.",
    )
    parser.add_argument("--no-html", action="store_true", help="Skip writing HTML alongside markdown.")
    parser.add_argument(
        "--paginate-output",
        action="store_true",
        help="Insert page separators into markdown/html.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. 'cuda:0'.")
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Optional attention implementation (flash_attention_2, etc.).",
    )
    parser.add_argument(
        "--layout-backend",
        choices=("chandra", "ppdoclayout", "PicoDet_layout_1x_table"),
        default="PicoDet_layout_1x_table",
        help="Layout analysis backend: 'chandra', 'ppdoclayout' (PP-DocLayout-L), or 'PicoDet_layout_1x_table'.",
    )
    return parser.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    os.environ["MODEL_CHECKPOINT"] = args.checkpoint
    if args.device:
        os.environ["TORCH_DEVICE"] = args.device
    if args.attn_impl:
        os.environ["TORCH_ATTN"] = args.attn_impl


def build_inference_options(args: argparse.Namespace) -> dict:
    options = {
        "include_images": args.include_images,
        "include_headers_footers": args.include_headers_footers,
    }
    if args.max_output_tokens is not None:
        options["max_output_tokens"] = args.max_output_tokens
    if args.method == "vllm":
        if args.max_workers is not None:
            options["max_workers"] = args.max_workers
        if args.max_retries is not None:
            options["max_retries"] = args.max_retries
    return options


def determine_batch_size(args: argparse.Namespace) -> int:
    """Pick a batch size based on CLI args and backend defaults."""
    if args.batch_size and args.batch_size > 0:
        return args.batch_size
    if args.method == "vllm":
        return 28
    return 1
