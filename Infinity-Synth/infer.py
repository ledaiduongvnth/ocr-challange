import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
USER_PROMPT = [{"role": "user", "content": [{"type": "image"}]}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for the model trained by Infinity-Synth/finetune.py")
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("OCR_INFER_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Directory created by trainer.save_model().",
    )
    parser.add_argument("--output", default=None, help="Output file (.json or .jsonl). Prints stdout if omitted.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--longest-edge", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-new-tokens-cap",
        type=int,
        default=8192,
        help="Upper bound for auto-expanding max_new_tokens.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count when generation appears truncated by token limit.",
    )
    parser.add_argument("--device", default="auto", help='Device: "auto", "cpu", "cuda", "cuda:0", ...')
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def patch_mistral_common_compat() -> None:
    try:
        import mistral_common.tokens.tokenizers.utils as token_utils
    except Exception:
        return

    if hasattr(token_utils, "get_one_valid_tokenizer_file") or not hasattr(
        token_utils, "_filter_valid_tokenizer_files"
    ):
        return

    def _get_one_valid_tokenizer_file(files: List[str]) -> str:
        valid_files = token_utils._filter_valid_tokenizer_files(files)
        if not valid_files:
            raise ValueError("No valid tokenizer file found.")
        return "tekken.json" if "tekken.json" in valid_files else sorted(valid_files)[-1]

    token_utils.get_one_valid_tokenizer_file = _get_one_valid_tokenizer_file


def model_kwargs_for_device(device: torch.device, trust_remote_code: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
    }
    if device.type == "cuda":
        kwargs["attn_implementation"] = "sdpa"
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def try_load(model_dir: Path, device: torch.device, processor_cls: Any, model_cls: Any, trust_remote_code: bool):
    processor = processor_cls.from_pretrained(
        model_dir, **({"trust_remote_code": True} if trust_remote_code else {})
    )
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
    model = model_cls.from_pretrained(model_dir, **model_kwargs_for_device(device, trust_remote_code)).to(device)
    model.eval()
    return processor, model


def load_processor_and_model(model_dir: Path, device: torch.device) -> Tuple[Any, Any]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    patch_mistral_common_compat()

    import transformers as tr

    attempts: List[Tuple[Any, Any, bool, str]] = []
    auto_processor = getattr(tr, "AutoProcessor", None)
    if auto_processor is None:
        raise ImportError("AutoProcessor is not available in this transformers build.")

    lighton_processor = getattr(tr, "LightOnOcrProcessor", None)
    lighton_model = getattr(tr, "LightOnOcrForConditionalGeneration", None)
    if lighton_processor is not None and lighton_model is not None:
        attempts.append((lighton_processor, lighton_model, False, "LightOnOcr*"))

    for class_name in (
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
    ):
        model_cls = getattr(tr, class_name, None)
        if model_cls is not None:
            attempts.append((auto_processor, model_cls, True, f"AutoProcessor + {class_name}"))

    errors: List[str] = []
    for processor_cls, model_cls, trust_remote_code, label in attempts:
        try:
            processor, model = try_load(model_dir, device, processor_cls, model_cls, trust_remote_code)
            print(f"Model loader: {label}")
            return processor, model
        except Exception as error:  # noqa: BLE001
            errors.append(f"{label}: {error}")

    raise RuntimeError(
        "Failed to load processor/model.\n"
        + "\n".join(errors)
        + "\nInstall dependencies from Infinity-Synth/requirement.txt in the same environment."
    )


def collect_image_paths(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext == ".pdf":
            raise ValueError("PDF input is not supported. Please provide an image file.")
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Allowed: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
        return [input_path]

    image_paths = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not image_paths:
        raise ValueError(f"No images found in {input_path}. Allowed: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
    return image_paths


def parse_json_or_none(text: str) -> Tuple[Any, str | None]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return None, str(error)


def generate_text(
    processor: Any,
    model: Any,
    image: Image.Image,
    device: torch.device,
    max_length: int,
    longest_edge: int,
    max_new_tokens: int,
) -> Tuple[str, int]:
    prompt = processor.apply_chat_template(USER_PROMPT, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[[image]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        size={"longest_edge": longest_edge},
    ).to(device)
    if "pixel_values" in inputs and device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_len:]
    pred_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return pred_text, generated.shape[1]


def infer_one(
    processor: Any,
    model: Any,
    image_path: Path,
    device: torch.device,
    max_length: int,
    longest_edge: int,
    max_new_tokens: int,
    max_new_tokens_cap: int,
    max_retries: int,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    current_max_new_tokens = max_new_tokens
    retries = 0
    pred_text = ""
    pred_json = None
    parse_error = None
    generated_tokens = 0
    truncated_by_limit = False

    for attempt in range(max_retries + 1):
        pred_text, generated_tokens = generate_text(
            processor=processor,
            model=model,
            image=image,
            device=device,
            max_length=max_length,
            longest_edge=longest_edge,
            max_new_tokens=current_max_new_tokens,
        )
        pred_json, parse_error = parse_json_or_none(pred_text)
        truncated_by_limit = generated_tokens >= current_max_new_tokens

        if pred_json is not None:
            break
        if not truncated_by_limit:
            break
        if current_max_new_tokens >= max_new_tokens_cap:
            break
        if attempt == max_retries:
            break

        current_max_new_tokens = min(current_max_new_tokens * 2, max_new_tokens_cap)
        retries += 1

    return {
        "image": str(image_path),
        "text": pred_text,
        "json": pred_json,
        "json_valid": pred_json is not None,
        "json_parse_error": parse_error,
        "generated_tokens": generated_tokens,
        "used_max_new_tokens": current_max_new_tokens,
        "truncated_by_token_limit": truncated_by_limit,
        "retries": retries,
    }


def save_results(results: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_dir = Path(args.model_dir).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()

    image_paths = collect_image_paths(input_path)
    print(f"Using device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Found {len(image_paths)} image(s)")

    processor, model = load_processor_and_model(model_dir, device)

    results = [
        infer_one(
            processor=processor,
            model=model,
            image_path=image_path,
            device=device,
            max_length=args.max_length,
            longest_edge=args.longest_edge,
            max_new_tokens=args.max_new_tokens,
            max_new_tokens_cap=args.max_new_tokens_cap,
            max_retries=args.max_retries,
        )
        for image_path in tqdm(image_paths, desc="Infer")
    ]

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        save_results(results, output_path)
        print(f"Saved predictions to: {output_path}")
        return

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
