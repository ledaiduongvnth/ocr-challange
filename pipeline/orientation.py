from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

_PADDLE_ORIENTATION_MODEL = None
_PADDLE_ORIENTATION_ERROR: Optional[Exception] = None


def _load_paddle_orientation_model():
    global _PADDLE_ORIENTATION_MODEL, _PADDLE_ORIENTATION_ERROR
    if _PADDLE_ORIENTATION_ERROR is not None:
        return None
    if _PADDLE_ORIENTATION_MODEL is None:
        try:
            from paddleocr import PPStructureV3

            _PADDLE_ORIENTATION_MODEL = PPStructureV3(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
            )
            print("Loaded Paddle document orientation classifier.")
        except Exception as exc:  # pragma: no cover - optional dependency
            _PADDLE_ORIENTATION_ERROR = exc
            print(f"     Paddle orientation classifier unavailable: {exc}")
            return None
    return _PADDLE_ORIENTATION_MODEL


def _extract_angle_from_result(res) -> Optional[int]:
    info = getattr(res, "doc_preprocessor_res", None)
    if info is None and hasattr(res, "to_dict"):
        info = res.to_dict().get("doc_preprocessor_res")
    if info is None and isinstance(res, dict):
        info = res.get("doc_preprocessor_res")
    if info is None:
        return None
    if hasattr(info, "to_dict"):
        info = info.to_dict()
    angle = None
    if isinstance(info, dict):
        angle = info.get("angle")
    else:
        angle = getattr(info, "angle", None)
    if angle is None:
        return None
    try:
        return int(round(float(angle)))
    except (TypeError, ValueError):
        return None


def detect_paddle_orientation(image: Image.Image) -> Optional[int]:
    model = _load_paddle_orientation_model()
    if model is None:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        preds = model.predict(input=tmp_path)
    except Exception as exc:  # pragma: no cover - external dependency
        print(f"     Paddle orientation detection failed: {exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if not preds:
        return None
    angle = _extract_angle_from_result(preds[0])
    return angle


def _describe_orientation(angle: int) -> str:
    normalized = angle % 360
    descriptions = {
        0: "upright (0°)",
        90: "rotated 90° CCW",
        180: "upside down (180°)",
        270: "rotated 90° CW",
    }
    return descriptions.get(normalized, f"rotated {normalized}°")


def normalize_image_orientation(image: Image.Image) -> Tuple[Image.Image, int]:
    """Rotate the image (0/90/180/270 CCW) so that text lines run horizontally."""
    paddle_angle = detect_paddle_orientation(image)
    if paddle_angle is None:
        print("     Paddle doc orientation unavailable; returning original image.")
        return image, 0
    normalized = paddle_angle % 360
    orientation_desc = _describe_orientation(normalized)
    print(f"     Paddle doc orientation angle: {normalized}° -> {orientation_desc}")
    rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
    return rotated, normalized


def normalize_page_images(
    images: Iterable[Image.Image],
    save_dir: Path | None = None,
    prefix: str = "page",
) -> List[Image.Image]:
    normalized = []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images, 1):
        fixed, angle = normalize_image_orientation(image)
        orientation_desc = _describe_orientation(angle)
        if angle:
            print(
                f"     page {idx}: {orientation_desc}; rotated {angle}° CCW to fix orientation"
            )
        else:
            print(f"     page {idx}: {orientation_desc}; orientation OK")
        if save_dir:
            out_path = save_dir / f"{prefix}_{idx:03d}.png"
            fixed.save(out_path)
            print(f"        saved rotated page -> {out_path}")
        normalized.append(fixed)
    return normalized
