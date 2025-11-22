from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
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


def _compute_projection_metrics(image: Image.Image) -> Tuple[float, float, float, float]:
    """Return (horizontal_variance, vertical_variance, top-bottom balance, left-right balance)."""
    arr = np.array(image.convert("L"))
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = cv2.GaussianBlur(arr, (5, 5), 0)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - thresh  # make text areas high-valued
    horiz = ink.sum(axis=1).astype(np.float64)
    vert = ink.sum(axis=0).astype(np.float64)
    height, width = ink.shape
    horiz_var = float(np.var(horiz) / (height * height + 1e-6))
    vert_var = float(np.var(vert) / (width * width + 1e-6))
    band = max(1, height // 4)
    top = float(horiz[:band].sum())
    bottom = float(horiz[-band:].sum())
    side_band = max(1, width // 4)
    left = float(vert[:side_band].sum())
    right = float(vert[-side_band:].sum())
    return horiz_var, vert_var, top - bottom, left - right


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


def _select_best_rotation(
    image: Image.Image, angles: Sequence[int], preferred: Set[int]
) -> Tuple[Image.Image, int]:
    best_angle = 0
    best_image = image
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    seen = set()
    for angle in angles:
        normalized = int(angle) % 360
        if normalized in seen:
            continue
        seen.add(normalized)
        rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
        horiz_var, vert_var, top_bottom, left_right = _compute_projection_metrics(rotated)
        alignment_score = horiz_var - vert_var
        rotation_penalty = -abs(normalized if normalized <= 180 else 360 - normalized)
        key = (
            alignment_score,
            1 if normalized in preferred else 0,
            left_right,
            top_bottom,
            rotation_penalty,
        )
        if key > best_key:
            best_key = key
            best_angle = normalized
            best_image = rotated
    return best_image, best_angle


def normalize_image_orientation(image: Image.Image) -> Tuple[Image.Image, int]:
    """Rotate the image (0/90/180/270 CCW) so that text lines run horizontally."""
    paddle_angle = detect_paddle_orientation(image)
    if paddle_angle is not None:
        normalized = paddle_angle % 360
        print(f"     Paddle doc orientation angle: {normalized}°")
        rotated = image if normalized == 0 else image.rotate(normalized, expand=True)
        return rotated, normalized
    return _select_best_rotation(image, (0, 90, 180, 270), set())


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
        if angle:
            print(f"     page {idx}: rotated {angle}° CCW to fix orientation")
        else:
            print(f"     page {idx}: orientation OK")
        if save_dir:
            out_path = save_dir / f"{prefix}_{idx:03d}.png"
            fixed.save(out_path)
            print(f"        saved rotated page -> {out_path}")
        normalized.append(fixed)
    return normalized
