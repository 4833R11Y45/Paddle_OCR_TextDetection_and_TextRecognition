#!/usr/bin/env python3
"""
Image Preprocessing for PaddleOCR
==================================
Handles image preparation before feeding into text detection:
- CLAHE contrast enhancement
- Scaling (respect PaddleOCR's 4000px hard limit)
- Tile generation with configurable overlap
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

from .config import OCRConfig

logger = logging.getLogger(__name__)


# ─── CLAHE Enhancement ──────────────────────────────────────────────────────

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a BGR image.

    Improves text visibility in low-contrast areas (e.g., faded blueprints).
    Detection runs on the enhanced image; recognition should use the original
    to preserve text colour/quality.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_image(image: np.ndarray, config: OCRConfig) -> np.ndarray:
    """
    Apply all configured preprocessing steps.

    Returns a new image; the original is never mutated.
    """
    out = image
    if config.use_clahe:
        out = apply_clahe(out, config.clahe_clip_limit, config.clahe_tile_grid)
    return out


# ─── Scaling ────────────────────────────────────────────────────────────────

def scale_image_if_needed(
    image: np.ndarray,
    max_side_px: int = 3800,
) -> Tuple[np.ndarray, float]:
    """
    Proportionally scale *image* so the longest side ≤ *max_side_px*.

    Returns:
        (scaled_image, scale_factor)
        scale_factor is 1.0 when no scaling was applied.
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side_px:
        return image, 1.0

    scale = max_side_px / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    logger.debug(f"Scaled {w}x{h} → {new_w}x{new_h} (factor {scale:.3f})")
    return scaled, scale


def scale_image_file(
    image_path: str,
    output_dir: Path,
    max_side_px: int = 3800,
) -> Tuple[str, int, int, int, int, bool]:
    """
    Scale an image *file* on disk (PIL-based, LANCZOS quality).

    Returns:
        (path_to_use, orig_w, orig_h, new_w, new_h, was_scaled)
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size

    if orig_w <= max_side_px and orig_h <= max_side_px:
        return image_path, orig_w, orig_h, orig_w, orig_h, False

    scale = max_side_px / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    scaled_path = output_dir / f"{stem}_scaled.png"

    with Image.open(image_path) as img:
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        scaled.save(str(scaled_path), dpi=(300, 300))

    logger.info(f"Scaled {Path(image_path).name}: {orig_w}x{orig_h} → {new_w}x{new_h}")
    return str(scaled_path), orig_w, orig_h, new_w, new_h, True


# ─── Tile Generation ────────────────────────────────────────────────────────

def generate_tiles(
    image: np.ndarray,
    tile_size: int = 1000,
    overlap: float = 0.50,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split *image* into overlapping square tiles.

    Args:
        image:     BGR image (np.ndarray).
        tile_size: Side length of each tile in pixels.
        overlap:   Fraction of overlap between adjacent tiles (0.0–1.0).

    Returns:
        List of ``(tile_image, x_offset, y_offset)`` tuples.
        Offsets are in original image coordinates.
    """
    h, w = image.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))
    tiles: List[Tuple[np.ndarray, int, int]] = []

    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = image[y:y_end, x:x_end]
            tiles.append((tile, x, y))
            x += stride
            if x >= w:
                break
        y += stride
        if y >= h:
            break

    logger.debug(f"Generated {len(tiles)} tiles ({tile_size}px, {overlap:.0%} overlap) from {w}x{h} image")
    return tiles
