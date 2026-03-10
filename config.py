#!/usr/bin/env python3
"""
PaddleOCR Configuration
=======================
Central configuration for text detection and text recognition.
All tunable parameters live here.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRConfig:
    """Configuration for the PaddleOCR text detection + recognition pipeline."""

    # ── Device ──────────────────────────────────────────────────────────
    use_gpu: bool = True
    device: str = ""  # Auto-resolved from use_gpu if empty

    # ── Model Selection ─────────────────────────────────────────────────
    det_model_name: str = "PP-OCRv5_server_det"
    rec_model_name: str = "PP-OCRv5_server_rec"

    # ── Detection Thresholds ────────────────────────────────────────────
    det_db_thresh: float = 0.25       # Binarization threshold for DB detector
    det_db_box_thresh: float = 0.5    # Box score threshold
    det_limit_side_len: int = 20000   # Max side length before internal resize

    # ── Recognition ─────────────────────────────────────────────────────
    rec_batch_size: int = 8           # Batch size for recognition inference
    rec_score_threshold: float = 0.5  # Minimum confidence for accepted text

    # ── Combined Confidence ─────────────────────────────────────────────
    min_confidence: float = 0.20      # Minimum combined (det × rec) confidence

    # ── Tiling ──────────────────────────────────────────────────────────
    tile_size: int = 1000             # Tile side length (px)
    tile_overlap: float = 0.50        # Overlap ratio between tiles (0.0–1.0)

    # ── Image Size Limits ───────────────────────────────────────────────
    max_side_px: int = 3800           # Scale down images above this (PaddleOCR 4000px limit)

    # ── Preprocessing ───────────────────────────────────────────────────
    use_clahe: bool = False           # CLAHE contrast enhancement
    clahe_clip_limit: float = 1.5     # CLAHE clip limit
    clahe_tile_grid: tuple = (8, 8)   # CLAHE tile grid size

    # ── Post-processing ─────────────────────────────────────────────────
    dedup_distance: float = 50.0      # Distance threshold for duplicate removal (px)
    min_box_width: int = 15           # Minimum box width to keep
    min_box_height: int = 8           # Minimum box height to keep

    def __post_init__(self):
        if not self.device:
            self.device = "gpu" if self.use_gpu else "cpu"
