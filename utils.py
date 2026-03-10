#!/usr/bin/env python3
"""
PaddleOCR Utilities
===================
Common data structures, box operations, and serialization helpers
shared across text detection and recognition components.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class TextBox:
    """A single detected text region with its recognized text."""

    polygon: np.ndarray       # Shape (4, 2) — four vertices (x, y)
    text: str                 # Recognized text
    det_score: float          # Detection confidence
    rec_score: float          # Recognition confidence
    tile_offset: Tuple[int, int] = (0, 0)  # Offset from tiling

    @property
    def confidence(self) -> float:
        """Combined confidence = min(detection, recognition)."""
        return min(self.det_score, self.rec_score)

    def get_center(self) -> Tuple[float, float]:
        """Center point (cx, cy) of the polygon."""
        return float(np.mean(self.polygon[:, 0])), float(np.mean(self.polygon[:, 1]))

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box: (x_min, y_min, x_max, y_max)."""
        xs = self.polygon[:, 0]
        ys = self.polygon[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @property
    def width(self) -> int:
        x_min, _, x_max, _ = self.get_bbox()
        return x_max - x_min

    @property
    def height(self) -> int:
        _, y_min, _, y_max = self.get_bbox()
        return y_max - y_min

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        x_min, y_min, x_max, y_max = self.get_bbox()
        cx, cy = self.get_center()
        return {
            "text": self.text,
            "bbox": [x_min, y_min, x_max, y_max],
            "polygon": self.polygon.tolist(),
            "center": [cx, cy],
            "det_score": round(self.det_score, 4),
            "rec_score": round(self.rec_score, 4),
            "confidence": round(self.confidence, 4),
        }


# ─── Box Operations ─────────────────────────────────────────────────────────

def box_area(bbox: Tuple[int, int, int, int]) -> int:
    """Area of an (x_min, y_min, x_max, y_max) bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def boxes_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def boxes_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Euclidean distance between the centers of two bounding boxes."""
    cx_a = (a[0] + a[2]) / 2.0
    cy_a = (a[1] + a[3]) / 2.0
    cx_b = (b[0] + b[2]) / 2.0
    cy_b = (b[1] + b[3]) / 2.0
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


# ─── Serialization ──────────────────────────────────────────────────────────

def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [to_serializable(i) for i in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj
