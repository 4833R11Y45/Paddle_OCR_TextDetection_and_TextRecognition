#!/usr/bin/env python3
"""
Visualization Helpers
=====================
Draw OCR results (bounding boxes, text labels, confidence scores)
onto images for debugging and QA.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .utils import TextBox


def draw_boxes(
    image: np.ndarray,
    boxes: List[TextBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_text: bool = True,
    show_confidence: bool = False,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detected text boxes on an image.

    Args:
        image:            BGR image (will be copied, not mutated).
        boxes:            List of TextBox results.
        color:            BGR color for boxes and text.
        thickness:        Line thickness.
        show_text:        Show recognized text above each box.
        show_confidence:  Append confidence score to label.
        font_scale:       Font scale for text labels.

    Returns:
        Annotated image copy.
    """
    vis = image.copy()

    for box in boxes:
        # Draw polygon
        pts = box.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        if show_text and box.text:
            x_min, y_min, _, _ = box.get_bbox()
            label = box.text
            if show_confidence:
                label += f" ({box.confidence:.2f})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Background rectangle for readability
            cv2.rectangle(vis, (x_min, y_min - th - 4), (x_min + tw + 4, y_min), (255, 255, 255), -1)
            cv2.putText(vis, label, (x_min + 2, y_min - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return vis


def draw_boxes_with_ids(
    image: np.ndarray,
    boxes: List[TextBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detected text boxes with sequential IDs for cross-referencing.

    Returns:
        Annotated image copy.
    """
    vis = image.copy()

    for idx, box in enumerate(boxes):
        pts = box.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        x_min, y_min, _, _ = box.get_bbox()
        label = f"[{idx}] {box.text}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x_min, y_min - th - 4), (x_min + tw + 4, y_min), (255, 255, 255), -1)
        cv2.putText(vis, label, (x_min + 2, y_min - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return vis


def save_annotated(
    image: np.ndarray,
    boxes: List[TextBox],
    output_path: str,
    **kwargs,
) -> str:
    """Draw boxes and save annotated image to disk."""
    vis = draw_boxes(image, boxes, **kwargs)
    cv2.imwrite(str(output_path), vis)
    return str(output_path)
