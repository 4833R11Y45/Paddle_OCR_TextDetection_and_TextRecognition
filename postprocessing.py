#!/usr/bin/env python3
"""
Post-processing for PaddleOCR Detections
=========================================
Deduplication, filtering, and fragment merging utilities
applied after text detection + recognition.
"""

import logging
import numpy as np
from typing import List

from .utils import TextBox

logger = logging.getLogger(__name__)


# ─── Filtering ──────────────────────────────────────────────────────────────

def filter_by_size(
    boxes: List[TextBox],
    min_width: int = 15,
    min_height: int = 8,
) -> List[TextBox]:
    """Remove text boxes that are too small to contain meaningful text."""
    return [b for b in boxes if b.width >= min_width and b.height >= min_height]


def filter_by_confidence(
    boxes: List[TextBox],
    min_confidence: float = 0.20,
) -> List[TextBox]:
    """Remove text boxes below minimum combined confidence."""
    return [b for b in boxes if b.confidence >= min_confidence]


# ─── Deduplication ──────────────────────────────────────────────────────────

def deduplicate_by_location(
    boxes: List[TextBox],
    distance_threshold: float = 50.0,
) -> List[TextBox]:
    """
    Remove duplicate detections from overlapping tiles.

    Keeps the *highest-confidence* box when two boxes are within
    ``distance_threshold`` pixels of each other.
    """
    if len(boxes) <= 1:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep: List[TextBox] = []

    for box in sorted_boxes:
        cx, cy = box.get_center()
        too_close = False
        for kept in keep:
            kcx, kcy = kept.get_center()
            dist = np.sqrt((cx - kcx) ** 2 + (cy - kcy) ** 2)
            if dist < distance_threshold:
                too_close = True
                break
        if not too_close:
            keep.append(box)

    if len(keep) < len(boxes):
        logger.debug(f"Dedup: {len(boxes)} → {len(keep)} (threshold={distance_threshold}px)")

    return keep


def deduplicate_by_iou(
    boxes: List[TextBox],
    iou_threshold: float = 0.5,
) -> List[TextBox]:
    """
    Remove overlapping detections using IoU (Intersection over Union).

    Keeps the highest-confidence box when two boxes have IoU above the threshold.
    """
    if len(boxes) <= 1:
        return boxes

    from .utils import boxes_iou

    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep: List[TextBox] = []

    for box in sorted_boxes:
        bbox = box.get_bbox()
        suppressed = False
        for kept in keep:
            if boxes_iou(bbox, kept.get_bbox()) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(box)

    if len(keep) < len(boxes):
        logger.debug(f"IoU NMS: {len(boxes)} → {len(keep)} (threshold={iou_threshold})")

    return keep


# ─── Fragment Merging ────────────────────────────────────────────────────────

def merge_horizontal_fragments(
    boxes: List[TextBox],
    max_horizontal_gap: int = 100,
    max_vertical_diff: int = 25,
) -> List[TextBox]:
    """
    Merge text fragments that were split across tile boundaries.

    When an image is tiled for OCR, words near tile edges often get split
    into separate detections (e.g., "10'" and "-6\""). This function
    merges horizontally adjacent fragments back together.

    Args:
        boxes:              All detected text boxes.
        max_horizontal_gap: Maximum horizontal gap (px) between left and right fragments.
        max_vertical_diff:  Maximum vertical difference (px) for same-line check.

    Returns:
        New list of TextBox with merged fragments replacing originals.
    """
    if len(boxes) <= 1:
        return boxes

    # Sort left-to-right, then top-to-bottom
    sorted_boxes = sorted(boxes, key=lambda b: (b.get_bbox()[1], b.get_bbox()[0]))
    used = set()
    merged: List[TextBox] = []

    for i, left in enumerate(sorted_boxes):
        if i in used:
            continue

        left_bbox = left.get_bbox()
        left_right_edge = left_bbox[2]
        left_cy = (left_bbox[1] + left_bbox[3]) / 2.0

        best_j = None
        best_gap = float("inf")

        for j, right in enumerate(sorted_boxes):
            if j <= i or j in used:
                continue

            right_bbox = right.get_bbox()
            right_left_edge = right_bbox[0]
            right_cy = (right_bbox[1] + right_bbox[3]) / 2.0

            h_gap = right_left_edge - left_right_edge
            v_diff = abs(right_cy - left_cy)

            if -10 < h_gap < max_horizontal_gap and v_diff < max_vertical_diff:
                if h_gap < best_gap:
                    best_gap = h_gap
                    best_j = j

        if best_j is not None:
            right = sorted_boxes[best_j]
            right_bbox = right.get_bbox()

            # Merge text
            merged_text = left.text.strip() + right.text.strip()

            # Merge bounding polygon
            merged_x_min = min(left_bbox[0], right_bbox[0])
            merged_y_min = min(left_bbox[1], right_bbox[1])
            merged_x_max = max(left_bbox[2], right_bbox[2])
            merged_y_max = max(left_bbox[3], right_bbox[3])

            merged_poly = np.array([
                [merged_x_min, merged_y_min],
                [merged_x_max, merged_y_min],
                [merged_x_max, merged_y_max],
                [merged_x_min, merged_y_max],
            ], dtype=np.float32)

            merged_box = TextBox(
                polygon=merged_poly,
                text=merged_text,
                det_score=min(left.det_score, right.det_score),
                rec_score=min(left.rec_score, right.rec_score),
                tile_offset=(0, 0),
            )
            merged.append(merged_box)
            used.add(i)
            used.add(best_j)
        else:
            merged.append(left)
            used.add(i)

    # Catch any remaining unprocessed boxes
    for i, box in enumerate(sorted_boxes):
        if i not in used:
            merged.append(box)

    merge_count = len(boxes) - len(merged)
    if merge_count > 0:
        logger.debug(f"Merged {merge_count} fragment pairs → {len(merged)} boxes")

    return merged


# ─── Convenience Pipeline ────────────────────────────────────────────────────

def postprocess(
    boxes: List[TextBox],
    min_confidence: float = 0.20,
    min_width: int = 15,
    min_height: int = 8,
    dedup_distance: float = 50.0,
    merge_fragments: bool = True,
) -> List[TextBox]:
    """
    Full post-processing pipeline:
      1.  Filter by confidence
      2.  Filter by minimum box size
      3.  Merge horizontal fragments (tile-boundary splits)
      4.  Deduplicate by spatial proximity

    Args:
        boxes:           Raw TextBox results.
        min_confidence:  Minimum combined confidence.
        min_width:       Minimum box width (px).
        min_height:      Minimum box height (px).
        dedup_distance:  Center-distance threshold for deduplication (px).
        merge_fragments: Whether to attempt fragment merging.

    Returns:
        Cleaned list of TextBox.
    """
    result = filter_by_confidence(boxes, min_confidence)
    result = filter_by_size(result, min_width, min_height)
    if merge_fragments:
        result = merge_horizontal_fragments(result)
    result = deduplicate_by_location(result, dedup_distance)
    return result
