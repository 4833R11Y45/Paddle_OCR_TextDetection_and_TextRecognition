#!/usr/bin/env python3
"""
PaddleOCR Engine — End-to-End Text Detection + Recognition
============================================================
Combines ``TextDetector`` and ``TextRecognizer`` into a single
high-level API.  Supports:

- Single-image OCR
- Tile-based OCR for large images (respects PaddleOCR's 4000px limit)
- Automatic scaling
- Configurable preprocessing and post-processing

NO PPStructure — uses only TextDetection + TextRecognition.

Usage:
    from app.paddle_ocr import PaddleOCREngine, OCRConfig

    config = OCRConfig(use_gpu=True, tile_size=1000, tile_overlap=0.50)
    engine = PaddleOCREngine(config)

    # Simple single-image OCR
    results = engine.run(image)
    for box in results:
        print(box.text, box.confidence, box.get_bbox())

    # Tile-based OCR for large images
    results = engine.run_tiled(large_image)

    # OCR from file path with auto-scaling
    results = engine.run_file("/path/to/image.png")
"""

import gc
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import OCRConfig
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .preprocessing import (
    preprocess_image,
    scale_image_if_needed,
    generate_tiles,
)
from .postprocessing import postprocess
from .utils import TextBox, to_serializable

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """
    End-to-end PaddleOCR pipeline: detect text → crop regions → recognize text.

    This is the primary public interface.  It owns a ``TextDetector`` and
    ``TextRecognizer`` and orchestrates the full pipeline.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        logger.info("Initializing PaddleOCREngine ...")
        self.detector = TextDetector(self.config)
        self.recognizer = TextRecognizer(self.config)
        logger.info("PaddleOCREngine ready")

    # ── Core: single-image OCR ──────────────────────────────────────────

    def run(
        self,
        image: np.ndarray,
        preprocess: bool = True,
    ) -> List[TextBox]:
        """
        Run text detection + recognition on a single image.

        Args:
            image:      BGR numpy array.
            preprocess: Apply configured preprocessing (CLAHE, etc.).

        Returns:
            List of ``TextBox`` with text, coordinates, and scores.
        """
        original = image
        det_image = preprocess_image(image, self.config) if preprocess else image

        # 1. Detect text regions
        detections = self.detector.detect(det_image)
        if not detections:
            return []

        # 2. Crop regions from ORIGINAL image (preserves text quality)
        h, w = original.shape[:2]
        crops = []
        valid_indices = []

        for idx, (poly, det_score) in enumerate(detections):
            poly_int = poly.astype(np.int32)
            x_min = max(0, int(poly_int[:, 0].min()))
            y_min = max(0, int(poly_int[:, 1].min()))
            x_max = min(w, int(poly_int[:, 0].max()))
            y_max = min(h, int(poly_int[:, 1].max()))

            if x_max > x_min and y_max > y_min:
                crop = original[y_min:y_max, x_min:x_max]
                crops.append(crop)
                valid_indices.append(idx)

        if not crops:
            return []

        # 3. Recognize text
        rec_results = self.recognizer.recognize(crops, batch_size=self.config.rec_batch_size)

        # 4. Build TextBox results
        boxes: List[TextBox] = []
        for rec_idx, (text, rec_score) in enumerate(rec_results):
            det_idx = valid_indices[rec_idx]
            poly, det_score = detections[det_idx]

            boxes.append(TextBox(
                polygon=poly,
                text=text,
                det_score=det_score,
                rec_score=rec_score,
            ))

        # 5. Post-process
        boxes = postprocess(
            boxes,
            min_confidence=self.config.min_confidence,
            min_width=self.config.min_box_width,
            min_height=self.config.min_box_height,
            dedup_distance=self.config.dedup_distance,
        )

        return boxes

    # ── Tile-based OCR for large images ─────────────────────────────────

    def run_tiled(
        self,
        image: np.ndarray,
        tile_size: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> List[TextBox]:
        """
        Tile-based OCR for large images that exceed PaddleOCR's 4000px limit.

        Text detection runs on CLAHE-enhanced tiles for better localization.
        Text recognition crops regions from the *original* image for quality.

        Args:
            image:     BGR numpy array (any size).
            tile_size: Override config tile size.
            overlap:   Override config tile overlap.

        Returns:
            Deduplicated list of ``TextBox`` across all tiles.
        """
        ts = tile_size or self.config.tile_size
        ov = overlap or self.config.tile_overlap

        original = image
        det_image = preprocess_image(image, self.config)

        tiles = generate_tiles(det_image, tile_size=ts, overlap=ov)
        logger.info(f"Processing {len(tiles)} tiles ({ts}px, {ov:.0%} overlap)")

        h, w = original.shape[:2]
        all_boxes: List[TextBox] = []

        for tile_img, x_off, y_off in tiles:
            # Detect on preprocessed tile
            detections = self.detector.detect(tile_img)
            if not detections:
                continue

            # Crop from ORIGINAL image using absolute coordinates
            crops = []
            valid_det = []

            for poly, det_score in detections:
                poly_abs = poly.copy()
                poly_abs[:, 0] += x_off
                poly_abs[:, 1] += y_off

                poly_int = poly_abs.astype(np.int32)
                x_min = max(0, int(poly_int[:, 0].min()))
                y_min = max(0, int(poly_int[:, 1].min()))
                x_max = min(w, int(poly_int[:, 0].max()))
                y_max = min(h, int(poly_int[:, 1].max()))

                if x_max > x_min and y_max > y_min:
                    crop = original[y_min:y_max, x_min:x_max]
                    crops.append(crop)
                    valid_det.append((poly_abs, det_score))

            if not crops:
                continue

            # Recognize text
            rec_results = self.recognizer.recognize(crops, batch_size=self.config.rec_batch_size)

            for rec_idx, (text, rec_score) in enumerate(rec_results):
                poly_abs, det_score = valid_det[rec_idx]
                all_boxes.append(TextBox(
                    polygon=poly_abs,
                    text=text,
                    det_score=det_score,
                    rec_score=rec_score,
                    tile_offset=(x_off, y_off),
                ))

        # Post-process (dedup overlapping tile results)
        all_boxes = postprocess(
            all_boxes,
            min_confidence=self.config.min_confidence,
            min_width=self.config.min_box_width,
            min_height=self.config.min_box_height,
            dedup_distance=self.config.dedup_distance,
            merge_fragments=True,
        )

        logger.info(f"Tile OCR complete: {len(all_boxes)} text regions")
        return all_boxes

    # ── File-based OCR with auto-scaling ────────────────────────────────

    def run_file(
        self,
        image_path: str,
        use_tiling: bool = True,
    ) -> List[TextBox]:
        """
        Run OCR on an image file.  Automatically scales large images.

        Args:
            image_path: Path to image file.
            use_tiling:  Use tile-based OCR (recommended for large images).

        Returns:
            List of ``TextBox`` results.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        # Scale if needed
        img, scale = scale_image_if_needed(img, self.config.max_side_px)

        if use_tiling:
            boxes = self.run_tiled(img)
        else:
            boxes = self.run(img)

        # If we scaled, map coordinates back to original space
        if scale < 1.0:
            inv_scale = 1.0 / scale
            for box in boxes:
                box.polygon = (box.polygon * inv_scale).astype(np.float32)

        return boxes

    # ── Batch processing ────────────────────────────────────────────────

    def run_batch(
        self,
        image_paths: List[str],
        use_tiling: bool = True,
    ) -> Dict[str, List[TextBox]]:
        """
        Run OCR on multiple image files.

        Args:
            image_paths: List of image file paths.
            use_tiling:  Use tile-based OCR.

        Returns:
            Dict mapping each image path to its list of ``TextBox`` results.
        """
        results: Dict[str, List[TextBox]] = {}

        for i, path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{len(image_paths)}] {Path(path).name}")
            results[path] = self.run_file(path, use_tiling=use_tiling)
            self.cleanup_gpu()

        return results

    # ── Convenience: extract plain text ─────────────────────────────────

    def extract_text(
        self,
        image: np.ndarray,
        use_tiling: bool = False,
        join: str = " ",
    ) -> str:
        """
        One-liner: get all text from an image as a single string.

        Args:
            image:      BGR numpy array.
            use_tiling:  Use tile-based OCR for large images.
            join:       Separator between text boxes.

        Returns:
            Concatenated recognized text.
        """
        boxes = self.run_tiled(image) if use_tiling else self.run(image)
        # Sort top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b.get_bbox()[1], b.get_bbox()[0]))
        return join.join(b.text for b in boxes if b.text.strip())

    # ── Results to JSON ─────────────────────────────────────────────────

    @staticmethod
    def results_to_dicts(boxes: List[TextBox]) -> List[Dict[str, Any]]:
        """Convert TextBox list to JSON-serializable dicts."""
        return to_serializable([b.to_dict() for b in boxes])

    # ── GPU Cleanup ──────────────────────────────────────────────────────

    def cleanup_gpu(self):
        """Release GPU memory from both detector and recognizer."""
        self.detector.cleanup_gpu()
        self.recognizer.cleanup_gpu()

    def shutdown(self):
        """Full cleanup: release models and free GPU memory."""
        self.cleanup_gpu()
        self.detector = None
        self.recognizer = None
        gc.collect()
        try:
            import paddle
            if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                paddle.device.cuda.empty_cache()
        except Exception:
            pass
        logger.info("PaddleOCREngine shut down")
