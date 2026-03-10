#!/usr/bin/env python3
"""
PaddleOCR Text Detector
========================
Thin wrapper around PaddleOCR 3.3.2 ``TextDetection``.
Handles model initialization, GPU memory cleanup, and result normalization.

NO PPStructure — only the lightweight DB-based text detection model.
"""

import gc
import logging
import os
import numpy as np
from typing import List, Tuple, Optional

from .config import OCRConfig

logger = logging.getLogger(__name__)


class TextDetector:
    """
    Text bounding-box detection using PaddleOCR ``TextDetection``.

    Outputs a list of (polygon, score) tuples per image.  Each polygon
    is an (N, 2) float32 array of vertex coordinates.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()

        if not self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from paddleocr import TextDetection

        logger.info(f"Initializing TextDetection (model={self.config.det_model_name}, "
                     f"device={self.config.device}) ...")

        self._model = TextDetection(
            model_name=self.config.det_model_name,
            device=self.config.device,
        )
        logger.info("TextDetection ready")

    # ── Public API ───────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        batch_size: int = 1,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Detect text regions in *image*.

        Args:
            image:      BGR numpy array.
            batch_size: Batch size passed to ``predict()``.

        Returns:
            List of ``(polygon, det_score)`` tuples.
            ``polygon`` is an ``np.ndarray`` with shape ``(N, 2)`` (float32).
        """
        results: List[Tuple[np.ndarray, float]] = []

        det_results = self._model.predict(image, batch_size=batch_size)

        for det_res in det_results:
            result_dict = det_res.res if hasattr(det_res, "res") else det_res

            dt_polys = result_dict.get("dt_polys")
            dt_scores = result_dict.get("dt_scores")

            if dt_polys is None or len(dt_polys) == 0:
                continue

            for poly, score in zip(dt_polys, dt_scores):
                poly_arr = np.array(poly, dtype=np.float32)
                results.append((poly_arr, float(score)))

        return results

    # ── GPU Cleanup ──────────────────────────────────────────────────────

    def cleanup_gpu(self):
        """Release cached GPU memory (safe no-op on CPU)."""
        if not self.config.use_gpu:
            return
        try:
            import paddle
            if hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "empty_cache"):
                paddle.device.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
