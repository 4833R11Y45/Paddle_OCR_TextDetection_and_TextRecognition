#!/usr/bin/env python3
"""
PaddleOCR Text Recognizer
==========================
Thin wrapper around PaddleOCR 3.3.2 ``TextRecognition``.
Accepts cropped text-region images and returns ``(text, score)`` pairs.

NO PPStructure — only the lightweight CRNN/SVTR recognition model.
"""

import gc
import logging
import os
import numpy as np
from typing import List, Tuple, Optional

from .config import OCRConfig

logger = logging.getLogger(__name__)


class TextRecognizer:
    """
    Text recognition on cropped text-region images.

    Each input should be a tightly cropped BGR numpy array containing
    a single line (or word) of text.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()

        if not self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from paddleocr import TextRecognition

        logger.info(f"Initializing TextRecognition (model={self.config.rec_model_name}, "
                     f"device={self.config.device}) ...")

        self._model = TextRecognition(
            model_name=self.config.rec_model_name,
            device=self.config.device,
        )
        logger.info("TextRecognition ready")

    # ── Public API ───────────────────────────────────────────────────────

    def recognize(
        self,
        crops: List[np.ndarray],
        batch_size: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Recognize text in a list of cropped images.

        Args:
            crops:      List of BGR numpy arrays (one per text region).
            batch_size: Override config batch size for this call.

        Returns:
            List of ``(text, rec_score)`` tuples, one per crop (same order).
        """
        if not crops:
            return []

        bs = batch_size or self.config.rec_batch_size
        results: List[Tuple[str, float]] = []

        rec_results = self._model.predict(crops, batch_size=bs)

        for rec_res in rec_results:
            rec_dict = rec_res.res if hasattr(rec_res, "res") else rec_res
            text = rec_dict.get("rec_text", "")
            score = float(rec_dict.get("rec_score", 0.0))
            results.append((text, score))

        return results

    def recognize_single(self, crop: np.ndarray) -> Tuple[str, float]:
        """Convenience: recognize a single crop image."""
        results = self.recognize([crop], batch_size=1)
        return results[0] if results else ("", 0.0)

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
