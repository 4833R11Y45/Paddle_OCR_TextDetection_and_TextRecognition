#!/usr/bin/env python3
"""
PaddleOCR Module — Text Detection & Text Recognition
=====================================================
Standalone PaddleOCR wrappers for the project.
Uses PaddleOCR 3.3.2 TextDetection and TextRecognition only.
NO PPStructure / PPStructureV3.

Public API:
    from app.paddle_ocr import PaddleOCREngine, TextDetector, TextRecognizer
    from app.paddle_ocr import OCRConfig
    from app.paddle_ocr import preprocess_image, generate_tiles
"""

from .config import OCRConfig
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .ocr_engine import PaddleOCREngine

__all__ = [
    "OCRConfig",
    "TextDetector",
    "TextRecognizer",
    "PaddleOCREngine",
]
