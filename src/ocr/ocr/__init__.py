"""
OCR Engine Package.
"""

from src.ocr.tesseract_engine import (
    find_tesseract_binary,
    TesseractOCREngine,
    TesseractResult,
    OCRLine,
    OCRWord,
)
from src.ocr.evaluator import (
    compute_cer,
    compute_wer,
    evaluate_ocr,
    EvaluationReport,
)

__all__ = [
    "find_tesseract_binary",
    "TesseractOCREngine",
    "TesseractResult",
    "OCRLine",
    "OCRWord",
    "compute_cer",
    "compute_wer",
    "evaluate_ocr",
    "EvaluationReport",
]
