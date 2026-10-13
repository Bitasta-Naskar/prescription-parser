"""
Image Preprocessing Package for Prescription Parser.
"""

from src.preprocessing.illumination import (
    remove_shadows_morphological,
    apply_clahe,
    correct_illumination,
)
from src.preprocessing.deskew import (
    rotate_image,
    detect_skew_projection_profile,
    detect_skew_hough,
    deskew,
)
from src.preprocessing.binarization import (
    denoise_bilateral,
    sauvola_threshold,
    binarize,
    clean_speckles,
)
from src.preprocessing.segmentation import (
    WordSegment,
    LineSegment,
    segment_lines_morphological,
    segment_lines_projection,
    segment_words_in_line,
    segment_prescription,
)
from src.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingResult,
)

__all__ = [
    "remove_shadows_morphological",
    "apply_clahe",
    "correct_illumination",
    "rotate_image",
    "detect_skew_projection_profile",
    "detect_skew_hough",
    "deskew",
    "denoise_bilateral",
    "sauvola_threshold",
    "binarize",
    "clean_speckles",
    "WordSegment",
    "LineSegment",
    "segment_lines_morphological",
    "segment_lines_projection",
    "segment_words_in_line",
    "segment_prescription",
    "PreprocessingPipeline",
    "PreprocessingResult",
]
