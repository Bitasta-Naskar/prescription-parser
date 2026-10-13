"""
Unified Preprocessing Pipeline for medical prescription images.
Orchestrates illumination correction, deskewing, binarization, denoising,
and hierarchical line/word segmentation for both scanned documents and mobile photos.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import time
import cv2
import numpy as np

from src.preprocessing.illumination import correct_illumination
from src.preprocessing.deskew import deskew
from src.preprocessing.binarization import denoise_bilateral, binarize, clean_speckles
from src.preprocessing.segmentation import segment_prescription, LineSegment


@dataclass
class PreprocessingResult:
    """Encapsulates the complete output of the preprocessing pipeline."""
    original: np.ndarray
    illumination_corrected: Optional[np.ndarray]
    deskewed: np.ndarray
    binarized: np.ndarray
    skew_angle: float
    lines: List[LineSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    @property
    def num_words(self) -> int:
        return sum(len(line.words) for line in self.lines)


class PreprocessingPipeline:
    """
    End-to-end image preprocessing pipeline configured for medical prescriptions.
    Supports both mobile camera photos (with shadows and lighting gradients)
    and flatbed document scans.
    """

    def __init__(
        self,
        mode: Literal["photo", "scan", "auto"] = "photo",
        binarization_method: Literal["sauvola", "adaptive_gaussian", "otsu"] = "sauvola",
        deskew_method: str = "projection",
        clean_speckles_flag: bool = True,
        segment_words_flag: bool = True,
    ):
        self.mode = mode
        self.binarization_method = binarization_method
        self.deskew_method = deskew_method
        self.clean_speckles_flag = clean_speckles_flag
        self.segment_words_flag = segment_words_flag

    def process(self, image: np.ndarray, mode: Optional[str] = None) -> PreprocessingResult:
        """
        Run full preprocessing pipeline on input image.
        
        Args:
            image: Input image (BGR or Grayscale).
            mode: Override default mode ('photo' or 'scan').
            
        Returns:
            PreprocessingResult containing cleaned images and segmented lines/words.
        """
        t0 = time.time()
        active_mode = mode or self.mode
        timings: Dict[str, float] = {}

        if len(image.shape) == 2:
            # Convert grayscale to BGR for consistent multi-channel operations
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # Step 1: Illumination Correction (Photos only)
        t_step = time.time()
        if active_mode in ("photo", "auto"):
            illum_corrected = correct_illumination(image_bgr, use_morphology=True, use_clahe=True)
            timings["illumination_correction"] = round(time.time() - t_step, 4)
        else:
            illum_corrected = None
            timings["illumination_correction"] = 0.0

        target_for_deskew = illum_corrected if illum_corrected is not None else image_bgr

        # Step 2: Deskew Orientation Correction
        t_step = time.time()
        deskewed_img, skew_angle = deskew(target_for_deskew, method=self.deskew_method)
        timings["deskew"] = round(time.time() - t_step, 4)

        # Step 3: Grayscale Conversion & Denoising
        t_step = time.time()
        if len(deskewed_img.shape) == 3:
            gray = cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = deskewed_img.copy()

        denoised = denoise_bilateral(gray, d=7, sigma_color=50, sigma_space=50)
        timings["denoising"] = round(time.time() - t_step, 4)

        # Step 4: Binarization
        t_step = time.time()
        binary = binarize(denoised, method=self.binarization_method, block_size=25, sauvola_k=0.2)
        if self.clean_speckles_flag:
            binary = clean_speckles(binary, min_area=8)
        timings["binarization"] = round(time.time() - t_step, 4)

        # Step 5: Hierarchical Line and Word Segmentation
        t_step = time.time()
        line_segments = segment_prescription(
            binary=binary,
            original_image=deskewed_img,
            method="morphological",
            segment_words=self.segment_words_flag,
        )
        timings["segmentation"] = round(time.time() - t_step, 4)
        timings["total_pipeline_time"] = round(time.time() - t0, 4)

        metadata = {
            "mode": active_mode,
            "skew_angle_deg": round(skew_angle, 2),
            "binarization_method": self.binarization_method,
            "num_lines": len(line_segments),
            "num_words": sum(len(line.words) for line in line_segments),
            "timings_sec": timings,
        }

        return PreprocessingResult(
            original=image_bgr,
            illumination_corrected=illum_corrected,
            deskewed=deskewed_img,
            binarized=binary,
            skew_angle=skew_angle,
            lines=line_segments,
            metadata=metadata,
        )
