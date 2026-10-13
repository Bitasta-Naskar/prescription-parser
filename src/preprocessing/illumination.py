"""
Illumination and shadow correction module for document photos.
Handles lighting gradients, shadows (e.g. overhead lamps, phone/hand shadows),
and contrast degradation typical in mobile captures of prescription paper.
"""

from typing import Tuple
import cv2
import numpy as np


def remove_shadows_morphological(
    image: np.ndarray,
    kernel_size: int = 25,
    blur_kernel: int = 15,
) -> np.ndarray:
    """
    Remove uneven lighting and shadows using morphological background division.
    
    Dilates the image with a large kernel to estimate the paper background,
    blurs the background estimate, and divides the original image by it.
    
    Args:
        image: Grayscale or BGR image (uint8).
        kernel_size: Size of square structuring element for morphological dilation.
        blur_kernel: Size of Gaussian blur kernel applied to background map.
        
    Returns:
        Normalized image with shadows removed and even background (same channels as input).
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    
    if is_color:
        # Process each channel separately or in LAB color space
        planes = cv2.split(image)
        result_planes = []
        for plane in planes:
            dilated = cv2.dilate(plane, np.ones((kernel_size, kernel_size), np.uint8))
            bg = cv2.medianBlur(dilated, blur_kernel)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(norm)
        return cv2.merge(result_planes)
    else:
        # Grayscale division
        dilated = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))
        bg = cv2.medianBlur(dilated, blur_kernel)
        diff = 255 - cv2.absdiff(image, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return norm


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance local contrast using Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        image: Grayscale or BGR image (uint8).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram equalization (default 8x8).
        
    Returns:
        Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def correct_illumination(
    image: np.ndarray,
    use_morphology: bool = True,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Full illumination normalization pipeline for mobile prescription photos.
    
    Combines morphological background division with CLAHE contrast enhancement.
    """
    processed = image.copy()
    if use_morphology:
        processed = remove_shadows_morphological(processed)
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clip_limit)
    return processed
