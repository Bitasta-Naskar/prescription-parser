"""
Binarization and denoising module for medical prescriptions.
Includes Otsu, Adaptive Gaussian, and an efficient Sauvola thresholding implementation,
plus stroke-preserving bilateral denoising and morphological speckle removal.
"""

from typing import Literal
import cv2
import numpy as np


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Bilateral filter: smooths paper grain while preserving sharp ink stroke edges.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def sauvola_threshold(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """
    Vectorized Sauvola adaptive binarization using OpenCV boxFilter.
    
    Sauvola is the gold standard for degraded document and handwriting binarization:
    T(x, y) = mean(x, y) * (1 + k * (std(x, y) / R - 1))
    
    Args:
        gray: Grayscale input image (uint8).
        window_size: Size of local neighborhood window (must be odd).
        k: Sensitivity parameter (typically 0.2 to 0.5).
        r: Dynamic range of standard deviation (128 for 8-bit images).
        
    Returns:
        Binary image (0 = ink, 255 = paper background).
    """
    if window_size % 2 == 0:
        window_size += 1

    gray_f = gray.astype(np.float32)

    # Compute local mean
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REFLECT)

    # Compute local standard deviation: std = sqrt(E[X^2] - (E[X])^2)
    sq_mean = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(sq_mean - (mean ** 2), 0)
    std = np.sqrt(variance)

    # Sauvola formula
    threshold_map = mean * (1.0 + k * ((std / r) - 1.0))

    # Apply threshold
    binary = np.where(gray_f > threshold_map, 255, 0).astype(np.uint8)
    return binary


def binarize(
    gray: np.ndarray,
    method: Literal["otsu", "adaptive_gaussian", "sauvola"] = "sauvola",
    block_size: int = 25,
    c_param: int = 10,
    sauvola_k: float = 0.2,
) -> np.ndarray:
    """
    Binarize grayscale image with specified thresholding technique.
    
    Args:
        gray: Grayscale image.
        method: 'otsu', 'adaptive_gaussian', or 'sauvola'.
        block_size: Window size for adaptive algorithms (odd integer).
        c_param: Constant subtracted from mean in adaptive gaussian.
        sauvola_k: Sensitivity factor for Sauvola.
        
    Returns:
        Binary image (0 = text, 255 = background).
    """
    if block_size % 2 == 0:
        block_size += 1

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c_param,
        )
    elif method == "sauvola":
        binary = sauvola_threshold(gray, window_size=block_size, k=sauvola_k)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


def clean_speckles(
    binary: np.ndarray,
    min_area: int = 10,
    max_area_ratio: float = 0.95,
) -> np.ndarray:
    """
    Remove isolated salt-and-pepper noise specks from binarized image using connected components.
    
    Args:
        binary: Binary image where background is 255 and text is 0.
        min_area: Minimum pixel area for a valid ink component (smaller specks deleted).
        max_area_ratio: Ratio above which a component is considered background.
        
    Returns:
        Cleaned binary image.
    """
    # Invert so ink components are foreground (255)
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    h, w = binary.shape[:2]
    total_area = h * w
    cleaned_inv = np.zeros_like(inv)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Keep components that are not tiny specks and not entire image background
        if min_area <= area <= total_area * max_area_ratio:
            cleaned_inv[labels == i] = 255

    return cv2.bitwise_not(cleaned_inv)
