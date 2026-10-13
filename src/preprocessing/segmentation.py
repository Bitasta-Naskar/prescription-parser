"""
Line and word segmentation module for handwritten prescription documents.
Segments lines using morphological dilation and horizontal projection profiles,
and breaks lines into word/token crops for OCR inference.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class WordSegment:
    """Represents a segmented word or token."""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) relative to full image
    line_index: int
    word_index: int


@dataclass
class LineSegment:
    """Represents a segmented line of handwriting."""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    line_index: int
    words: List[WordSegment]


def segment_lines_morphological(
    binary: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    kernel_width: int = 45,
    kernel_height: int = 5,
    min_line_height: int = 15,
    min_line_width: int = 40,
    pad: int = 4,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment text lines using horizontal morphological smearing followed by contour extraction.
    
    Args:
        binary: Binary image where text is black (0) and background is white (255).
        original_image: Source image to crop from (if None, crops from binary).
        kernel_width: Horizontal width of structuring element for smearing words into lines.
        kernel_height: Vertical height of structuring element.
        min_line_height: Minimum bounding box height to be considered a text line.
        min_line_width: Minimum bounding box width.
        pad: Padding pixels around line bounding box.
        
    Returns:
        List of (line_crop, bbox) sorted top-to-bottom, where bbox is (x, y, w, h).
    """
    source = original_image if original_image is not None else binary
    h_img, w_img = binary.shape[:2]

    # Invert binary so text is white (255)
    text_mask = cv2.bitwise_not(binary)

    # Horizontal dilation: connects adjacent characters and words on the same line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(text_mask, kernel, iterations=2)

    # Find contours representing smeared lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_line_height and w >= min_line_width:
            boxes.append((x, y, w, h))

    # Sort lines top-to-bottom
    boxes.sort(key=lambda b: b[1])

    line_crops = []
    for x, y, w, h in boxes:
        # Apply padding safely within image boundaries
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        crop = source[y1:y2, x1:x2]
        line_crops.append((crop, (x1, y1, x2 - x1, y2 - y1)))

    return line_crops


def segment_lines_projection(
    binary: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    smooth_kernel: int = 15,
    threshold_ratio: float = 0.05,
    min_line_height: int = 15,
    pad: int = 4,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment text lines using horizontal projection profile valley detection.
    
    Args:
        binary: Binary image (text=0, bg=255).
        original_image: Image to crop from.
        smooth_kernel: Gaussian 1D kernel size for smoothing profile.
        threshold_ratio: Ratio of max profile value to identify inter-line valleys.
        min_line_height: Minimum height of a detected line.
        pad: Vertical padding.
        
    Returns:
        List of (line_crop, bbox) sorted top-to-bottom.
    """
    source = original_image if original_image is not None else binary
    h_img, w_img = binary.shape[:2]

    # Inverted: text is 255
    inv = (binary == 0).astype(np.float32)
    profile = np.sum(inv, axis=1)

    # Smooth profile using 1D Gaussian kernel
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, smooth_kernel), 0).flatten()

    valley_thresh = np.max(smoothed) * threshold_ratio
    in_line = False
    start_y = 0
    line_ranges = []

    for y, val in enumerate(smoothed):
        if val > valley_thresh and not in_line:
            in_line = True
            start_y = y
        elif val <= valley_thresh and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                line_ranges.append((start_y, y))

    if in_line and (h_img - start_y) >= min_line_height:
        line_ranges.append((start_y, h_img))

    line_crops = []
    for y1, y2 in line_ranges:
        py1 = max(0, y1 - pad)
        py2 = min(h_img, y2 + pad)
        crop = source[py1:py2, 0:w_img]
        line_crops.append((crop, (0, py1, w_img, py2 - py1)))

    return line_crops


def segment_words_in_line(
    line_binary: np.ndarray,
    line_crop: np.ndarray,
    line_bbox: Tuple[int, int, int, int],
    line_index: int,
    kernel_size: Tuple[int, int] = (10, 3),
    min_word_width: int = 12,
    min_word_height: int = 10,
    pad: int = 2,
) -> List[WordSegment]:
    """
    Segment individual words or medicine dosage tokens within a single text line crop.
    
    Args:
        line_binary: Binarized slice of the line (text=0, bg=255).
        line_crop: RGB/Grayscale slice of the line.
        line_bbox: Absolute (x, y, w, h) bounding box of the line.
        line_index: Index of current line.
        kernel_size: Morphological structuring element size for grouping characters into words.
        min_word_width: Minimum word bounding width.
        min_word_height: Minimum word bounding height.
        pad: Padding around word bounding box.
        
    Returns:
        List of WordSegment objects sorted left-to-right.
    """
    line_h, line_w = line_binary.shape[:2]
    line_x, line_y, _, _ = line_bbox

    # Text is 255
    text_mask = cv2.bitwise_not(line_binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(text_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_word_width and h >= min_word_height:
            boxes.append((x, y, w, h))

    # Sort left to right
    boxes.sort(key=lambda b: b[0])

    words = []
    for idx, (x, y, w, h) in enumerate(boxes):
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(line_w, x + w + pad)
        y2 = min(line_h, y + h + pad)

        word_crop = line_crop[y1:y2, x1:x2]
        abs_bbox = (line_x + x1, line_y + y1, x2 - x1, y2 - y1)
        words.append(WordSegment(
            image=word_crop,
            bbox=abs_bbox,
            line_index=line_index,
            word_index=idx,
        ))

    return words


def segment_prescription(
    binary: np.ndarray,
    original_image: np.ndarray,
    method: str = "morphological",
    segment_words: bool = True,
) -> List[LineSegment]:
    """
    Hierarchical segmentation of prescription into structured LineSegment and WordSegment objects.
    
    Args:
        binary: Binarized image.
        original_image: Clean or color image for crops.
        method: 'morphological' (default) or 'projection'.
        segment_words: Whether to further segment words inside each line.
        
    Returns:
        List of LineSegment objects containing line crops and word crops.
    """
    if method == "projection":
        lines_raw = segment_lines_projection(binary, original_image)
    else:
        lines_raw = segment_lines_morphological(binary, original_image)

    line_segments = []
    for line_idx, (line_img, line_bbox) in enumerate(lines_raw):
        # Extract binary for the same bounding box
        x, y, w, h = line_bbox
        line_bin = binary[y:y+h, x:x+w]

        words = []
        if segment_words and line_bin.size > 0:
            words = segment_words_in_line(
                line_binary=line_bin,
                line_crop=line_img,
                line_bbox=line_bbox,
                line_index=line_idx,
            )

        line_segments.append(LineSegment(
            image=line_img,
            bbox=line_bbox,
            line_index=line_idx,
            words=words,
        ))

    return line_segments
