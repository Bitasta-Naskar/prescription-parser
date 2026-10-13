"""
Deskew module for document and prescription orientation correction.
Implements Hough line detection, contour minAreaRect, and projection profile variance.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def rotate_image(
    image: np.ndarray,
    angle: float,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Rotate image by angle (in degrees) around its center, expanding borders so no text is cropped.
    
    Args:
        image: Input image (BGR or Grayscale).
        angle: Angle in degrees (positive = counter-clockwise).
        background_color: Fill color for newly exposed corners (default white).
        
    Returns:
        Rotated image.
    """
    if abs(angle) < 0.1:
        return image.copy()
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    border_val = background_color[0] if len(image.shape) == 2 else background_color
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


def detect_skew_projection_profile(
    binary_inv: np.ndarray,
    angle_range: Tuple[float, float] = (-20.0, 20.0),
    step: float = 0.5,
) -> float:
    """
    Find skew angle by maximizing variance of the horizontal projection profile.
    
    When text lines are parallel to the horizontal axis, the horizontal projection profile
    shows high peaks (text rows) and deep valleys (inter-line spaces), maximizing variance.
    
    Args:
        binary_inv: Inverted binary image (text is white 255, background is black 0).
        angle_range: (min_angle, max_angle) in degrees to search.
        step: Search step size in degrees.
        
    Returns:
        Estimated skew angle in degrees.
    """
    angles = np.arange(angle_range[0], angle_range[1] + step, step)
    best_angle = 0.0
    max_variance = -1.0
    
    (h, w) = binary_inv.shape[:2]
    center = (w // 2, h // 2)

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_inv,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        # Horizontal projection profile: sum pixels along rows
        profile = np.sum(rotated, axis=1)
        # Compute differences between adjacent rows or variance
        variance = np.sum((profile[1:] - profile[:-1]) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = float(angle)
            
    return best_angle


def detect_skew_hough(
    binary_inv: np.ndarray,
    max_angle: float = 30.0,
) -> float:
    """
    Estimate skew angle using probabilistic Hough Line Transform on text strokes.
    
    Args:
        binary_inv: Inverted binary image (text white, background black).
        max_angle: Maximum expected skew angle in degrees to consider.
        
    Returns:
        Median angle of detected near-horizontal line segments.
    """
    edges = cv2.Canny(binary_inv, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=int(binary_inv.shape[1] * 0.1),
        maxLineGap=20,
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        # Filter for near-horizontal lines
        if abs(angle_deg) < max_angle:
            angles.append(angle_deg)
        elif abs(angle_deg - 180) < max_angle:
            angles.append(angle_deg - 180)
        elif abs(angle_deg + 180) < max_angle:
            angles.append(angle_deg + 180)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(
    image: np.ndarray,
    method: str = "projection",
    angle_range: Tuple[float, float] = (-25.0, 25.0),
    step: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Detect skew and rotate image to upright horizontal orientation.
    
    Args:
        image: Input image (BGR or Grayscale).
        method: 'projection' (default, most robust for documents) or 'hough'.
        angle_range: Range of angles to search if using projection.
        step: Search resolution for projection method.
        
    Returns:
        (deskewed_image, angle_corrected)
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create inverted binary image for analysis (text is white)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if method == "hough":
        angle = detect_skew_hough(thresh, max_angle=abs(angle_range[1]))
    else:
        # Resize small for fast projection profile search if large image
        h, w = thresh.shape
        scale = 1.0
        if w > 1000:
            scale = 1000.0 / w
            small_thresh = cv2.resize(thresh, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        else:
            small_thresh = thresh
            
        angle = detect_skew_projection_profile(small_thresh, angle_range=angle_range, step=step)

    deskewed = rotate_image(image, angle)
    return deskewed, angle
