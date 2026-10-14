"""
Tesseract OCR Engine wrapper for medical prescription baseline text recognition.
Supports full-image recognition and per-line segment recognition with configurable
Page Segmentation Modes (PSM) and confidence scoring.
"""

from dataclasses import dataclass, field
import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pytesseract

from src.preprocessing.pipeline import PreprocessingResult
from src.preprocessing.segmentation import LineSegment


def find_tesseract_binary() -> Optional[str]:
    """
    Locate tesseract executable in standard Windows paths or system PATH.
    """
    # 1. Environment variable
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 3. Standard Windows locations
    user_home = os.path.expanduser("~")
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(user_home, r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.join(user_home, r"AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    return None


# Configure pytesseract path
_tesseract_path = find_tesseract_binary()
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path


@dataclass
class OCRWord:
    """Represents a recognized word with bounding box and confidence."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class OCRLine:
    """Represents a recognized line with confidence and words."""
    line_index: int
    text: str
    confidence: float
    words: List[OCRWord] = field(default_factory=list)


@dataclass
class TesseractResult:
    """Full OCR recognition result comparing page-level vs line-level recognition."""
    full_page_text: str
    line_segmented_text: str
    lines: List[OCRLine] = field(default_factory=list)
    average_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TesseractOCREngine:
    """
    Baseline OCR engine using Tesseract (LSTM model).
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or find_tesseract_binary()
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def is_available(self) -> bool:
        """Check if Tesseract binary is found and executable."""
        cmd = find_tesseract_binary()
        if not cmd:
            return False
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def get_version(self) -> str:
        """Return installed Tesseract version."""
        if not self.is_available():
            return "Not Installed"
        return str(pytesseract.get_tesseract_version())

    def recognize_image(
        self,
        image: np.ndarray,
        psm: int = 6,
        oem: int = 1,
        whitelist: Optional[str] = None,
    ) -> str:
        """
        Run OCR on an image.
        
        Args:
            image: Grayscale, binary, or BGR image.
            psm: Page segmentation mode:
                 3 = Fully automatic page segmentation
                 6 = Assume a single uniform block of text
                 7 = Treat the image as a single text line
                 8 = Treat the image as a single word
                 11 = Sparse text
            oem: OCR Engine Mode (1 = LSTM neural net).
            whitelist: Optional character whitelist (e.g. alphanumeric + medical symbols).
        """
        config = f"--oem {oem} --psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"

        # If image is BGR, convert to RGB for Pillow/Tesseract
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        text = pytesseract.image_to_string(rgb, config=config)
        return text.strip()

    def recognize_with_confidence(
        self,
        image: np.ndarray,
        psm: int = 6,
    ) -> Tuple[str, float, List[OCRWord]]:
        """
        Extract text along with per-word bounding boxes and confidence scores.
        """
        config = f"--oem 1 --psm {psm}"
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        data = pytesseract.image_to_data(rgb, config=config, output_type=pytesseract.Output.DICT)
        
        words = []
        confidences = []
        assembled_words = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word_text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if word_text and conf > 0:
                words.append(OCRWord(
                    text=word_text,
                    confidence=conf,
                    bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                ))
                confidences.append(conf)
                assembled_words.append(word_text)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        full_text = " ".join(assembled_words)
        return full_text, avg_conf, words

    def process_prescription(
        self,
        prep_result: PreprocessingResult,
    ) -> TesseractResult:
        """
        Run both full-page OCR and per-line segment OCR on a PreprocessingResult.
        Line-by-line OCR uses PSM 7 (single line mode), which prevents line skipping
        and multi-column misalignment common in handwriting.
        """
        # 1. Full Page Recognition on binarized image (PSM 6)
        page_text, page_conf, page_words = self.recognize_with_confidence(prep_result.binarized, psm=6)

        # 2. Line-by-Line Recognition on cropped segments (PSM 7: Single line)
        line_results = []
        line_texts = []
        all_confs = []

        for line_seg in prep_result.lines:
            # We can run on either binary crop or grayscale/color crop; grayscale often gives better antialiasing to LSTM
            l_img = line_seg.image
            text_line, conf, words = self.recognize_with_confidence(l_img, psm=7)
            
            line_results.append(OCRLine(
                line_index=line_seg.line_index,
                text=text_line,
                confidence=conf,
                words=words,
            ))
            if text_line:
                line_texts.append(text_line)
            if conf > 0:
                all_confs.append(conf)

        avg_line_conf = float(np.mean(all_confs)) if all_confs else page_conf
        line_full_text = "\n".join(line_texts)

        return TesseractResult(
            full_page_text=page_text,
            line_segmented_text=line_full_text,
            lines=line_results,
            average_confidence=round(avg_line_conf, 2),
            metadata={
                "engine": "Tesseract (LSTM)",
                "version": self.get_version(),
                "num_lines_processed": len(prep_result.lines),
                "num_lines_with_text": len(line_texts),
            },
        )
