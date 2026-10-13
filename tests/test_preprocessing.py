"""
Unit tests for OpenCV Preprocessing Pipeline (Week 1).
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.illumination import remove_shadows_morphological, apply_clahe, correct_illumination
from src.preprocessing.deskew import rotate_image, detect_skew_projection_profile, deskew
from src.preprocessing.binarization import denoise_bilateral, sauvola_threshold, binarize, clean_speckles
from src.preprocessing.segmentation import segment_prescription, segment_lines_morphological
from src.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic white document image with black text stripes
        self.height, self.width = 400, 500
        self.test_img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        # Draw simulated text lines
        for y in [100, 160, 220, 280]:
            cv2.putText(
                self.test_img,
                "Sample Medical Rx Line",
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
            )

    def test_illumination_correction(self):
        # Add gradient shadow to image
        gradient = np.tile(np.linspace(0.4, 1.0, self.width), (self.height, 1))
        shadowed = (self.test_img.astype(np.float32) * gradient[:, :, None]).astype(np.uint8)

        corrected = correct_illumination(shadowed)
        self.assertEqual(corrected.shape, shadowed.shape)
        self.assertEqual(corrected.dtype, np.uint8)
        # Background should be significantly brighter after correction
        self.assertGreater(np.mean(corrected), np.mean(shadowed))

    def test_rotation_and_deskew(self):
        # Rotate image by 5 degrees
        rotated = rotate_image(self.test_img, 5.0)
        self.assertGreaterEqual(rotated.shape[0], self.height)
        self.assertGreaterEqual(rotated.shape[1], self.width)

        # Deskew back
        deskewed, detected_angle = deskew(rotated, method="projection")
        self.assertAlmostEqual(detected_angle, -5.0, delta=1.5)

    def test_binarization_methods(self):
        gray = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
        
        # Test Sauvola
        sauv = sauvola_threshold(gray, window_size=25, k=0.2)
        self.assertEqual(sauv.shape, gray.shape)
        self.assertTrue(set(np.unique(sauv)).issubset({0, 255}))

        # Test Adaptive Gaussian & Otsu
        otsu = binarize(gray, method="otsu")
        self.assertTrue(set(np.unique(otsu)).issubset({0, 255}))

        # Test speckle removal
        cleaned = clean_speckles(sauv)
        self.assertEqual(cleaned.shape, sauv.shape)

    def test_segmentation(self):
        gray = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        lines = segment_prescription(binary, self.test_img, method="morphological")
        # We drew 4 lines, expecting 4 line segments
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertGreater(len(line.words), 0)
            self.assertEqual(len(line.image.shape), 3)

    def test_end_to_end_pipeline(self):
        pipeline = PreprocessingPipeline(mode="photo", binarization_method="sauvola")
        result = pipeline.process(self.test_img)

        self.assertIsNotNone(result.binarized)
        self.assertIsNotNone(result.deskewed)
        self.assertEqual(result.num_lines, 4)
        self.assertIn("timings_sec", result.metadata)


if __name__ == "__main__":
    unittest.main()
