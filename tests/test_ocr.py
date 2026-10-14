"""
Unit tests for OCR Engine and Evaluator (Week 1-2).
"""

import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ocr.evaluator import compute_cer, compute_wer, evaluate_ocr, levenshtein_distance
from src.ocr.tesseract_engine import TesseractOCREngine, OCRWord, OCRLine


class TestOCREvaluator(unittest.TestCase):

    def test_levenshtein_distance(self):
        dist = levenshtein_distance(list("kitten"), list("sitting"))
        self.assertEqual(dist, 3)

    def test_cer_wer_identical(self):
        ref = "Tab. Paracetamol 500 mg 1-0-1"
        hyp = "Tab. Paracetamol 500 mg 1-0-1"
        self.assertEqual(compute_cer(ref, hyp), 0.0)
        self.assertEqual(compute_wer(ref, hyp), 0.0)

    def test_cer_on_handwriting_typo(self):
        # Realistic OCR handwriting typo: "Paracetmol" vs "Paracetamol"
        ref = "Paracetamol"
        hyp = "Paracetmol"
        cer = compute_cer(ref, hyp)
        # 1 deletion out of 11 characters
        self.assertAlmostEqual(cer, 1 / 11, places=2)

    def test_wer_calculation(self):
        ref = "1. Tab. Paracetamol 500mg 1-0-1 x 5 days"
        hyp = "1. Tab. Paracetamol 500mg 101 x 5 days"
        wer = compute_wer(ref, hyp)
        # 1 substitution ("1-0-1" -> "101") out of 8 words
        self.assertAlmostEqual(wer, 1 / 8, places=2)

    def test_evaluate_ocr_report(self):
        gt = "Dr. Rajesh Sharma M.D. Rx Paracetamol 500 mg"
        pred = "Dr. Rajesh Sharma M.D. Re Pavacelmol 500 mg"
        report = evaluate_ocr(gt, pred)
        self.assertGreater(report.cer, 0.0)
        self.assertGreater(report.wer, 0.0)
        self.assertIn("word_recall", report.analysis)
        self.assertIn("paracetamol", report.analysis["missed_gt_words"])

    def test_engine_structure(self):
        engine = TesseractOCREngine()
        # Should not crash even if Tesseract binary is not yet in PATH
        avail = engine.is_available()
        self.assertIsInstance(avail, bool)


if __name__ == "__main__":
    unittest.main()
