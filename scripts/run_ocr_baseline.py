"""
Week 1-2 Baseline OCR Benchmark Runner.
Executes Tesseract OCR on full images and segmented line crops,
computes CER/WER metrics against ground truth, and highlights handwriting failure modes.
"""

import argparse
import json
import os
import sys
import time
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.ocr.tesseract_engine import TesseractOCREngine, find_tesseract_binary
from src.ocr.evaluator import evaluate_ocr


SYNTHETIC_GROUND_TRUTH = """
METRO CARE HOSPITAL & CLINIC
Dr. Rajesh Sharma, M.D. (Internal Medicine)
Reg. No: MCI-2015-88392 | Phone: +91 98765 43210
Patient: Rajesh Kumar, 38 M Date: 12-10-2026
Diagnosis: Acute Upper Respiratory Tract Infection
Rx
1. Tab. Paracetmol 500 mg 1-0-1 x 5 days (after food)
2. Cap. Amoxicillin 250 mg 1-1-1 x 7 days TDS
3. Syp. Azithromycin 200 mg 0-1-0 x 3 days OD
4. Tab. Pantoprazole 40 mg 1-0-0 x 5 days (before breakfast)
5. Tab. Cetirizine 10 mg 0-0-1 x 5 days HS (at bedtime)
Advice:
- Warm saline gargle twice daily
- Review after 5 days if fever persists
Dr. R. Sharma
Authorized Signatory
""".strip()


def run_benchmark_on_image(
    image_path: str,
    pipeline: PreprocessingPipeline,
    engine: TesseractOCREngine,
    ground_truth: str = None,
    output_dir: str = "data/processed",
):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*75}")
    print(f" OCR Benchmark: {basename}")
    print(f"{'='*75}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 1. Run Preprocessing Pipeline
    is_photo = "photo" in basename.lower()
    mode = "photo" if is_photo else "scan"
    print(f" Running Stage 1 Preprocessing (Mode: {mode.upper()})...")
    prep_result = pipeline.process(image, mode=mode)
    print(f" Extracted {prep_result.num_lines} line segments in {prep_result.metadata['timings_sec']['total_pipeline_time']}s")

    # 2. Run Tesseract Baseline OCR
    t0 = time.time()
    print(" Running Tesseract OCR (Full Page vs Line Segments)...")
    ocr_result = engine.process_prescription(prep_result)
    ocr_time = round(time.time() - t0, 4)

    print(f"\n--- [Tesseract Page-Level OCR (PSM 6)] ---")
    page_preview = ocr_result.full_page_text[:300] + ("..." if len(ocr_result.full_page_text) > 300 else "")
    print(page_preview if page_preview.strip() else "[NO TEXT DETECTED ON FULL PAGE]")

    print(f"\n--- [Tesseract Line-Segment OCR (PSM 7)] ---")
    line_preview = ocr_result.line_segmented_text[:300] + ("..." if len(ocr_result.line_segmented_text) > 300 else "")
    print(line_preview if line_preview.strip() else "[NO TEXT DETECTED ON LINES]")
    print(f" Average Confidence: {ocr_result.average_confidence:.1f}%")

    # 3. Evaluation if ground truth is available
    eval_report = None
    if ground_truth:
        eval_report = evaluate_ocr(ground_truth, ocr_result.line_segmented_text)
        print(f"\n--- [Accuracy Evaluation vs Ground Truth] ---")
        print(f" Character Error Rate (CER) : {eval_report.cer * 100:.1f}%")
        print(f" Word Error Rate (WER)      : {eval_report.wer * 100:.1f}%")
        print(f" Word Recall                : {eval_report.analysis['word_recall'] * 100:.1f}%")
        print(f" Missed Words Sample        : {eval_report.analysis['missed_gt_words'][:5]}")

    # 4. Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    out_json = {
        "image": image_path,
        "engine": "Tesseract",
        "version": engine.get_version(),
        "runtime_sec": ocr_time,
        "average_confidence": ocr_result.average_confidence,
        "full_page_text": ocr_result.full_page_text,
        "line_segmented_text": ocr_result.line_segmented_text,
        "lines": [
            {"line_index": l.line_index, "text": l.text, "confidence": l.confidence}
            for l in ocr_result.lines
        ],
    }
    if eval_report:
        out_json["metrics"] = {
            "cer": eval_report.cer,
            "wer": eval_report.wer,
            "word_recall": eval_report.analysis["word_recall"],
            "missed_words": eval_report.analysis["missed_gt_words"],
        }

    json_path = os.path.join(output_dir, f"{basename}_ocr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n Saved OCR benchmark report to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Week 1-2 Tesseract OCR Benchmark")
    parser.add_argument("--input", type=str, default=None, help="Image path or None for defaults")
    args = parser.parse_args()

    engine = TesseractOCREngine()
    if not engine.is_available():
        print("Tesseract binary not found! Please run python scripts/install_tesseract.py or install Tesseract OCR.")
        return

    print(f"Tesseract OCR Engine Initialized (Version: {engine.get_version()})")
    pipeline = PreprocessingPipeline(mode="photo", binarization_method="sauvola")

    if args.input:
        gt = SYNTHETIC_GROUND_TRUTH if "prescription_" in args.input else None
        run_benchmark_on_image(args.input, pipeline, engine, ground_truth=gt)
    else:
        # Default test cases
        test_images = [
            ("data/raw/prescription_scan.png", SYNTHETIC_GROUND_TRUTH),
            ("data/raw/prescription_photo.png", SYNTHETIC_GROUND_TRUTH),
            ("data/raw/sample_real.jpg", None),
        ]
        for path, gt in test_images:
            if os.path.exists(path):
                run_benchmark_on_image(path, pipeline, engine, ground_truth=gt)


if __name__ == "__main__":
    main()
