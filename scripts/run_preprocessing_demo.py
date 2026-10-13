"""
Demo script to execute and benchmark the Week 1 OpenCV Preprocessing Pipeline.
Processes prescription images (scans and mobile photos), saves intermediate visual stages,
and exports extracted line crops to data/segments/.
"""

import argparse
import os
import sys
from glob import glob
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.visualizer import plot_preprocessing_stages


def process_file(
    image_path: str,
    pipeline: PreprocessingPipeline,
    output_dir: str = "data/processed",
    segments_dir: str = "data/segments",
    save_crops: bool = True,
):
    print(f"\n{'='*70}")
    print(f" Processing: {os.path.basename(image_path)}")
    print(f"{'='*70}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    basename = os.path.splitext(os.path.basename(image_path))[0]
    is_photo = "photo" in basename.lower()
    mode = "photo" if is_photo else "scan"

    print(f" Image Dimensions : {image.shape[1]}x{image.shape[0]} (Channels: {image.shape[2]})")
    print(f" Selected Mode    : {mode.upper()}")

    result = pipeline.process(image, mode=mode)

    print(f"\n[Results]")
    print(f" Detected Skew Angle : {result.skew_angle:+.2f} degrees")
    print(f" Lines Detected      : {result.num_lines}")
    print(f" Words/Tokens Found  : {result.num_words}")
    print(f" Pipeline Timings    :")
    for step, duration in result.metadata["timings_sec"].items():
        print(f"   - {step:<25}: {duration:.4f} s")

    # 1. Save Diagnostic Stage Comparison
    os.makedirs(output_dir, exist_ok=True)
    diag_path = os.path.join(output_dir, f"{basename}_stages.png")
    plot_preprocessing_stages(result, save_path=diag_path, show_plot=False)

    # 2. Save Clean Processed Binarized & Deskewed Images
    bin_path = os.path.join(output_dir, f"{basename}_binarized.png")
    deskew_path = os.path.join(output_dir, f"{basename}_deskewed.png")
    cv2.imwrite(bin_path, result.binarized)
    cv2.imwrite(deskew_path, result.deskewed)

    # 3. Export Line Crops
    if save_crops and result.lines:
        item_segments_dir = os.path.join(segments_dir, basename)
        os.makedirs(item_segments_dir, exist_ok=True)
        for idx, line in enumerate(result.lines, 1):
            line_filename = os.path.join(item_segments_dir, f"line_{idx:02d}.png")
            cv2.imwrite(line_filename, line.image)
        print(f" Exported {len(result.lines)} line crops to: {item_segments_dir}")


def main():
    parser = argparse.ArgumentParser(description="Medical Prescription Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to single image or dir")
    parser.add_argument("--binarization", type=str, default="sauvola", choices=["sauvola", "adaptive_gaussian", "otsu"])
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "photo", "scan"])
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        mode=args.mode,
        binarization_method=args.binarization,
    )

    if args.input:
        if os.path.isdir(args.input):
            files = sorted(glob(os.path.join(args.input, "*.*")))
        else:
            files = [args.input]
    else:
        files = sorted(glob("data/raw/*.*"))

    if not files:
        print("No raw images found in data/raw/. Generating synthetic samples first...")
        from scripts.generate_synthetic_prescription import generate_samples
        generate_samples()
        files = sorted(glob("data/raw/*.*"))

    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            process_file(f, pipeline)

    print(f"\n{'='*70}")
    print(" Preprocessing Complete! Inspect outputs in data/processed/ and data/segments/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
