"""
view_samples.py

Quick-look tool for Day 3: open a handful of raw dataset images and lay
them out in a grid so you can eyeball them before writing any processing
code. Run this, look at the output image (or the pop-up windows), and
jot down what you notice in data/observations.md.

Usage:
    python view_samples.py                # shows 9 random samples
    python view_samples.py --n 20         # shows 20 samples
    python view_samples.py --save grid.png  # save the grid instead of / as well as showing it
"""

import argparse
import math
import random
from pathlib import Path

from PIL import Image

RAW_DIR = Path(__file__).parent / "data" / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def build_grid(paths, thumb_size=(300, 300)):
    n = len(paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")

    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        img.thumbnail(thumb_size)
        x = (i % cols) * thumb_size[0]
        y = (i // cols) * thumb_size[1]
        grid.paste(img, (x, y))
        print(f"  {p.name}: original size {img.size}")

    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="number of sample images to view")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--save", type=Path, default=None, help="path to save the contact-sheet grid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(
            f"Raw data dir not found: {args.raw_dir}\n"
            f"Did you unzip the Kaggle dataset into data/raw/ ?"
        )

    all_images = find_images(args.raw_dir)
    print(f"Found {len(all_images)} image files under {args.raw_dir}")
    if not all_images:
        raise SystemExit("No images found — check the unzip step / folder structure.")

    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n, len(all_images)))

    print(f"\nSampling {len(sample)} images:")
    grid = build_grid(sample)

    out_path = args.save or (Path(__file__).parent / "data" / "sample_grid.png")
    grid.save(out_path)
    print(f"\nSaved contact sheet to {out_path}")
    print("Open it and look closely — this is the whole point of today.")

    # Also try to pop up an interactive viewer if a display is available
    try:
        grid.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
