import cv2  # OpenCV library for img load/process
import os   # std library for interacting w files


def load_and_grayscale(image_path):
    """Load an image and convert it to grayscale. Returns the grayscale array (not saved yet)."""
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def binarize(img, method="otsu"):
    """
    img: grayscale image (numpy array)
    method: 'otsu' or 'adaptive'
    returns: binarized (black/white) image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        out = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=10
        )

    else:
        raise ValueError("method must be 'otsu' or 'adaptive'")

    return out


def process_sample_images(raw_dir="data/raw", processed_dir="data/processed", limit=4):
    """Runs grayscale + both binarization methods on a handful of sample images."""
    otsu_dir = os.path.join(processed_dir, "otsu")
    adaptive_dir = os.path.join(processed_dir, "adaptive")
    os.makedirs(otsu_dir, exist_ok=True)
    os.makedirs(adaptive_dir, exist_ok=True)

    sample_files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:limit]

    for fname in sample_files:
        path = os.path.join(raw_dir, fname)
        gray = load_and_grayscale(path)

        otsu_result = binarize(gray, method="otsu")
        adaptive_result = binarize(gray, method="adaptive")

        cv2.imwrite(os.path.join(otsu_dir, fname), otsu_result)
        cv2.imwrite(os.path.join(adaptive_dir, fname), adaptive_result)

        print(f"Processed {fname}: saved otsu + adaptive versions")


if __name__ == "__main__":
    process_sample_images()
