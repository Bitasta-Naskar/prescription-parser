"""
Synthetic Prescription Generator.
Generates realistic medical prescription test images using system handwriting fonts
(Ink Free, Segoe Print) for doctor notes and standard sans-serif for clinic headers.
Produces both clean 'scan' versions and challenging 'photo' versions with lighting
gradients, shadows, skew, and paper noise.
"""

import os
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(font_name: str, size: int) -> ImageFont.ImageFont:
    """Load font from Windows font directory with fallback to default."""
    windows_font_path = os.path.join(r"C:\Windows\Fonts", font_name)
    if os.path.exists(windows_font_path):
        try:
            return ImageFont.truetype(windows_font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def create_base_prescription(width: int = 1000, height: int = 1350) -> Image.Image:
    """
    Render a prescription layout: printed clinic header + handwritten doctor Rx notes.
    """
    # Slightly off-white medical pad paper color
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_header_title = get_font("arialbd.ttf", 28)
    font_header_sub = get_font("arial.ttf", 15)
    font_rx_symbol = get_font("segoeprb.ttf", 46)
    font_handwriting = get_font("Inkfree.ttf", 26)
    font_handwriting_alt = get_font("segoepr.ttf", 24)

    # 1. Clinic Header (Printed)
    header_color = (25, 45, 85)
    draw.text((60, 45), "METRO CARE HOSPITAL & CLINIC", fill=header_color, font=font_header_title)
    draw.text((60, 85), "Dr. Rajesh Sharma, M.D. (Internal Medicine)", fill=(60, 60, 60), font=font_header_sub)
    draw.text((60, 110), "Reg. No: MCI-2015-88392 | Phone: +91 98765 43210", fill=(100, 100, 100), font=font_header_sub)
    
    # Header divider rule
    draw.line([(50, 145), (width - 50, 145)], fill=(180, 190, 205), width=2)

    # Patient info
    draw.text((60, 160), "Patient: Rajesh Kumar, 38 M", fill=(50, 50, 50), font=font_header_sub)
    draw.text((width - 250, 160), "Date: 12-10-2026", fill=(50, 50, 50), font=font_header_sub)
    draw.text((60, 190), "Diagnosis: Acute Upper Respiratory Tract Infection", fill=(70, 70, 70), font=font_header_sub)
    draw.line([(50, 220), (width - 50, 220)], fill=(210, 215, 225), width=1)

    # 2. Rx Symbol (Doctor's handwriting)
    rx_ink = (20, 25, 65)  # Dark blue ink ballpoint pen
    draw.text((60, 235), "Rx", fill=rx_ink, font=font_rx_symbol)

    # 3. Handwritten Medicine Lines
    prescriptions = [
        ("Tab. Paracetmol 500 mg", "1-0-1", "x 5 days", "(after food)"),
        ("Cap. Amoxicillin 250 mg", "1-1-1", "x 7 days", "TDS"),
        ("Syp. Azithromycin 200 mg", "0-1-0", "x 3 days", "OD"),
        ("Tab. Pantoprazole 40 mg", "1-0-0", "x 5 days", "(before breakfast)"),
        ("Tab. Cetirizine 10 mg", "0-0-1", "x 5 days", "HS (at bedtime)"),
    ]

    curr_y = 320
    for idx, (drug, freq, duration, instructions) in enumerate(prescriptions, 1):
        line_text = f"{idx}. {drug}   {freq}   {duration}   {instructions}"
        # Small random vertical jitter to simulate human handwriting fluctuation
        y_offset = (idx * 7) % 5 - 2
        draw.text((80, curr_y + y_offset), line_text, fill=rx_ink, font=font_handwriting)
        curr_y += 75

    # Additional doctor advice
    curr_y += 30
    draw.text((80, curr_y), "Advice:", fill=rx_ink, font=font_handwriting_alt)
    curr_y += 45
    draw.text((100, curr_y), "- Warm saline gargle twice daily", fill=rx_ink, font=font_handwriting)
    curr_y += 45
    draw.text((100, curr_y), "- Review after 5 days if fever persists", fill=rx_ink, font=font_handwriting)

    # Doctor signature
    draw.text((width - 280, height - 160), "Dr. R. Sharma", fill=rx_ink, font=font_handwriting)
    draw.line([(width - 300, height - 110), (width - 80, height - 110)], fill=(80, 80, 80), width=1)
    draw.text((width - 250, height - 100), "Authorized Signatory", fill=(120, 120, 120), font=font_header_sub)

    return img


def add_photo_artifacts(
    pil_img: Image.Image,
    skew_angle: float = 4.5,
    add_shadow: bool = True,
    add_noise: bool = True,
) -> np.ndarray:
    """
    Simulate real-world mobile phone photo capture artifacts:
    - Tilting / camera skew
    - Uneven shadow from overhead ceiling light
    - Vignette (darker corners)
    - Paper texture grain / sensor noise
    """
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # 1. Skew / Tilt rotation
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 0.98)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(220, 220, 220))

    result = rotated.astype(np.float32)

    # 2. Uneven Lighting & Shadow Gradient (diagonal soft shadow across bottom-right)
    if add_shadow:
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        # Diagonal gradient from top-left (light) to bottom-right (darker)
        diag = (x_coords / w + y_coords / h) / 2.0
        shadow_map = 1.0 - 0.35 * diag

        # Localized curved shadow (e.g. hand or phone shadow on top left / middle)
        dist_from_spot = np.sqrt(((x_coords - w * 0.7) / w) ** 2 + ((y_coords - h * 0.8) / h) ** 2)
        lamp_shadow = np.clip(1.0 - 0.25 * np.exp(-dist_from_spot * 3), 0.65, 1.0)

        total_light = shadow_map * lamp_shadow
        for c in range(3):
            result[:, :, c] = result[:, :, c] * total_light

    # 3. Add subtle sensor noise
    if add_noise:
        noise = np.random.normal(0, 3.5, (h, w, 3))
        result = np.clip(result + noise, 0, 255)

    return result.astype(np.uint8)


def generate_samples(output_dir: str = "data/raw"):
    """Generate both scan and photo test prescriptions."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/synthetic", exist_ok=True)

    base = create_base_prescription()

    # 1. Clean scan
    scan_bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
    scan_path = os.path.join(output_dir, "prescription_scan.png")
    cv2.imwrite(scan_path, scan_bgr)
    cv2.imwrite(os.path.join("data/synthetic", "prescription_scan.png"), scan_bgr)
    print(f"[Generated] Clean scan prescription: {scan_path}")

    # 2. Mobile camera photo (skewed, shadowed)
    photo_bgr = add_photo_artifacts(base, skew_angle=5.0, add_shadow=True, add_noise=True)
    photo_path = os.path.join(output_dir, "prescription_photo.png")
    cv2.imwrite(photo_path, photo_bgr)
    cv2.imwrite(os.path.join("data/synthetic", "prescription_photo.png"), photo_bgr)
    print(f"[Generated] Mobile photo prescription with shadows & skew: {photo_path}")


if __name__ == "__main__":
    generate_samples()
