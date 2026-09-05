import cv2
import numpy as np

img = cv2.imread('scan.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Couldn't find scan.png — check the filename/path")

# Step 1: Denoise
denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
cv2.imwrite('denoised_output.png', denoised)
print("Step 1 done — saved as denoised_output.png")

# Step 2: Threshold (binarize)
_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: Morphological cleanup
kernel = np.ones((2,2), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('cleaned.png', cleaned)
print("Step 2 & 3 done — saved as cleaned.png")