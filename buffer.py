img = cv2.imread('scan.jpg', cv2.IMREAD_GRAYSCALE)
denoised = cv2.fastNlMeansDenoising(img, h=10)
_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((2,2), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('cleaned.jpg', cleaned)