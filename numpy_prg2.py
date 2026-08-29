import numpy as np
import cv2 # Often used for displaying images
import matplotlib.pyplot as plt
'''
# Create a dummy grayscale image (e.g., 100 rows, 200 columns)
# Initially, all pixels are 0 (black)
img = np.zeros((100, 200), dtype=np.uint8)

# Set the first row to white
img[0, :] = 255

# You could then display or save the image
# For example, using OpenCV:
# cv2.imshow("Modified Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("Image array after modification (showing first few rows):")
print(img[:5, :]) # Print the first 5 rows to see the change

plt.imshow(img, cmap='gray')
plt.show()
'''

color = np.zeros((5,5,3), dtype=np.uint8)   # tells numpy/matplotlib "these are 0-255 integers"
color[2,1,1] = 150   # what does this do? guess before running
plt.imshow(color, cmap='gray')
plt.show()
