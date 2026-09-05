import cv2
import os
import matplotlib.pyplot as plt

raw_dir = "data/raw"
otsu_dir = "data/processed/otsu"
adaptive_dir = "data/processed/adaptive"

sample_files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))][:4]

fig, axes = plt.subplots(len(sample_files), 3, figsize=(10, 4*len(sample_files)))

for i, fname in enumerate(sample_files):
    orig = cv2.imread(os.path.join(raw_dir, fname), cv2.IMREAD_GRAYSCALE)
    otsu = cv2.imread(os.path.join(otsu_dir, fname), cv2.IMREAD_GRAYSCALE)
    adap = cv2.imread(os.path.join(adaptive_dir, fname), cv2.IMREAD_GRAYSCALE)

    axes[i][0].imshow(orig, cmap="gray"); axes[i][0].set_title("original")
    axes[i][1].imshow(otsu, cmap="gray"); axes[i][1].set_title("otsu")
    axes[i][2].imshow(adap, cmap="gray"); axes[i][2].set_title("adaptive")
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("comparison_grid.png")
print("Saved comparison_grid.png")
