"""
Visualization utilities for inspecting image preprocessing stages.
Generates side-by-side diagnostic figures showing original, illumination correction,
binarization, and overlaying line & word bounding boxes.
"""

from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.pipeline import PreprocessingResult


def draw_segmentation_overlay(
    image: np.ndarray,
    result: PreprocessingResult,
    draw_words: bool = True,
    line_color: tuple = (0, 200, 0),    # Green
    word_color: tuple = (255, 100, 0),  # Blue (in BGR: (0, 100, 255) / in RGB: (255, 100, 0))
    line_thickness: int = 2,
    word_thickness: int = 1,
) -> np.ndarray:
    """
    Draw colored bounding boxes of segmented lines and words on the deskewed image.
    
    Args:
        image: Source RGB/BGR image.
        result: PreprocessingResult containing line and word coordinates.
        draw_words: Whether to draw individual word boxes inside lines.
        
    Returns:
        Image with overlay boxes drawn.
    """
    overlay = image.copy()
    
    for line in result.lines:
        lx, ly, lw, lh = line.bbox
        cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), line_color, line_thickness)
        
        # Label line number
        cv2.putText(
            overlay,
            f"L{line.line_index + 1}",
            (lx + 2, max(15, ly - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            line_color,
            1,
            cv2.LINE_AA,
        )

        if draw_words:
            for word in line.words:
                wx, wy, ww, wh = word.bbox
                cv2.rectangle(overlay, (wx, wy), (wx + ww, wy + wh), word_color, word_thickness)

    return overlay


def plot_preprocessing_stages(
    result: PreprocessingResult,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Generate a 4-panel diagnostic plot:
    1. Original Input Image
    2. Illumination / Shadow Corrected (or Grayscale)
    3. Final Clean Binarized (Sauvola)
    4. Segmented Lines & Words Overlay
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # 1. Original
    orig_rgb = cv2.cvtColor(result.original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title(f"1. Original ({result.original.shape[1]}x{result.original.shape[0]})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Illumination Corrected or Deskewed
    if result.illumination_corrected is not None:
        illum_rgb = cv2.cvtColor(result.illumination_corrected, cv2.COLOR_BGR2RGB)
        axes[1].imshow(illum_rgb)
        axes[1].set_title("2. Illumination & Shadow Normalized", fontsize=12, fontweight="bold")
    else:
        deskewed_rgb = cv2.cvtColor(result.deskewed, cv2.COLOR_BGR2RGB)
        axes[1].imshow(deskewed_rgb)
        axes[1].set_title(f"2. Deskewed (Angle: {result.skew_angle:.2f} deg)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Binarized
    axes[2].imshow(result.binarized, cmap="gray")
    method_name = result.metadata.get("binarization_method", "binarized").title()
    axes[2].set_title(f"3. {method_name} Binarization + Despeckle", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # 4. Segmentation Overlay
    deskewed_rgb = cv2.cvtColor(result.deskewed, cv2.COLOR_BGR2RGB)
    overlay = draw_segmentation_overlay(
        deskewed_rgb,
        result,
        draw_words=True,
        line_color=(34, 139, 34),    # Forest green
        word_color=(30, 144, 255),   # Dodger blue
    )
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"4. Segmented Output ({result.num_lines} Lines, {result.num_words} Words)",
        fontsize=12,
        fontweight="bold",
    )
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualizer] Saved diagnostic plot to: {save_path}")

    if show_plot:
        plt.show()

    return fig
