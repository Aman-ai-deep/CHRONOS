"""
Utility functions for visualization, logging, and false-color composites.
"""
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_matches(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray = None) -> plt.Figure:
    """
    Creates a side-by-side plot of matches, showing lines connecting them.
    Colors inliers in green and outliers in red if a RANSAC mask is provided.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure images are in BGR format for plotting
    img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    canvas[:h1, :w1] = img1_c
    canvas[:h2, w1:w1+w2] = img2_c
    
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    if len(pts1) > 0 and len(pts2) > 0:
        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            x1, y1 = p1
            x2, y2 = p2
            x2_shifted = x2 + w1
            
            if mask is not None:
                is_inlier = mask[i][0] == 1 if hasattr(mask[i], '__len__') else mask[i] == 1
                color = 'lime' if is_inlier else 'red'
                alpha = 0.8 if is_inlier else 0.15
                linewidth = 1.0 if is_inlier else 0.5
            else:
                color = 'cyan'
                alpha = 0.6
                linewidth = 1.0
                
            ax.plot([x1, x2_shifted], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
            ax.scatter([x1, x2_shifted], [y1, y2], color=color, s=10, alpha=alpha)
            
    inliers = int(np.sum(mask)) if mask is not None else len(pts1)
    total = len(pts1)
    ax.set_title(f"Feature Correspondences (RANSAC Inliers: {inliers} / {total})")
    plt.tight_layout()
    return fig

def overlay_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays source warped image on reference image using alpha blending."""
    img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
    
    # Resize if not matching
    if img1_c.shape != img2_c.shape:
        img2_c = cv2.resize(img2_c, (img1_c.shape[1], img1_c.shape[0]))
        
    return cv2.addWeighted(img1_c, alpha, img2_c, 1.0 - alpha, 0.0)

def create_alignment_composite(ref: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """
    Creates a false-color alignment composite to show registration quality.
    Red channel = Warped Source Image
    Green & Blue channels = Reference Image
    
    In areas of perfect alignment, the image looks grayscale (black & white).
    In areas of misalignments/shadow differences, bright red or cyan fringes are visible.
    """
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if len(ref.shape) == 3 else ref.copy()
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
    
    if ref_gray.shape != warped_gray.shape:
        warped_gray = cv2.resize(warped_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
    composite = np.zeros((ref_gray.shape[0], ref_gray.shape[1], 3), dtype=np.uint8)
    composite[..., 0] = warped_gray  # Red = Warped
    composite[..., 1] = ref_gray     # Green = Reference
    composite[..., 2] = ref_gray     # Blue = Reference
    
    return composite
