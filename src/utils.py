"""
Utility functions for visualization, logging, and false-color composites.
Robustly handles any dtype (uint8, uint16, float32) and shape combinations.
"""
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Ensures an array is a valid 2D uint8 image in range [0, 255]."""
    if img is None or img.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)
        
    if img.dtype == np.uint8:
        out = img
    else:
        # Scale to [0, 255] uint8
        img_float = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        p_low = float(np.percentile(img_float, 1.0)) if img_float.size > 0 else 0.0
        p_high = float(np.percentile(img_float, 99.0)) if img_float.size > 0 else 255.0
        if p_high <= p_low:
            p_high = p_low + 1.0
        out = np.clip((img_float - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
        
    if len(out.shape) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.shape[2] == 3 else out[..., 0]
        
    return np.ascontiguousarray(out)

def plot_matches(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray = None) -> plt.Figure:
    """
    Creates a side-by-side plot of matches, showing lines connecting them.
    Colors inliers in green and outliers in red if a RANSAC mask is provided.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    img1_u8 = ensure_uint8(img1)
    img2_u8 = ensure_uint8(img2)
    
    img1_c = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2BGR)
    img2_c = cv2.cvtColor(img2_u8, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = img1_u8.shape[:2]
    h2, w2 = img2_u8.shape[:2]
    
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    canvas[:h1, :w1] = img1_c
    canvas[:h2, w1:w1+w2] = img2_c
    
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    if pts1 is not None and pts2 is not None and len(pts1) > 0 and len(pts2) > 0:
        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            x1, y1 = p1
            x2, y2 = p2
            x2_shifted = x2 + w1
            
            if mask is not None and i < len(mask):
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
            
    inliers = int(np.sum(mask)) if mask is not None and len(mask) > 0 else (len(pts1) if pts1 is not None else 0)
    total = len(pts1) if pts1 is not None else 0
    ax.set_title(f"Feature Correspondences (RANSAC Inliers: {inliers} / {total})")
    plt.tight_layout()
    return fig

def overlay_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays source warped image on reference image using alpha blending."""
    img1_u8 = ensure_uint8(img1)
    img2_u8 = ensure_uint8(img2)
    
    img1_c = cv2.cvtColor(img1_u8, cv2.COLOR_GRAY2BGR)
    img2_c = cv2.cvtColor(img2_u8, cv2.COLOR_GRAY2BGR)
    
    # Resize if not matching
    if img1_c.shape != img2_c.shape:
        img2_c = cv2.resize(img2_c, (img1_c.shape[1], img1_c.shape[0]))
        
    return cv2.addWeighted(img1_c, alpha, img2_c, 1.0 - alpha, 0.0)

def create_alignment_composite(ref: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """
    Creates a false-color alignment composite to show registration quality.
    Red channel = Warped Source Image
    Green & Blue channels = Reference Image
    """
    ref_u8 = ensure_uint8(ref)
    warped_u8 = ensure_uint8(warped)
    
    if ref_u8.shape != warped_u8.shape:
        warped_u8 = cv2.resize(warped_u8, (ref_u8.shape[1], ref_u8.shape[0]))
        
    composite = np.zeros((ref_u8.shape[0], ref_u8.shape[1], 3), dtype=np.uint8)
    composite[..., 0] = warped_u8  # Red = Warped
    composite[..., 1] = ref_u8     # Green = Reference
    composite[..., 2] = ref_u8     # Blue = Reference
    
    return composite
