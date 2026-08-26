"""
Utility functions for visualization, logging, and configuration.
"""
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_matches(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray = None) -> plt.Figure:
    """Creates a visualization of matches side by side with lines linking them."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.hstack([img1, img2]), cmap='gray')
    ax.set_title("Matches Visualization")
    return fig

def overlay_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays source warped image on reference image."""
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
