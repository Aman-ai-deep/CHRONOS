"""
Outlier rejection module using RANSAC.
"""
from typing import Tuple
import numpy as np
import cv2

def estimate_homography(pts1: np.ndarray, pts2: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates homography matrix and filters outliers using RANSAC."""
    # Placeholder
    H = np.eye(3, dtype=np.float32)
    mask = np.zeros((len(pts1), 1), dtype=np.uint8)
    return H, mask
