"""
Outlier rejection module using RANSAC homography estimation.
"""
from typing import Tuple
import numpy as np
import cv2

def estimate_homography(pts1: np.ndarray, pts2: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates homography matrix mapping pts1 to pts2 and filters outlier matches using RANSAC.
    Requires at least 4 matches.
    Returns:
        H (np.ndarray): 3x3 homography transformation matrix. If estimation fails, returns 3x3 Identity.
        mask (np.ndarray): Nx1 binary array (values 0 or 1), where 1 indicates an inlier match.
    """
    # Homography requires at least 4 point matches
    if len(pts1) < 4:
        return np.eye(3, dtype=np.float64), np.zeros((len(pts1), 1), dtype=np.uint8)
        
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    
    if H is None:
        H = np.eye(3, dtype=np.float64)
        mask = np.zeros((len(pts1), 1), dtype=np.uint8)
        
    return H, mask
