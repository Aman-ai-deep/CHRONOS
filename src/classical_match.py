"""
Classical matching module using SIFT/ORB with FLANN/BFMatcher.
"""
from typing import Tuple, List
import numpy as np
import cv2

def match_sift(img1: np.ndarray, img2: np.ndarray, ratio_threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """Detects and matches SIFT features between two images."""
    # Placeholder
    return np.zeros((0, 2)), np.zeros((0, 2))

def match_orb(img1: np.ndarray, img2: np.ndarray, max_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Detects and matches ORB features between two images."""
    # Placeholder
    return np.zeros((0, 2)), np.zeros((0, 2))
