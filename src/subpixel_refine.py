"""
Sub-pixel coordinate refinement module using OpenCV corner sub-pixel optimization.
"""
from typing import Tuple
import numpy as np
import cv2

def refine_corners_subpixel(image: np.ndarray, pts: np.ndarray, win_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Refines keypoint coordinates to sub-pixel accuracy using cv2.cornerSubPix.
    Expects grayscale uint8 image and float32 points.
    """
    if len(pts) == 0:
        return pts
        
    # Ensure image is grayscale and uint8
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        # Scale to [0, 255] if float
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    # Reshape points to (N, 1, 2) float32 for cornerSubPix
    pts_f32 = pts.reshape(-1, 1, 2).astype(np.float32)
    
    # Termination criteria: max iterations = 40 or epsilon = 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    
    try:
        # Refines keypoint locations using local image gradients
        refined = cv2.cornerSubPix(image, pts_f32, win_size, (-1, -1), criteria)
        return refined.reshape(-1, 2)
    except Exception as e:
        print(f"[ WARNING ] cornerSubPix refinement failed: {e}. Returning original points.")
        return pts

def refine_matches_subpixel(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, method: str = "corner") -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified entry point to refine match point coordinate pairs to sub-pixel accuracy.
    Refines pts1 in img1 and pts2 in img2 independently.
    Supported methods:
        - "corner": Uses cv2.cornerSubPix (gradient-based local search)
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return pts1, pts2
        
    if method.lower() == "corner":
        pts1_refined = refine_corners_subpixel(img1, pts1)
        pts2_refined = refine_corners_subpixel(img2, pts2)
        return pts1_refined, pts2_refined
    else:
        # Pass-through if unsupported method
        return pts1, pts2
