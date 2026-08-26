"""
Illumination-invariant matching using phase congruency maps and scale pyramids.
"""
from typing import Tuple, List
import numpy as np
import cv2

from src.classical_match import match_sift
from src.outlier_rejection import estimate_homography

def compute_phase_congruency(image: np.ndarray) -> np.ndarray:
    """
    Computes Phase Congruency map of an image using phasepack.
    Phase congruency provides a structural edge-like map that is invariant to
    absolute pixel intensities and illumination changes.
    Falls back to Sobel edge magnitude if phasepack fails.
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    try:
        from phasepack import phasecong
        # Convert image to double precision float in range [0, 1]
        img_double = image.astype(np.float64) / 255.0
        
        # Compute phase congruency
        # phasecong returns 7 values: M, m, ori, ft, PC, EO, T
        # M is the maximum moment of phase congruency covariance (primary edge map)
        M, _, _, _, _, _, _ = phasecong(img_double, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55)
        pc = M
        
        # Clean NaN/Inf values and normalize to [0, 255]
        pc = np.nan_to_num(pc, nan=0.0)
        pc_normalized = np.clip(pc, 0.0, 1.0)
        return (pc_normalized * 255.0).astype(np.uint8)
        
    except Exception as e:
        # Fallback to Sobel edge magnitude if phasepack is missing or fails
        print(f"[ WARNING ] Phasepack failed: {e}. Falling back to Sobel filter.")
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to [0, 255]
        max_val = magnitude.max()
        if max_val > 0:
            magnitude = (magnitude / max_val * 255.0).astype(np.uint8)
        else:
            magnitude = np.zeros_like(image, dtype=np.uint8)
        return magnitude

def match_phase_congruency_multi_scale(img1: np.ndarray, img2: np.ndarray, scales: List[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Finds matches on Phase Congruency maps over a range of scales.
    Estimates the scale ratio by selecting the scale factor that yields the maximum
    number of RANSAC inliers.
    Returns:
        pts1 (np.ndarray): Matches in img1 (scaled back to original dimensions).
        pts2 (np.ndarray): Matches in img2.
        best_scale (float): The estimated scale factor (img1_height / img1_scaled_height).
    """
    if scales is None:
        scales = [0.5, 0.7, 1.0, 1.4, 2.0]
        
    best_pts1 = np.zeros((0, 2), dtype=np.float32)
    best_pts2 = np.zeros((0, 2), dtype=np.float32)
    best_scale = 1.0
    max_inliers = -1
    
    # Compute reference Phase Congruency (does not change scale)
    pc2 = compute_phase_congruency(img2)
    
    for s in scales:
        # Resize source image (img1)
        if s == 1.0:
            img1_scaled = img1.copy()
        else:
            h, w = img1.shape[:2]
            new_w = int(w * s)
            new_h = int(h * s)
            interpolation = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            img1_scaled = cv2.resize(img1, (new_w, new_h), interpolation=interpolation)
            
        pc1_scaled = compute_phase_congruency(img1_scaled)
        
        # Match SIFT on Phase Congruency maps
        pts1_scaled, pts2_matched = match_sift(pc1_scaled, pc2)
        
        if len(pts1_scaled) >= 4:
            _, mask = estimate_homography(pts1_scaled, pts2_matched, threshold=5.0)
            inliers = int(np.sum(mask))
            
            if inliers > max_inliers:
                max_inliers = inliers
                # Scale coordinates of img1 back to original dimensions
                best_pts1 = pts1_scaled / s
                best_pts2 = pts2_matched
                best_scale = s
                
    # Fallback to scale 1.0 SIFT if no scale yielded enough matches
    if max_inliers < 4:
        print("[ INFO ] Multi-scale SIFT on Phase Congruency did not find stable matches. Returning default scale 1.0.")
        pc1 = compute_phase_congruency(img1)
        pts1, pts2 = match_sift(pc1, pc2)
        return pts1, pts2, 1.0
        
    print(f"[ SUCCESS ] Multi-scale Phase Congruency finished. Best scale: {best_scale} (Inliers: {max_inliers})")
    return best_pts1, best_pts2, best_scale
