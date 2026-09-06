"""
Main end-to-end registration pipeline orchestrator.
Robustly handles raw planetary formats, high-resolution arrays, and memory limits.
"""
from typing import Dict, Any, Tuple
import numpy as np
import cv2

from src.data_loader import load_image
from src.preprocessing import preprocess_image
from src.classical_match import match_sift, match_orb
from src.outlier_rejection import estimate_homography
from src.illum_invariant import match_phase_congruency_multi_scale
from src.deep_match import match_loftr
from src.subpixel_refine import refine_matches_subpixel
from src.evaluate import evaluate_registration

def calculate_reprojection_rmse(pts1: np.ndarray, pts2: np.ndarray, H: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) of inlier matches reprojected
    through the estimated homography H.
    """
    try:
        if mask is None or len(mask) == 0 or pts1 is None or pts2 is None:
            return 0.0
            
        inlier_mask = mask.reshape(-1) == 1
        inliers1 = pts1[inlier_mask]
        inliers2 = pts2[inlier_mask]
        
        if len(inliers1) == 0:
            return 0.0
            
        inliers1 = np.asarray(inliers1).reshape(-1, 2)
        inliers2 = np.asarray(inliers2).reshape(-1, 2)
        
        pts1_reshaped = inliers1.reshape(-1, 1, 2).astype(np.float32)
        pts1_projected = cv2.perspectiveTransform(pts1_reshaped, H).reshape(-1, 2)
        
        errors = np.linalg.norm(pts1_projected - inliers2, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        return float(rmse)
    except Exception:
        return 0.0

def run_registration(source_path: str, reference_path: str, method: str = "sift", config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrates the entire registration pipeline end-to-end.
    Steps: Load -> Preprocess -> Match -> RANSAC -> Warp.
    Returns:
        warped_image (np.ndarray): The aligned/registered source image.
        metrics (Dict[str, Any]): Registration metrics (RMSE, inlier count, ratio).
        details (Dict[str, Any]): Intermediate arrays for visual debugging.
    """
    if config is None:
        config = {}
        
    # 1. Load Raw Images
    ref_raw, ref_meta = load_image(reference_path)
    src_raw, src_meta = load_image(source_path)
    
    # 2. Preprocess Images (Normalizes bit depth, cleans NaNs, caps max_dim to 1024px for RAM safety)
    ref_prep = preprocess_image(ref_raw, config)
    src_prep = preprocess_image(src_raw, config)
    
    # 3. Feature Detection and Matching
    method_lower = method.lower()
    best_scale = 1.0
    
    if method_lower == "sift":
        pts1, pts2 = match_sift(src_prep, ref_prep)
    elif method_lower == "orb":
        pts1, pts2 = match_orb(src_prep, ref_prep)
    elif method_lower == "phase":
        pts1, pts2, best_scale = match_phase_congruency_multi_scale(src_prep, ref_prep)
    elif method_lower == "loftr":
        max_dim = config.get("max_dim", 1024)
        pts1, pts2 = match_loftr(src_prep, ref_prep, max_inference_dim=max_dim)
    else:
        raise ValueError(f"Unknown registration method: {method}")
        
    # 3.5 Sub-pixel Refinement
    enable_subpixel = config.get("enable_subpixel", True)
    if enable_subpixel and len(pts1) >= 4:
        pts1, pts2 = refine_matches_subpixel(src_prep, ref_prep, pts1, pts2)
        
    # 4. Outlier Rejection via RANSAC Homography
    ransac_thresh = config.get("ransac_threshold", 5.0)
    H, mask = estimate_homography(pts1, pts2, threshold=ransac_thresh)
    
    # 5. Image Warping on clean preprocessed array
    h_ref, w_ref = ref_prep.shape[:2]
    warped_image = cv2.warpPerspective(
        src_prep, H, (w_ref, h_ref), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )
    
    # 6. Compute Metrics
    metrics = evaluate_registration(pts1, pts2, mask, H, ref_prep.shape)
    metrics["estimated_scale"] = best_scale
    
    # Store intermediate products for plotting / analysis
    details = {
        "ref_raw": ref_raw,
        "src_raw": src_raw,
        "ref_preprocessed": ref_prep,
        "src_preprocessed": src_prep,
        "pts1": pts1,
        "pts2": pts2,
        "mask": mask,
        "H": H
    }
    
    return warped_image, metrics, details
