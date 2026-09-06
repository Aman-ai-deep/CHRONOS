"""
Evaluation metrics calculation module for registration quality (RMSE, Uniformity, Inliers).
Safely handles 0, 1, or N inlier matches without array shape collapse.
"""
from typing import Dict, Any, Tuple
import numpy as np
import cv2
import json

def calculate_rmse(pts1: np.ndarray, pts2: np.ndarray, H: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) of reprojected matches.
    If a RANSAC mask is provided, only computes the error for inliers.
    """
    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0:
        return 0.0
        
    try:
        if mask is not None and len(mask) > 0:
            inlier_mask = mask.reshape(-1) == 1
            inliers1 = pts1[inlier_mask]
            inliers2 = pts2[inlier_mask]
        else:
            inliers1 = pts1
            inliers2 = pts2
            
        if len(inliers1) == 0:
            return 0.0
            
        # Ensure 2D shape (N, 2)
        inliers1 = np.asarray(inliers1).reshape(-1, 2)
        inliers2 = np.asarray(inliers2).reshape(-1, 2)
        
        pts1_reshaped = inliers1.reshape(-1, 1, 2).astype(np.float32)
        pts1_projected = cv2.perspectiveTransform(pts1_reshaped, H).reshape(-1, 2)
        
        errors = np.linalg.norm(pts1_projected - inliers2, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))
        return float(rmse)
    except Exception:
        return 0.0

def calculate_uniformity(pts: np.ndarray, img_shape: Tuple[int, int], grid_size: int = 4) -> Tuple[float, float]:
    """
    Calculates spatial distribution uniformity metrics of point matches.
    Divides the image into a grid of grid_size x grid_size cells.
    Returns:
        occupancy_ratio (float): Fraction of grid cells containing at least one point.
        uniformity_index (float): exp(-CV) where CV is the Coefficient of Variation of cell counts.
    """
    if pts is None or len(pts) == 0:
        return 0.0, 0.0
        
    pts = np.asarray(pts).reshape(-1, 2)
    if len(pts) == 0:
        return 0.0, 0.0
        
    H, W = img_shape[:2]
    if H <= 0 or W <= 0:
        return 0.0, 0.0
        
    num_cells = grid_size * grid_size
    counts = np.zeros(num_cells, dtype=np.int32)
    
    for pt in pts:
        x = float(pt[0])
        y = float(pt[1])
        
        col = int(x / float(W) * grid_size)
        row = int(y / float(H) * grid_size)
        
        # Clamp to grid limits
        col = min(max(col, 0), grid_size - 1)
        row = min(max(row, 0), grid_size - 1)
        
        cell_idx = row * grid_size + col
        counts[cell_idx] += 1
        
    occupied = np.sum(counts > 0)
    occupancy_ratio = float(occupied / num_cells)
    
    mean_val = float(np.mean(counts))
    std_val = float(np.std(counts))
    
    if mean_val > 0:
        cv = std_val / mean_val
        uniformity_index = float(np.exp(-cv))
    else:
        uniformity_index = 0.0
        
    return occupancy_ratio, uniformity_index

def evaluate_registration(pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray, H: np.ndarray, img_shape: Tuple[int, int], grid_size: int = 4) -> Dict[str, Any]:
    """
    Computes all alignment quality metrics safely.
    """
    total_matches = len(pts1) if pts1 is not None else 0
    
    if mask is not None and len(mask) > 0 and total_matches > 0:
        inliers_bool = mask.reshape(-1) == 1
        inlier_pts2 = pts2[inliers_bool]
        inlier_count = int(np.sum(inliers_bool))
    else:
        inlier_pts2 = np.zeros((0, 2), dtype=np.float32)
        inlier_count = 0
        
    inlier_ratio = float(inlier_count / total_matches) if total_matches > 0 else 0.0
    rmse = calculate_rmse(pts1, pts2, H, mask)
    
    # Calculate uniformity on inlier matches mapped to the reference image shape
    occupancy_ratio, uniformity_index = calculate_uniformity(inlier_pts2, img_shape, grid_size)
    
    return {
        "rmse": rmse,
        "inlier_count": inlier_count,
        "total_matches": total_matches,
        "inlier_ratio": inlier_ratio,
        "grid_occupancy": occupancy_ratio,
        "uniformity_index": uniformity_index
    }

def save_metrics_report(metrics: Dict[str, Any], output_path: str):
    """Saves the metrics report as a formatted JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
