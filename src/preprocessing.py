"""
Preprocessing module for contrast enhancement, bit-depth stretching, and scale adaptation.
Includes robust NaN/Inf cleaning and nodata handling for raw planetary images.
"""
from typing import Dict, Any, Tuple
import numpy as np
import cv2

def normalize_bit_depth(image: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """
    Stretches the image dynamic range between lower_pct and upper_pct percentiles,
    and scales it to standard 8-bit unsigned integer (uint8) range [0, 255].
    Robust against NaNs, Infs, and outlier pixels/nodata in planetary datasets.
    """
    if image is None or image.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)
        
    # 1. Clean NaNs and Infs
    img_clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    # 2. Mask common planetary nodata fill values (-32768, -9999, 65535, etc.)
    valid_mask = (img_clean != -32768) & (img_clean != -9999) & (img_clean != 65535) & (img_clean > -1e4) & (img_clean < 1e6)
    if np.any(valid_mask):
        valid_pixels = img_clean[valid_mask]
    else:
        valid_pixels = img_clean.ravel()
        
    # 3. Percentile stretching
    p_low = float(np.percentile(valid_pixels, lower_pct))
    p_high = float(np.percentile(valid_pixels, upper_pct))
    
    # Avoid divide-by-zero on uniform/flat images
    if p_high <= p_low:
        p_high = p_low + 1.0
        
    stretched = (img_clean - p_low) / (p_high - p_low) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(stretched)

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to locally 
    normalize contrast. This reduces lighting differences caused by sun elevation/azimuth.
    Input must be uint8.
    """
    if image.dtype != np.uint8:
        image = normalize_bit_depth(image)
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Resizes image by a given scale factor. 
    Uses INTER_AREA interpolation for downsampling (anti-aliasing) and INTER_CUBIC for upsampling.
    """
    if scale_factor == 1.0 or scale_factor <= 0:
        return image
        
    h, w = image.shape[:2]
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

def limit_max_dimension(image: np.ndarray, max_dim: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Resizes image if its max dimension exceeds max_dim.
    Prevents Out-Of-Memory (OOM) crashes on cloud servers (e.g. Streamlit Cloud).
    Returns (resized_image, scale_applied).
    """
    h, w = image.shape[:2]
    current_max = max(h, w)
    
    if current_max <= max_dim or max_dim <= 0:
        return image, 1.0
        
    scale = max_dim / float(current_max)
    resized = resize_image(image, scale)
    return resized, scale

def preprocess_image(image: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
    """
    Unified entry point for preprocessing pipeline:
    1. Normalizes dynamic range to [0, 255] uint8.
    2. Limits max dimension to prevent cloud RAM OOM.
    3. Applies local contrast enhancement (CLAHE).
    4. Resizes image if scale factor is not 1.0.
    """
    if config is None:
        config = {}
        
    enable_clahe = config.get("enable_clahe", True)
    clip_limit = config.get("clahe_clip_limit", 2.0)
    grid_size = config.get("clahe_grid_size", (8, 8))
    scale_factor = config.get("scale_factor", 1.0)
    max_dim = config.get("max_dim", 1024)
    lower_pct = config.get("lower_pct", 1.0)
    upper_pct = config.get("upper_pct", 99.0)
    
    # 1. Normalize dynamic range to uint8
    processed = normalize_bit_depth(image, lower_pct, upper_pct)
    
    # 2. Limit max dimension for RAM safety
    processed, auto_scale = limit_max_dimension(processed, max_dim=max_dim)
    
    # 3. Local contrast enhancement (CLAHE)
    if enable_clahe:
        processed = apply_clahe(processed, clip_limit, grid_size)
        
    # 4. User-requested scaling
    if scale_factor != 1.0:
        processed = resize_image(processed, scale_factor)
        
    return processed
