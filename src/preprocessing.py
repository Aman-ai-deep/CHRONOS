"""
Preprocessing module for contrast enhancement, bit-depth stretching, and scale adaptation.
"""
from typing import Dict, Any, Tuple
import numpy as np
import cv2

def normalize_bit_depth(image: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """
    Stretches the image dynamic range between lower_pct and upper_pct percentiles,
    and scales it to standard 8-bit unsigned integer (uint8) range [0, 255].
    Robust against outlier pixels (hot pixels/shadows) in planetary datasets.
    """
    img_float = image.astype(np.float32)
    
    p_low = np.percentile(img_float, lower_pct)
    p_high = np.percentile(img_float, upper_pct)
    
    # Avoid divide-by-zero on uniform/flat images
    if p_high == p_low:
        p_high = p_low + 1e-5
        
    stretched = (img_float - p_low) / (p_high - p_low) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched

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
    if scale_factor == 1.0:
        return image
        
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

def preprocess_image(image: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
    """
    Unified entry point for preprocessing pipeline:
    1. Normalizes dynamic range to [0, 255] uint8.
    2. Applies local contrast enhancement (CLAHE).
    3. Resizes image if scale factor is not 1.0.
    """
    if config is None:
        config = {}
        
    enable_clahe = config.get("enable_clahe", True)
    clip_limit = config.get("clahe_clip_limit", 2.0)
    grid_size = config.get("clahe_grid_size", (8, 8))
    scale_factor = config.get("scale_factor", 1.0)
    lower_pct = config.get("lower_pct", 1.0)
    upper_pct = config.get("upper_pct", 99.0)
    
    # 1. Normalize dynamic range
    processed = normalize_bit_depth(image, lower_pct, upper_pct)
    
    # 2. Local contrast enhancement (CLAHE)
    if enable_clahe:
        processed = apply_clahe(processed, clip_limit, grid_size)
        
    # 3. Resizing
    if scale_factor != 1.0:
        processed = resize_image(processed, scale_factor)
        
    return processed
