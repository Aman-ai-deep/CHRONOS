"""
Data loading module for lunar images (PDS formats and standard formats).
Safely handles 16-bit, 32-bit float, 4-channel RGBA, multi-band TIFF, and PDS formats.
"""
import os
from typing import Tuple, Dict, Any
import numpy as np
import cv2

try:
    import pdr
except ImportError:
    pdr = None

try:
    import rasterio
except ImportError:
    rasterio = None

def sanitize_loaded_array(arr: np.ndarray) -> np.ndarray:
    """
    Cleans NaNs, Infs, and reduces 3D/4D arrays to a contiguous 2D grayscale array.
    """
    if arr is None or arr.size == 0:
        return np.zeros((100, 100), dtype=np.float32)
        
    # Replace NaNs/Infs
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Squeeze single dimensions
    arr = np.squeeze(arr)
    
    # Handle multi-channel or multi-band arrays
    if len(arr.shape) == 3:
        # If 3-channel (BGR/RGB) or 4-channel (BGRA/RGBA)
        channels = arr.shape[2] if arr.shape[2] in [3, 4] else arr.shape[0]
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif arr.shape[0] in [3, 4]: # shape (C, H, W)
            arr = arr[0, ...] # Select first band
        else:
            arr = arr[..., 0] # Select first band
            
    return np.ascontiguousarray(arr)

def load_pds_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a PDS image (.img/.lbl) using PlanetaryDataReader (pdr) with fallback."""
    metadata = {"format": "PDS", "path": path}
    
    if pdr is None:
        print(f"[ WARNING ] pdr library not installed. Falling back to standard loader for {path}.")
        return load_standard_image(path)
        
    try:
        dataset = pdr.read(path)
        
        # Read metadata
        if hasattr(dataset, 'metadata') and dataset.metadata is not None:
            metadata.update(dict(dataset.metadata))
        elif hasattr(dataset, 'label') and dataset.label is not None:
            metadata.update(dict(dataset.label))
        
        image_data = None
        if hasattr(dataset, 'keys'):
            keys = list(dataset.keys())
            for key in ['IMAGE', 'image', 'IMAGE_DATA', 'BAND1', 'BAND_1']:
                if key in keys:
                    image_data = dataset[key]
                    break
                    
        if image_data is None and hasattr(dataset, 'values'):
            for val in dataset.values():
                if isinstance(val, np.ndarray):
                    image_data = val
                    break
                    
        if image_data is not None:
            sanitized = sanitize_loaded_array(image_data)
            return sanitized, metadata
            
    except Exception as e:
        print(f"[ WARNING ] pdr read failed for {path}: {e}. Falling back to standard image loader.")
        
    # Fallback to standard reader if pdr fails or missing label
    return load_standard_image(path)

def load_standard_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads standard images (.tif, .png, .jpg, .img) using rasterio or opencv."""
    metadata = {"format": "Standard", "path": path}
    
    # 1. Try rasterio first (geospatial tiff/png)
    if rasterio is not None:
        try:
            with rasterio.open(path) as src:
                image_data = src.read(1)  # Read first band
                metadata.update({
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "transform": list(src.transform) if src.transform else None,
                    "bounds": list(src.bounds) if src.bounds else None,
                    "count": src.count
                })
                sanitized = sanitize_loaded_array(image_data)
                return sanitized, metadata
        except Exception:
            pass

    # 2. OpenCV fallback
    image_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image_data is None:
        # Try reading raw bytes via numpy if opencv returns None
        try:
            arr = np.fromfile(path, dtype=np.uint8)
            image_data = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass
            
    if image_data is None:
        raise FileNotFoundError(f"Could not load image file at {path}. Format may be unreadable or corrupt.")
        
    sanitized = sanitize_loaded_array(image_data)
    metadata.update({
        "width": sanitized.shape[1],
        "height": sanitized.shape[0],
        "dtype": str(sanitized.dtype)
    })
    return sanitized, metadata

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unified entry point to load any lunar image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")
        
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.img', '.lbl']:
        return load_pds_image(path)
    else:
        return load_standard_image(path)
