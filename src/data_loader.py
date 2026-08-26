"""
Data loading module for lunar images (PDS format and standard formats).
"""
import os
from typing import Tuple, Dict, Any
import numpy as np

def load_pds_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a PDS image (.img/.lbl) using PlanetaryDataReader (pdr)."""
    # Placeholder implementation
    metadata = {"format": "PDS", "path": path}
    return np.zeros((512, 512), dtype=np.float32), metadata

def load_standard_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads standard images (.tif, .png, .jpg) using rasterio or opencv."""
    # Placeholder implementation
    metadata = {"format": "Standard", "path": path}
    return np.zeros((512, 512), dtype=np.uint8), metadata

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unified entry point to load any lunar image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.img', '.lbl']:
        return load_pds_image(path)
    else:
        return load_standard_image(path)
