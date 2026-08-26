"""
Data loading module for lunar images (PDS formats and standard formats).
"""
import os
from typing import Tuple, Dict, Any
import numpy as np

def load_pds_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a PDS image (.img/.lbl) using PlanetaryDataReader (pdr)."""
    try:
        import pdr
        dataset = pdr.read(path)
        
        # Attempt to read metadata from the label
        metadata = {}
        if hasattr(dataset, 'metadata') and dataset.metadata is not None:
            metadata = dict(dataset.metadata)
        elif hasattr(dataset, 'label') and dataset.label is not None:
            metadata = dict(dataset.label)
        
        # Check for image data in common dictionary keys
        image_data = None
        if hasattr(dataset, 'keys'):
            keys = list(dataset.keys())
            for key in ['IMAGE', 'image', 'IMAGE_DATA', 'BAND1', 'BAND_1']:
                if key in keys:
                    image_data = dataset[key]
                    break
                    
        # Check attributes if dictionary lookup failed
        if image_data is None:
            for attr in ['IMAGE', 'image', 'IMAGE_DATA', 'data']:
                if hasattr(dataset, attr):
                    val = getattr(dataset, attr)
                    if isinstance(val, np.ndarray):
                        image_data = val
                        break
                        
        # Last resort: search values for numpy arrays
        if image_data is None and hasattr(dataset, 'values'):
            for val in dataset.values():
                if isinstance(val, np.ndarray):
                    image_data = val
                    break
                    
        if image_data is None:
            raise ValueError(f"Could not find image array in PDS dataset loaded from {path}")
            
        # Ensure it is a 2D grayscale array
        if len(image_data.shape) > 2:
            # If multi-band (e.g., hyperspectral or color), select the first band
            image_data = image_data[..., 0]
            
        metadata["format"] = "PDS"
        metadata["path"] = path
        return image_data, metadata
    except Exception as e:
        raise RuntimeError(f"Error reading PDS file {path} via pdr: {e}")

def load_standard_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads standard images (.tif, .png, .jpg) using rasterio or opencv."""
    metadata = {"format": "Standard", "path": path}
    
    # Try rasterio first (geospatial tiff/png)
    try:
        import rasterio
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
            return image_data, metadata
    except ImportError:
        pass
    except Exception:
        # Fall back to opencv if rasterio fails
        pass

    # OpenCV fallback
    import cv2
    image_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image_data is None:
        raise FileNotFoundError(f"Could not load image at {path} via OpenCV or Rasterio.")
        
    if len(image_data.shape) == 3:
        # RGB/BGR to Gray
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        
    metadata.update({
        "width": image_data.shape[1],
        "height": image_data.shape[0],
        "dtype": str(image_data.dtype)
    })
    return image_data, metadata

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unified entry point to load any lunar image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")
        
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.img', '.lbl']:
        return load_pds_image(path)
    else:
        return load_standard_image(path)
