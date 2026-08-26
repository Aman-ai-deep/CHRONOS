"""
Deep learning-based feature matching module using LoFTR.
"""
from typing import Tuple
import numpy as np

def match_loftr(img1: np.ndarray, img2: np.ndarray, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Detects correspondences using pretrained LoFTR."""
    # Placeholder
    return np.zeros((0, 2)), np.zeros((0, 2))
