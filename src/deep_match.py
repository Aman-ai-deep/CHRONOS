"""
Deep learning matcher using local feature transformers (LoFTR) via Kornia.
Includes automatic memory safety caps to prevent OOM server crashes on large raw images.
"""
from typing import Tuple
import numpy as np
import cv2
import torch

try:
    import kornia.feature as KF
except ImportError:
    KF = None

def match_loftr(img1: np.ndarray, img2: np.ndarray, max_inference_dim: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes dense correspondences between img1 and img2 using Kornia's LoFTR.
    Inputs:
        img1 (np.ndarray): Grayscale source (moving) image.
        img2 (np.ndarray): Grayscale reference (fixed) image.
        max_inference_dim (int): Maximum spatial dimension for LoFTR tensor evaluation.
                                Prevents Out-Of-Memory (OOM) crashes on Cloud servers.
    Returns:
        pts1 (np.ndarray): Nx2 array of matching coordinates in original img1 space.
        pts2 (np.ndarray): Nx2 array of matching coordinates in original img2 space.
    """
    # Ensure images are single channel grayscale uint8
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    h1_orig, w1_orig = img1.shape[:2]
    h2_orig, w2_orig = img2.shape[:2]
    
    # 1. Cap maximum resolution for LoFTR to prevent OOM
    scale1 = 1.0
    scale2 = 1.0
    
    if max(h1_orig, w1_orig) > max_inference_dim and max_inference_dim > 0:
        scale1 = max_inference_dim / float(max(h1_orig, w1_orig))
        img1 = cv2.resize(img1, (int(w1_orig * scale1), int(h1_orig * scale1)), interpolation=cv2.INTER_AREA)
        
    if max(h2_orig, w2_orig) > max_inference_dim and max_inference_dim > 0:
        scale2 = max_inference_dim / float(max(h2_orig, w2_orig))
        img2 = cv2.resize(img2, (int(w2_orig * scale2), int(h2_orig * scale2)), interpolation=cv2.INTER_AREA)
        
    # Select execution device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Resize images to multiples of 16 (strict constraint of LoFTR encoder layers)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    new_h1 = max(16, ((h1 + 15) // 16) * 16)
    new_w1 = max(16, ((w1 + 15) // 16) * 16)
    new_h2 = max(16, ((h2 + 15) // 16) * 16)
    new_w2 = max(16, ((w2 + 15) // 16) * 16)
    
    # Apply resize
    img1_resized = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA) if (new_h1 != h1 or new_w1 != w1) else img1
    img2_resized = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA) if (new_h2 != h2 or new_w2 != w2) else img2
    
    # Scale factors to map coordinates back to ORIGINAL input dimensions
    map_x1 = w1_orig / float(new_w1)
    map_y1 = h1_orig / float(new_h1)
    map_x2 = w2_orig / float(new_w2)
    map_y2 = h2_orig / float(new_h2)
    
    # 3. Convert to PyTorch tensors and normalize to range [0.0, 1.0]
    t1 = torch.from_numpy(img1_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    t2 = torch.from_numpy(img2_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 4. Load pretrained LoFTR model with fallback
    try:
        if KF is None:
            raise ImportError("kornia package is not installed")
            
        matcher = KF.LoFTR(pretrained='outdoor').to(device)
        matcher.eval()
        
        input_dict = {"image0": t1, "image1": t2}
        
        with torch.no_grad():
            correspondences = matcher(input_dict)
            
        pts0 = correspondences['keypoints0'].cpu().numpy()
        pts1 = correspondences['keypoints1'].cpu().numpy()
        
        if len(pts0) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
            
        # Map matches back to original image dimensions
        pts0_orig = pts0 * np.array([map_x1, map_y1])
        pts1_orig = pts1 * np.array([map_x2, map_y2])
        
        return pts0_orig, pts1_orig
        
    except Exception as e:
        try:
            from src.classical_match import match_sift
        except ImportError:
            from classical_match import match_sift
        return match_sift(img1, img2)
