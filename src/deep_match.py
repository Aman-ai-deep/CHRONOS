"""
Deep learning matcher using local feature transformers (LoFTR) via Kornia.
"""
from typing import Tuple
import numpy as np
import cv2
import torch
import kornia.feature as KF

def match_loftr(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes dense correspondences between img1 and img2 using Kornia's LoFTR.
    Inputs:
        img1 (np.ndarray): Grayscale source (moving) image.
        img2 (np.ndarray): Grayscale reference (fixed) image.
    Returns:
        pts1 (np.ndarray): Nx2 array of matching coordinates in img1.
        pts2 (np.ndarray): Nx2 array of matching coordinates in img2.
    """
    # Ensure images are single channel grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    # Select execution device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ INFO ] Running LoFTR inference on device: {device}")
    
    # 1. Resize images to multiples of 16 (strict constraint of LoFTR encoder layers)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    new_h1 = ((h1 + 15) // 16) * 16
    new_w1 = ((w1 + 15) // 16) * 16
    new_h2 = ((h2 + 15) // 16) * 16
    new_w2 = ((w2 + 15) // 16) * 16
    
    # Apply resize
    img1_resized = cv2.resize(img1, (new_w1, new_h1), interpolation=cv2.INTER_AREA) if (new_h1 != h1 or new_w1 != w1) else img1
    img2_resized = cv2.resize(img2, (new_w2, new_h2), interpolation=cv2.INTER_AREA) if (new_h2 != h2 or new_w2 != w2) else img2
    
    # Scale factors to map coordinates back to original size
    scale_x1 = w1 / new_w1
    scale_y1 = h1 / new_h1
    scale_x2 = w2 / new_w2
    scale_y2 = h2 / new_h2
    
    # 2. Convert to PyTorch tensors and normalize to range [0.0, 1.0]
    # Input shape must be (1, 1, H, W)
    t1 = torch.from_numpy(img1_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    t2 = torch.from_numpy(img2_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 3. Load pretrained LoFTR model
    try:
        # Load LoFTR with outdoor weights
        matcher = KF.LoFTR(pretrained='outdoor').to(device)
        matcher.eval()
        
        input_dict = {"image0": t1, "image1": t2}
        
        # 4. Inference
        with torch.no_grad():
            correspondences = matcher(input_dict)
            
        # 5. Extract points as numpy arrays
        pts0 = correspondences['keypoints0'].cpu().numpy()
        pts1 = correspondences['keypoints1'].cpu().numpy()
        
        if len(pts0) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
            
        # 6. Map matches back to original image dimensions
        pts0_orig = pts0 * np.array([scale_x1, scale_y1])
        pts1_orig = pts1 * np.array([scale_x2, scale_y2])
        
        print(f"[ SUCCESS ] LoFTR matched {len(pts0_orig)} keypoint pairs.")
        return pts0_orig, pts1_orig
        
    except Exception as e:
        print(f"[ ERROR ] LoFTR inference failed: {e}. Returning empty matches.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
