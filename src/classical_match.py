"""
Classical matching module using SIFT/ORB with Lowe's ratio test and cross-check.
"""
from typing import Tuple
import numpy as np
import cv2

def match_sift(img1: np.ndarray, img2: np.ndarray, ratio_threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects and matches SIFT features between two images.
    Applies Lowe's ratio test to filter outlier matches.
    Returns:
        pts1 (np.ndarray): Nx2 array of matching coordinates in img1.
        pts2 (np.ndarray): Nx2 array of matching coordinates in img2.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # If no keypoints are detected or descriptors computed, return empty arrays
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        
    # Match using Brute Force Matcher with L2 norm (Standard for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # k=2 matches for ratio test
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                
    if len(good_matches) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2

def match_orb(img1: np.ndarray, img2: np.ndarray, max_features: int = 1000, ratio_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects and matches ORB features between two images.
    Uses NORM_HAMMING distance and Lowe's ratio test (or cross-check fallback).
    Returns:
        pts1 (np.ndarray): Nx2 array of matching coordinates in img1.
        pts2 (np.ndarray): Nx2 array of matching coordinates in img2.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        
    # Match using Brute Force Matcher with Hamming distance (Standard for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                
    # Fallback to simple cross-check matching if ratio test is too strict for ORB
    if len(good_matches) < 4:
        bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_cross = bf_cross.match(des1, des2)
        # Sort matches by distance
        matches_cross = sorted(matches_cross, key=lambda x: x.distance)
        good_matches = matches_cross[:50]  # Take top 50 best matches
        
    if len(good_matches) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2
