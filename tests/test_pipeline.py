"""
Unit tests for the registration pipeline.
Simulates a translation warp and verifies that the registration engine
recovers the geometry with high inliers and low RMSE.
"""
import unittest
import os
import numpy as np
import cv2

from src.registration_pipeline import run_registration

class TestRegistrationPipeline(unittest.TestCase):
    def setUp(self):
        self.ref_path = "data/sample/reference.png"
        self.temp_src_path = "data/sample/temp_translated.png"
        
    def test_sift_registration_translation(self):
        """
        Registers an image against a slightly translated copy of itself.
        Under identical lighting, SIFT should recover the shift with near-zero RMSE.
        """
        # Skip if sample data hasn't been generated
        if not os.path.exists(self.ref_path):
            self.skipTest(f"Reference image not found at {self.ref_path}. Run generate_synthetic_data.py first.")
            
        # 1. Load reference image
        ref_img = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
        h, w = ref_img.shape[:2]
        
        # 2. Apply a known translation: dx = 12 pixels, dy = -8 pixels
        # Warp matrix M:
        # [1, 0, dx]
        # [0, 1, dy]
        M = np.float32([[1, 0, 12.0], [0, 1, -8.0]])
        translated_img = cv2.warpAffine(ref_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Save temporary moving image
        cv2.imwrite(self.temp_src_path, translated_img)
        
        try:
            # 3. Run SIFT registration
            config = {
                "enable_clahe": False,  # No contrast enhancement needed as lighting is identical
                "scale_factor": 1.0,
                "ransac_threshold": 5.0,
                "enable_subpixel": True
            }
            
            warped, metrics, details = run_registration(
                source_path=self.temp_src_path,
                reference_path=self.ref_path,
                method="sift",
                config=config
            )
            
            # Print test results
            print(f"\n[ TEST RESULT ] Registration on Translation Warp:")
            print(f"  Total matches detected: {metrics['total_matches']}")
            print(f"  RANSAC inlier matches : {metrics['inlier_count']}")
            print(f"  Reprojection RMSE     : {metrics['rmse']:.4f} pixels")
            print(f"  Estimated homography matrix:\n{details['H']}")
            
            # 4. Assert correctness
            self.assertGreaterEqual(
                metrics['inlier_count'], 10, 
                "SIFT should find at least 10 inlier matches on translation warp without lighting changes."
            )
            self.assertLess(
                metrics['rmse'], 1.5, 
                "Reprojection error (RMSE) under identical lighting should be low (< 1.5 pixels)."
            )
            
            # Verify translation terms in estimated homography matrix
            H = details['H']
            est_dx = H[0, 2]
            est_dy = H[1, 2]
            print(f"  True shift      : dx = 12.0, dy = -8.0")
            print(f"  Estimated shift : dx = {est_dx:.2f}, dy = {est_dy:.2f}")
            
            self.assertAlmostEqual(est_dx, -12.0, delta=1.5, msg="Estimated X shift is not close to true shift")
            self.assertAlmostEqual(est_dy, 8.0, delta=1.5, msg="Estimated Y shift is not close to true shift")
            
        finally:
            # Clean up temporary test file
            if os.path.exists(self.temp_src_path):
                os.remove(self.temp_src_path)

if __name__ == '__main__':
    unittest.main()
