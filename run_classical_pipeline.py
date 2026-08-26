"""
Execution script for Phase 3 - Classical Baseline.
Runs the registration pipeline using classical SIFT & ORB descriptors,
plots correspondences, warps the images, and computes false-color overlays.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.registration_pipeline import run_registration
from src.utils import plot_matches, overlay_images, create_alignment_composite

def run_baseline_comparison():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    os.makedirs("outputs/match_visualizations", exist_ok=True)
    os.makedirs("outputs/registered_images", exist_ok=True)
    
    print("=" * 60)
    print("RUNNING CLASSICAL BASELINE REGISTRATION")
    print("=" * 60)
    
    config = {
        "enable_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "ransac_threshold": 5.0
    }
    
    for method in ["SIFT", "ORB"]:
        print(f"\n--- Running {method} Pipeline ---")
        try:
            warped, metrics, details = run_registration(
                source_path=src_path,
                reference_path=ref_path,
                method=method,
                config=config
            )
            
            print(f"Metrics for {method}:")
            print(f"  Total Matches detected : {metrics['total_matches']}")
            print(f"  RANSAC Inlier matches  : {metrics['inlier_count']}")
            print(f"  Inlier Ratio           : {metrics['inlier_ratio']:.4f}")
            print(f"  Reprojection RMSE      : {metrics['rmse']:.4f} pixels")
            
            # 1. Save Matches Plot
            fig = plot_matches(
                details["src_preprocessed"], 
                details["ref_preprocessed"], 
                details["pts1"], 
                details["pts2"], 
                details["mask"]
            )
            match_plot_path = f"outputs/match_visualizations/classical_{method.lower()}_matches.png"
            fig.savefig(match_plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved match visualization to: {match_plot_path}")
            
            # 2. Save Warped Output
            warped_path = f"outputs/registered_images/classical_{method.lower()}_registered.png"
            cv2.imwrite(warped_path, warped)
            print(f"  Saved registered output to: {warped_path}")
            
            # 3. Save Overlay and False-Color Composites
            ref_raw = details["ref_raw"]
            
            # Alpha blend overlay
            overlay = overlay_images(ref_raw, warped, alpha=0.5)
            overlay_path = f"outputs/registered_images/classical_{method.lower()}_overlay.png"
            cv2.imwrite(overlay_path, overlay)
            
            # False-color composite (Red = warped, Green/Blue = reference)
            composite = create_alignment_composite(ref_raw, warped)
            composite_path = f"outputs/registered_images/classical_{method.lower()}_composite.png"
            cv2.imwrite(composite_path, composite)
            print(f"  Saved false-color composite to: {composite_path}")
            
        except Exception as e:
            print(f"  Error running {method}: {e}")
            
    print("\n" + "=" * 60)
    print("Pipeline run completed. Note that SIFT/ORB may fail under 180° lighting differences.")
    print("=" * 60)

if __name__ == "__main__":
    run_baseline_comparison()
