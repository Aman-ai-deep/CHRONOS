"""
Comparative script to evaluate SIFT vs. Phase Congruency vs. LoFTR
on the synthetic lunar dataset under 180-degree shadow changes.
"""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.registration_pipeline import run_registration
from src.utils import plot_matches, create_alignment_composite

def main():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    os.makedirs("outputs/match_visualizations", exist_ok=True)
    os.makedirs("outputs/registered_images", exist_ok=True)
    
    print("=" * 70)
    print("SIH LUNAR IMAGE REGISTRATION - COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    config = {
        "enable_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "ransac_threshold": 5.0
    }
    
    methods = [
        ("sift", "SIFT Baseline"),
        ("phase", "Phase Congruency (Illum-Invariant)"),
        ("loftr", "LoFTR (Deep Learning)")
    ]
    
    results = {}
    
    for method_key, method_name in methods:
        print(f"\nEvaluating: {method_name} ...")
        t_start = time.time()
        
        try:
            warped, metrics, details = run_registration(
                source_path=src_path,
                reference_path=ref_path,
                method=method_key,
                config=config
            )
            t_elapsed = time.time() - t_start
            
            # Print performance metrics
            print(f"  Execution Time   : {t_elapsed:.2f} seconds")
            print(f"  Total Matches    : {metrics['total_matches']}")
            print(f"  RANSAC Inliers   : {metrics['inlier_count']}")
            print(f"  Inlier Ratio     : {metrics['inlier_ratio'] * 100.0:.1f}%")
            print(f"  Reprojection RMSE: {metrics['rmse']:.4f} pixels")
            print(f"  Estimated Scale  : {metrics['estimated_scale']:.2f}")
            
            # Store results for summary
            results[method_key] = {
                "name": method_name,
                "time": t_elapsed,
                "matches": metrics['total_matches'],
                "inliers": metrics['inlier_count'],
                "ratio": metrics['inlier_ratio'],
                "rmse": metrics['rmse'],
                "scale": metrics['estimated_scale']
            }
            
            # 1. Save match visualization
            fig = plot_matches(
                details["src_preprocessed"],
                details["ref_preprocessed"],
                details["pts1"],
                details["pts2"],
                details["mask"]
            )
            fig.savefig(f"outputs/match_visualizations/compare_{method_key}_matches.png", dpi=150)
            plt.close(fig)
            
            # 2. Save false-color composite
            composite = create_alignment_composite(details["ref_raw"], warped)
            cv2.imwrite(f"outputs/registered_images/compare_{method_key}_composite.png", composite)
            
        except Exception as e:
            print(f"  Failed running {method_name}: {e}")
            
    # Print comparison summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<32} | {'Time (s)':<8} | {'Matches':<8} | {'Inliers':<8} | {'Inlier %':<8} | {'RMSE':<8}")
    print("-" * 70)
    for m_key, _ in methods:
        if m_key in results:
            res = results[m_key]
            print(f"{res['name']:<32} | {res['time']:<8.2f} | {res['matches']:<8} | {res['inliers']:<8} | {res['ratio']*100.0:<7.1f}% | {res['rmse']:<8.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
