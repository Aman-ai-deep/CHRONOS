"""
Script to verify the sub-pixel coordinate refinement.
Runs the registration pipeline with and without sub-pixel refinement,
compares RMSE, and displays raw vs refined keypoint coordinates.
"""
from src.registration_pipeline import run_registration

def main():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    print("=" * 70)
    print("RUNNING SUB-PIXEL REFINEMENT VERIFICATION")
    print("=" * 70)
    
    # 1. Run WITHOUT sub-pixel refinement
    config_no_sub = {
        "enable_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "ransac_threshold": 5.0,
        "enable_subpixel": False
    }
    
    print("Running LoFTR WITHOUT sub-pixel refinement...")
    _, metrics_no_sub, details_no_sub = run_registration(
        source_path=src_path,
        reference_path=ref_path,
        method="loftr",
        config=config_no_sub
    )
    
    # 2. Run WITH sub-pixel refinement
    config_sub = {
        "enable_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "ransac_threshold": 5.0,
        "enable_subpixel": True
    }
    
    print("\nRunning LoFTR WITH sub-pixel refinement...")
    _, metrics_sub, details_sub = run_registration(
        source_path=src_path,
        reference_path=ref_path,
        method="loftr",
        config=config_sub
    )
    
    # 3. Print coordinate comparison for the first 5 inliers
    print("\n" + "-" * 60)
    print("KEYPOINT COORDINATES COMPARISON (First 5 matches)")
    print("-" * 60)
    pts1_raw = details_no_sub["pts1"]
    pts2_raw = details_no_sub["pts2"]
    
    pts1_refined = details_sub["pts1"]
    pts2_refined = details_sub["pts2"]
    
    mask = details_sub["mask"].squeeze()
    inlier_indices = [i for i, val in enumerate(mask) if val == 1][:5]
    
    print(f"{'Index':<5} | {'Raw (Pixel)':<25} | {'Refined (Sub-pixel)':<25}")
    print("-" * 60)
    for idx in inlier_indices:
        p1_r = pts1_raw[idx]
        p1_ref = pts1_refined[idx]
        print(f"{idx:<5} | ({p1_r[0]:.2f}, {p1_r[1]:.2f}){'':<10} | ({p1_ref[0]:.6f}, {p1_ref[1]:.6f})")
        
    # 4. Print metrics comparison table
    print("\n" + "=" * 70)
    print(f"{'Sub-pixel Refinement':<25} | {'RANSAC Inliers':<15} | {'RMSE (pixels)':<15}")
    print("-" * 70)
    print(f"{'Disabled':<25} | {metrics_no_sub['inlier_count']:<15} | {metrics_no_sub['rmse']:<15.4f}")
    print(f"{'Enabled (cornerSubPix)':<25} | {metrics_sub['inlier_count']:<15} | {metrics_sub['rmse']:<15.4f}")
    print("=" * 70)
    
    improvement = (metrics_no_sub['rmse'] - metrics_sub['rmse']) / metrics_no_sub['rmse'] * 100.0 if metrics_no_sub['rmse'] > 0 else 0.0
    print(f"RMSE Accuracy Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()
