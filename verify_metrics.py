"""
Execution script for Phase 6 - Evaluation Metrics.
Runs SIFT baseline and LoFTR deep learning pipelines,
computes advanced metrics (RMSE, Grid Occupancy, Uniformity Index),
saves structured JSON reports to outputs/metrics/, and prints a comparison.
"""
import os
from src.registration_pipeline import run_registration
from src.evaluate import save_metrics_report

def main():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    os.makedirs("outputs/metrics", exist_ok=True)
    
    print("=" * 70)
    print("RUNNING ADVANCED METRICS EVALUATION")
    print("=" * 70)
    
    config = {
        "enable_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "ransac_threshold": 5.0,
        "enable_subpixel": True
    }
    
    methods = [
        ("sift", "SIFT Baseline"),
        ("loftr", "LoFTR (Deep Learning)")
    ]
    
    results = {}
    
    for method_key, method_name in methods:
        print(f"Running registration using {method_name}...")
        try:
            _, metrics, _ = run_registration(
                source_path=src_path,
                reference_path=ref_path,
                method=method_key,
                config=config
            )
            
            # Save metrics to JSON file
            report_path = f"outputs/metrics/{method_key}_metrics.json"
            save_metrics_report(metrics, report_path)
            print(f"  Saved JSON report to: {report_path}")
            
            results[method_key] = metrics
        except Exception as e:
            print(f"  Error running {method_name}: {e}")
            
    # Print metrics comparison table
    print("\n" + "=" * 80)
    print(f"{'Metric':<25} | {'SIFT Baseline':<25} | {'LoFTR (Deep Learning)':<25}")
    print("-" * 80)
    
    # We list all evaluation parameters
    metric_labels = [
        ("rmse", "Reprojection RMSE (px)", "{:.4f}"),
        ("inlier_count", "RANSAC Inliers", "{:d}"),
        ("total_matches", "Total Matches", "{:d}"),
        ("inlier_ratio", "Inlier Ratio (%)", "{:.1%}"),
        ("grid_occupancy", "Grid Occupancy (16 cells)", "{:.1%}"),
        ("uniformity_index", "Uniformity Index", "{:.4f}"),
        ("estimated_scale", "Estimated Scale", "{:.2f}")
    ]
    
    for key, label, fmt in metric_labels:
        sift_val = results.get("sift", {}).get(key, "N/A")
        loftr_val = results.get("loftr", {}).get(key, "N/A")
        
        sift_str = fmt.format(sift_val) if isinstance(sift_val, (int, float)) else str(sift_val)
        loftr_str = fmt.format(loftr_val) if isinstance(loftr_val, (int, float)) else str(loftr_val)
        
        print(f"{label:<25} | {sift_str:<25} | {loftr_str:<25}")
        
    print("=" * 80)
    print("Grid Occupancy measures the fraction of cells containing >= 1 inlier match.")
    print("Uniformity Index (exp(-CV)) measures the dispersion of matches across the grid.")
    print("=" * 80)

if __name__ == "__main__":
    main()
