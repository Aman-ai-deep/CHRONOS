"""
Script to verify and visualize the effect of preprocessing.
Loads the synthetic images, processes them, and saves a 2x2 before/after plot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_image
from src.preprocessing import preprocess_image, normalize_bit_depth

def main():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    print("=" * 60)
    print("RUNNING PREPROCESSING VERIFICATION")
    print("=" * 60)
    
    # 1. Load images
    ref_img, _ = load_image(ref_path)
    src_img, _ = load_image(src_path)
    
    # 2. Process images with CLAHE enabled
    config = {
        "enable_clahe": True,
        "clahe_clip_limit": 3.0,
        "clahe_grid_size": (8, 8),
        "scale_factor": 1.0,
        "lower_pct": 1.0,
        "upper_pct": 99.0
    }
    
    ref_processed = preprocess_image(ref_img, config)
    src_processed = preprocess_image(src_img, config)
    
    # Print statistics to verify range normalization
    print(f"Reference Image:")
    print(f"  Raw       -> Min: {ref_img.min():.1f}, Max: {ref_img.max():.1f}, Mean: {ref_img.mean():.1f}")
    print(f"  Processed -> Min: {ref_processed.min():.1f}, Max: {ref_processed.max():.1f}, Mean: {ref_processed.mean():.1f}")
    
    print(f"Source Image:")
    print(f"  Raw       -> Min: {src_img.min():.1f}, Max: {src_img.max():.1f}, Mean: {src_img.mean():.1f}")
    print(f"  Processed -> Min: {src_processed.min():.1f}, Max: {src_processed.max():.1f}, Mean: {src_processed.mean():.1f}")
    
    # 3. Plot 2x2 grid (Reference Before/After, Source Before/After)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Row 1: Reference
    axes[0, 0].imshow(ref_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Reference (Raw) - Sun NE 45°")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ref_processed, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Reference (Processed: Stretch + CLAHE)")
    axes[0, 1].axis('off')
    
    # Row 2: Source
    axes[1, 0].imshow(src_img, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title("Source (Warped Raw) - Sun SW 225°")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(src_processed, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title("Source (Warped Processed: Stretch + CLAHE)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs("outputs/match_visualizations", exist_ok=True)
    out_path = "outputs/match_visualizations/preprocessing_effect.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print("-" * 60)
    print(f"Success! Preprocessing visualization saved to: {out_path}")
    print("Notice how CLAHE equalizes lighting in shadow basins and enhances crater rims!")
    print("=" * 60)

if __name__ == "__main__":
    main()
