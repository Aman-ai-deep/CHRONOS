"""
Script to verify that the unified data loader works correctly.
Loads the synthetic images, checks their shape and datatype,
and creates a side-by-side verification plot.
"""
import os
import matplotlib.pyplot as plt
from src.data_loader import load_image

def verify_and_plot():
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    
    print("=" * 60)
    print("VERIFYING DATA LOADING LAYER")
    print("=" * 60)
    
    # Check file paths exist
    assert os.path.exists(ref_path), f"Missing reference file: {ref_path}"
    assert os.path.exists(src_path), f"Missing source file: {src_path}"
    
    # Load using unified load_image interface
    print(f"Loading reference image from: {ref_path} ...")
    ref_img, ref_meta = load_image(ref_path)
    print(f"  Shape: {ref_img.shape}, Dtype: {ref_img.dtype}")
    print(f"  Metadata: {ref_meta}")
    
    print(f"Loading source image from: {src_path} ...")
    src_img, src_meta = load_image(src_path)
    print(f"  Shape: {src_img.shape}, Dtype: {src_img.dtype}")
    print(f"  Metadata: {src_meta}")
    
    # Save a side-by-side visual validation plot
    os.makedirs("outputs/match_visualizations", exist_ok=True)
    out_path = "outputs/match_visualizations/sample_pair.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref_img, cmap='gray')
    axes[0].set_title(f"Reference (Fixed)\nShape: {ref_img.shape} | Sun azimuth: 45°")
    axes[0].axis('off')
    
    axes[1].imshow(src_img, cmap='gray')
    axes[1].set_title(f"Source (Moving)\nWarped: rot 12°, scale 0.9 | Sun azimuth: 225°")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print("-" * 60)
    print(f"Success! Side-by-side validation image saved to: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    verify_and_plot()
