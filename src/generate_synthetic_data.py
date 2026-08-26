"""
Synthetic Lunar Image Generator.
Generates physically plausible lunar-like heightmaps with craters,
applies Lambertian reflectance with different sun angles, and warps one
to create a geometrically distorted source image.
Saves the results in data/sample/ along with the ground truth homography.
"""
import os
import numpy as np
import cv2

def generate_lunar_heightmap(size: int = 512, num_craters: int = 25, seed: int = 42) -> np.ndarray:
    """Generates a synthetic lunar heightmap with craters and surface roughness."""
    np.random.seed(seed)
    H = np.zeros((size, size), dtype=np.float32)
    
    # 1. Add random craters of varying sizes
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    for _ in range(num_craters):
        cx = np.random.uniform(0.1 * size, 0.9 * size)
        cy = np.random.uniform(0.1 * size, 0.9 * size)
        R = np.random.uniform(10, 60)  # Radius
        D = np.random.uniform(5, 25)   # Depth / Height scale
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Crater bowl (depression)
        bowl = -D * np.exp(-dist**2 / (2 * (0.5 * R)**2))
        
        # Raised crater rim
        rim = (0.25 * D) * np.exp(-(dist - R)**2 / (2 * (0.15 * R)**2))
        
        H += (bowl + rim)
        
    # 2. Add fractal surface roughness (smoothed random noise)
    noise = np.random.normal(0, 1.0, (size, size)).astype(np.float32)
    noise_smoothed = cv2.GaussianBlur(noise, (15, 15), 0)
    H += noise_smoothed * 8.0
    
    # Normalize heightmap range to [0, 100] for consistency
    H = (H - H.min()) / (H.max() - H.min()) * 100.0
    return H

def shade_surface(H: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Applies Lambertian reflectance shading given sun azimuth and elevation angles."""
    # Convert angles to radians
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    # Compute surface gradients / normals
    dy, dx = np.gradient(H)
    # Normals N = (-dx, -dy, 1.0)
    norm = np.sqrt(dx**2 + dy**2 + 1.0)
    Nx = -dx / norm
    Ny = -dy / norm
    Nz = 1.0 / norm
    
    # Compute light vector L
    Lx = np.cos(elevation) * np.cos(azimuth)
    Ly = np.cos(elevation) * np.sin(azimuth)
    Lz = np.sin(elevation)
    
    # Dot product N.L (Lambertian shading)
    shading = Nx * Lx + Ny * Ly + Nz * Lz
    shading = np.clip(shading, 0.0, 1.0)
    
    # Convert to 8-bit image range [0, 255]
    shading_img = (shading * 255.0).astype(np.uint8)
    return shading_img

def main():
    os.makedirs("data/sample", exist_ok=True)
    
    size = 512
    # Generate common heightmap
    H = generate_lunar_heightmap(size=size, num_craters=25, seed=42)
    
    # Render Reference Image: Sun from North-East (45 deg azimuth, 55 deg elevation)
    ref_shaded = shade_surface(H, azimuth_deg=45.0, elevation_deg=55.0)
    
    # Render Source Image before warping: Sun from South-West (225 deg azimuth, 45 deg elevation)
    # This creates a massive 180-degree change in shadow directions!
    src_shaded_base = shade_surface(H, azimuth_deg=225.0, elevation_deg=45.0)
    
    # Define geometric transformation for Source Image (Translation, Rotation, Scale, Shear)
    # We want a rotation of ~12 degrees, scale of 0.9, translation (15, -10)
    center = (size / 2.0, size / 2.0)
    angle = 12.0
    scale = 0.90
    
    # 2D Affine transform matrix
    M_affine = cv2.getRotationMatrix2D(center, angle, scale)
    M_affine[0, 2] += 15.0  # translate x
    M_affine[1, 2] -= 10.0  # translate y
    
    # Convert to 3x3 Homography matrix
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[0:2, :] = M_affine
    
    # Apply warp perspective to source image
    src_warped = cv2.warpPerspective(src_shaded_base, H_gt, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Add sensor noise (speckle and Gaussian noise)
    np.random.seed(99)
    noise_ref = np.random.normal(0, 3.0, (size, size))
    noise_src = np.random.normal(0, 3.0, (size, size))
    
    ref_final = np.clip(ref_shaded.astype(np.float32) + noise_ref, 0, 255).astype(np.uint8)
    src_final = np.clip(src_warped.astype(np.float32) + noise_src, 0, 255).astype(np.uint8)
    
    # Save files
    ref_path = "data/sample/reference.png"
    src_path = "data/sample/source.png"
    cv2.imwrite(ref_path, ref_final)
    cv2.imwrite(src_path, src_final)
    
    gt_h_path = "data/sample/ground_truth_h.npy"
    np.save(gt_h_path, H_gt)
    
    print(f"Generated synthetic dataset successfully!")
    print(f"  Reference image saved to: {ref_path}")
    print(f"  Source image saved to: {src_path}")
    print(f"  Ground truth homography saved to: {gt_h_path}")
    print(f"Homography matrix:\n{H_gt}")

if __name__ == "__main__":
    main()
