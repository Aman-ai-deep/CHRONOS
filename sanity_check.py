"""
Environment Sanity Check Script.
Imports major libraries required for the Lunar Image Registration pipeline
and prints version numbers to confirm proper setup.
"""
import sys

def check_libraries():
    libraries = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("skimage", "scikit-image"),
        ("rasterio", "Rasterio"),
        ("pdr", "PlanetaryDataReader"),
        ("phasepack", "Phasepack"),
        ("streamlit", "Streamlit"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("kornia", "Kornia"),
    ]
    
    print("=" * 50)
    print("SIH LUNAR REGISTRATION - ENVIRONMENT CHECK")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print("-" * 50)
    
    all_ok = True
    for lib_name, print_name in libraries:
        try:
            # Special case for scikit-image import check
            if lib_name == "skimage":
                import skimage
                version = skimage.__version__
            else:
                module = __import__(lib_name)
                version = getattr(module, "__version__", "Unknown Version")
            print(f"[ SUCCESS ] {print_name:<25} : Version {version}")
        except ImportError as e:
            print(f"[ FAILED  ] {print_name:<25} : Could not import! Error: {e}")
            all_ok = False
            
    # GDAL is often imported as standard OSGeo, checking explicitly
    try:
        from osgeo import gdal
        print(f"[ SUCCESS ] {'GDAL':<25} : Version {gdal.__version__}")
    except ImportError as e:
        print(f"[ FAILED  ] {'GDAL':<25} : Could not import osgeo.gdal! Error: {e}")
        all_ok = False
        
    print("=" * 50)
    if all_ok:
        print("ALL LIBRARIES DETECTED AND SUCCESSFUL! READY FOR PHASE 1.")
    else:
        print("WARNING: Some libraries are missing. Please verify your pip installs.")
    print("=" * 50)
    
    return all_ok

if __name__ == "__main__":
    check_libraries()
