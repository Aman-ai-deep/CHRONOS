# Multi-modal Lunar Image Registration Pipeline

A software prototype for **Smart India Hackathon (SIH) Problem Statement ID 26166**, sponsored by **ISRO (Department of Space)**.

## Problem Statement Summary
**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

**Goal:** Establish correspondence (match points) with **sub-pixel accuracy** and **uniform spatial distribution** between Chandrayaan-2 acquired optical/hyperspectral images (OHRC, TMC-2, IIRS) and lunar reference images (LRO NAC), despite significant challenges:
- **Illumination variations** (changing sun azimuth/elevation, shadows)
- **Viewpoint variations** (rotation, translation, perspective distortions)
- **Scale differences** (varying spatial resolutions/altitudes)

---

## Repository Structure

```
sih-lunar-registration/
├── README.md                      # Setup instructions + problem summary
├── requirements.txt               # Pin dependencies
├── .gitignore                     # Ignore caches, data, outputs
├── data/
│   ├── raw/                       # Untouched raw PDS images
│   ├── processed/                 # Normalized and contrast-stretched images
│   └── sample/                    # Small substitute/placeholder lunar image pairs
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Image reading interface (PDS & standard formats)
│   ├── preprocessing.py           # Dynamics normalization, CLAHE, resizing
│   ├── classical_match.py         # SIFT/ORB baseline matchers
│   ├── illum_invariant.py         # Phase congruency extraction and matching
│   ├── deep_match.py              # LoFTR deep learning feature matcher
│   ├── outlier_rejection.py       # RANSAC homography estimation
│   ├── subpixel_refine.py         # Subpixel refinement algorithms
│   ├── evaluate.py                # RMSE, Inliers, Uniformity calculation
│   ├── registration_pipeline.py   # Main end-to-end pipeline runner
│   └── utils.py                   # Plotting, overlay, and helpers
├── notebooks/                     # Exploratory notebook experiments
├── app/
│   └── streamlit_app.py           # Web GUI for demonstration
├── outputs/
│   ├── registered_images/         # Aligned warped images
│   ├── match_visualizations/      # Images showing matched connection lines
│   └── metrics/                   # Structured JSON metrics reports
├── tests/
│   └── test_pipeline.py           # Testing modules
└── docs/
    ├── architecture.md            # Diagram and details on pipeline design
    └── limitations.md             # Project constraints and future works
```

---

## Environment Setup & Verification

### Step 1: Create Virtual Environment
Run the following commands in your shell to set up a clean environment:

```powershell
# Create environment
python -m venv sih_env

# Activate environment (Windows PowerShell)
.\sih_env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2: Install Dependencies
Install all required libraries using the package manager:

```powershell
pip install -r requirements.txt
```

> [!NOTE]
> If GDAL installation fails on Windows, install it via prebuilt Windows wheels from [Gohlke's unofficial binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/) or use OSGeo4W/conda environment. For testing without planetary file parsing (.lbl/.img), the pipeline will gracefully fall back to OpenCV/Rasterio loading of standard TIFF/PNG/JPG files.

### Step 3: Run Sanity Check
Confirm that all core packages have been installed and are importable:

```powershell
python sanity_check.py
```
Check that the script prints out correct library versions without throwing errors.
