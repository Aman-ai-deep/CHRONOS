# Troubleshooting Guide: Custom Image Upload Crashes in CHRONOS

This document analyzes the root causes of crashes when uploading custom images (such as WhatsApp JPEGs, phone photos, raw planetary GeoTIFFs, or PDS files) to the CHRONOS Streamlit Cloud app, and details the engineering fixes applied to solve them.

---

## 1. Primary Causes of Crashes with Custom Image Uploads

### Cause A: Raw Array Dtype & Channel Mismatch during Overlay Generation
* **Symptom:** Python throws `cv2.error: (-209:Sizes of input arguments do not match)` or `ValueError: cannot cast ufunc 'add' output from dtype`.
* **Root Cause:** In `app/streamlit_app.py`, the visualization block was passing `details["ref_raw"]` (the un-preprocessed, unnormalized original image array) into `create_alignment_composite()` and `overlay_images()`. 
  * Custom JPEGs, 16-bit TIFFs, or Float32 raw arrays have dtypes like `uint16`, `float32`, or `int32` and values in `[0, 65535]`.
  * The warped image output (`warped`) is a normalized 8-bit `uint8` array in `[0, 255]`.
  * Passing mixed dtypes into OpenCV compositing functions (`cv2.addWeighted` or channel assignment `composite[..., 1] = ref_gray`) causes OpenCV or NumPy type casting to crash immediately.
* **Fix:** 
  1. Updated `app/streamlit_app.py` to pass `details["ref_preprocessed"]` (which is guaranteed to be a normalized 8-bit `uint8` array).
  2. Updated `src/utils.py` to sanitize and normalize both inputs to 8-bit `uint8` before any compositing or alpha blending operations.

---

### Cause B: Zero Match & Homography Degeneracy on Low-Feature Images
* **Symptom:** App crashes or raises `IndexError` when rendering match lines or calculating metrics.
* **Root Cause:** WhatsApp JPEGs apply lossy compression, smoothing out fine surface textures. If two custom images have minimal overlap or low texture, feature matchers (SIFT, ORB, or LoFTR) find **0 matches** or **< 4 matches**.
  * If `pts1` is empty (`shape = (0, 2)`), RANSAC homography estimation returns an empty mask (`shape = (0, 1)`).
  * Accessing `mask[i][0]` or indexing empty arrays in visualization routines raised index errors.
* **Fix:** 
  1. Added zero-match bounds checks across `src/utils.py` (`plot_matches`, `create_alignment_composite`, `overlay_images`).
  2. If 0 inliers are found, the app gracefully renders a clear warning message rather than throwing an exception.

---

### Cause C: Filename Special Characters and Spaces
* **Symptom:** `FileNotFoundError` or `RasterioIOError` when resolving paths.
* **Root Cause:** WhatsApp image filenames contain spaces and special characters (e.g. `WhatsApp Image 2026-09-06 at 15.37.02.jpeg`). When saved to `/tmp`, unquoted path handling in certain drivers (like older GDAL/Rasterio wrappers) failed to locate the file.
* **Fix:** Added filename sanitization (replacing spaces with underscores and stripping non-alphanumeric characters) when saving uploaded files to temporary directories in `app/streamlit_app.py`.

---

### Cause E: Single-Inlier Array Shape Collapse (NumPy 0D Scalar Indexing)
* **Symptom:** `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'numpy.ndarray'` when RANSAC produces exactly 1 inlier match.
* **Root Cause:** When RANSAC produces 1 inlier (`mask` shape `(1, 1)`), calling `mask.squeeze()` collapses the array into a 0D scalar boolean `array(True)`. 
  * Indexing `pts2[array(True)]` prepends an axis, turning `(1, 2)` into `(1, 1, 2)`.
  * Later, in `calculate_uniformity()`, iterating `for pt in pts` returns a 1D array `pt = [x, y]` instead of numbers, causing `int(x)` to fail with a `TypeError` and crash Python.
* **Fix:** Replaced `.squeeze()` with `.reshape(-1)` in `src/evaluate.py` and `src/registration_pipeline.py`. `mask.reshape(-1)` always yields a clean 1D boolean array `(1,)`, keeping point array dimensions strictly `(1, 2)`.

---

## 2. Verification Matrix

| Upload Format | Issue Solved | Fix Applied |
| :--- | :--- | :--- |
| **WhatsApp JPEGs** | Type mismatch (`uint8` vs `float/int`), spaces in filename | Sanitized filenames, passed `ref_preprocessed` to overlays |
| **16-bit GeoTIFFs** | Dtype conflict (`uint16` vs `uint8`) | `sanitize_loaded_array` + `normalize_bit_depth` uint8 casting |
| **PDS `.img`/`.lbl`** | Missing label fallback crash | `pdr` fallback to standard OpenCV/Numpy binary reader |
| **Single Inlier Match** | NumPy 0D scalar boolean collapse | Replaced `.squeeze()` with `.reshape(-1)` across evaluation modules |
| **Low-texture / 0 matches** | Array index crash | Zero-match check in `plot_matches` and `estimate_homography` |

