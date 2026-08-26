"""
Streamlit Web Application for Lunar Image Registration Demo.
Provides a premium visual interface for aligning Chandrayaan-2 and LRO NAC images.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

from src.registration_pipeline import run_registration
from src.utils import plot_matches, overlay_images, create_alignment_composite

# Page Configurations
st.set_page_config(
    page_title="CHRONOS - Lunar Image Registration",
    page_icon="🌖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Injection
st.markdown("""
    <style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #1E88E5;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #78909C;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #1565C0 !important;
        color: white !important;
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
        height: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Headers
st.markdown("<h1 class='main-title'>🌖 C.H.R.O.N.O.S</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'><b>Chandrayaan-based Registration across Optical Sensors</b><br>Smart India Hackathon 2026 — sponsored by ISRO | Department of Space</p>", unsafe_allow_html=True)

# Sidebar - Pipeline Settings
st.sidebar.header("🛠️ Pipeline Configurations")

method_disp = {
    "LoFTR (Deep Learning)": "loftr",
    "Phase Congruency (Illum-Invariant)": "phase",
    "SIFT Baseline": "sift",
    "ORB Baseline": "orb"
}
selected_method_name = st.sidebar.selectbox("Matching Strategy", list(method_disp.keys()))
method = method_disp[selected_method_name]

st.sidebar.markdown("---")

# Collapsible Settings Panels
with st.sidebar.expander("✨ Preprocessing Settings", expanded=False):
    enable_clahe = st.checkbox("Enable CLAHE", value=True)
    clahe_clip_limit = st.slider("CLAHE Clip Limit", 0.5, 5.0, 2.5, 0.5)
    grid_size_val = st.slider("CLAHE Grid Size (NxN)", 4, 16, 8, 2)
    clahe_grid_size = (grid_size_val, grid_size_val)
    scale_factor = st.slider("Downscale Image (compute speed)", 0.25, 1.0, 1.0, 0.25)

with st.sidebar.expander("🎯 Matcher & RANSAC Settings", expanded=False):
    enable_subpixel = st.checkbox("Sub-pixel Refinement (cornerSubPix)", value=True)
    ransac_threshold = st.slider("RANSAC Rejection Threshold (px)", 1.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("---")
use_sample_data = st.sidebar.checkbox("Use Demo Synthetic Lunar Dataset", value=True)
st.sidebar.info(
    "If checked, default synthetic lunar crater images under 180° shadow changes are loaded automatically."
)

# Main Area Layout - Uploads
col1, col2 = st.columns(2)

source_file = None
ref_file = None

with col1:
    st.subheader("📸 Source (Moving) Image")
    if not use_sample_data:
        source_file = st.file_uploader("Upload Chandrayaan-2 (OHRC/TMC/IIRS) Image", type=["png", "jpg", "tif", "img"])
    else:
        st.write("Using default synthetic source: `data/sample/source.png` (Sun azimuth: 225°)")
        if os.path.exists("data/sample/source.png"):
            st.image("data/sample/source.png", width=350, caption="Source Image (Warped & Illuminated)")
        else:
            st.warning("Sample dataset not found. Please run synthetic generation or untick 'Use Demo'.")

with col2:
    st.subheader("🗺️ Reference (Fixed) Image")
    if not use_sample_data:
        ref_file = st.file_uploader("Upload Lunar Reference Image (LRO NAC)", type=["png", "jpg", "tif", "img"])
    else:
        st.write("Using default synthetic reference: `data/sample/reference.png` (Sun azimuth: 45°)")
        if os.path.exists("data/sample/reference.png"):
            st.image("data/sample/reference.png", width=350, caption="Reference Image (Fixed)")
        else:
            st.warning("Sample dataset not found. Please run synthetic generation or untick 'Use Demo'.")

st.markdown("---")

# Execution trigger
if st.button("🚀 Execute Registration Pipeline"):
    # 1. Resolve paths (use uploaded files or fallback to sample)
    src_path = None
    ref_path = None
    
    # Save uploaded files temporarily to read paths
    if not use_sample_data:
        if source_file is not None and ref_file is not None:
            # Temp source
            temp_dir = tempfile.gettempdir()
            src_path = os.path.join(temp_dir, source_file.name)
            with open(src_path, "wb") as f:
                f.write(source_file.getbuffer())
            
            # Temp reference
            ref_path = os.path.join(temp_dir, ref_file.name)
            with open(ref_path, "wb") as f:
                f.write(ref_file.getbuffer())
        else:
            st.error("Please upload both Source and Reference images, or check 'Use Demo Synthetic Lunar Dataset' in the sidebar.")
    else:
        src_path = "data/sample/source.png"
        ref_path = "data/sample/reference.png"
        
    if src_path and ref_path:
        # Build config
        pipeline_config = {
            "enable_clahe": enable_clahe,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_grid_size": clahe_grid_size,
            "scale_factor": scale_factor,
            "ransac_threshold": ransac_threshold,
            "enable_subpixel": enable_subpixel
        }
        
        with st.spinner("Processing pipeline (Loading -> Preprocessing -> Keypoint Matching -> Outlier Rejection -> Alignment)..."):
            try:
                # Execute pipeline
                warped, metrics, details = run_registration(
                    source_path=src_path,
                    reference_path=ref_path,
                    method=method,
                    config=pipeline_config
                )
                
                st.success("Registration pipeline finished successfully!")
                
                # 2. Render Metrics Widgets
                st.subheader("📊 Performance Evaluation Metrics")
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("RANSAC Inliers", f"{metrics['inlier_count']} / {metrics['total_matches']}")
                m_col2.metric("Inlier Ratio", f"{metrics['inlier_ratio'] * 100.0:.1f}%")
                m_col3.metric("Reprojection RMSE", f"{metrics['rmse']:.4f} px")
                m_col4.metric("Grid Occupancy (16 cells)", f"{metrics['grid_occupancy'] * 100.0:.1f}%")
                m_col5.metric("Uniformity Index", f"{metrics['uniformity_index']:.4f}")
                
                # SIFT/ORB Fail warning
                if method in ["sift", "orb"] and metrics['inlier_count'] < 8:
                    st.warning(
                        "⚠️ VERY LOW INLIER MATCH COUNT. Classical SIFT/ORB struggle heavily under this shadow change. "
                        "Try executing 'LoFTR (Deep Learning)' for dense correspondences!"
                    )
                
                # 3. Render Visualizations
                st.markdown("---")
                st.subheader("🖼️ Registration Visualizations")
                
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.markdown("**Matched Keypoints (Green = Inliers, Red = Outliers)**")
                    fig = plot_matches(
                        details["src_preprocessed"],
                        details["ref_preprocessed"],
                        details["pts1"],
                        details["pts2"],
                        details["mask"]
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with v_col2:
                    st.markdown("**Alignment Overlay Inspection**")
                    overlay_type = st.radio("Overlay View", ["False-Color composite (Red=Warped, Cyan=Reference)", "Alpha Blended (50-50)"], horizontal=True)
                    
                    if overlay_type.startswith("False-Color"):
                        composite = create_alignment_composite(details["ref_raw"], warped)
                        # Convert BGR to RGB for streamlit
                        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                        st.image(composite_rgb, use_container_width=True, caption="Grayscale features align perfectly. Shadows/lighting differences trigger colored edges.")
                    else:
                        blend = overlay_images(details["ref_raw"], warped, alpha=0.5)
                        blend_rgb = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
                        st.image(blend_rgb, use_container_width=True, caption="50-50 Alpha Blend overlay")
                        
                # 4. Downloads
                st.markdown("---")
                st.subheader("💾 Export Products")
                d_col1, d_col2 = st.columns(2)
                
                # Save warped image to byte array for download
                is_success, buffer = cv2.imencode(".png", warped)
                if is_success:
                    d_col1.download_button(
                        label="💾 Download Registered (Warped) Image",
                        data=buffer.tobytes(),
                        file_name=f"registered_lunar_output_{method}.png",
                        mime="image/png"
                    )
                    
                # Save JSON metrics report for download
                metrics_json = json.dumps(metrics, indent=4)
                d_col2.download_button(
                    label="💾 Download Evaluation Metrics JSON",
                    data=metrics_json,
                    file_name=f"registration_metrics_{method}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Pipeline Execution Failed: {e}")
                st.exception(e)
