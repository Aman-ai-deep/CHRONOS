"""
Streamlit Web Application for Lunar Image Registration Demo.
"""
import streamlit as st
import numpy as np

st.set_page_config(page_title="SIH Lunar Image Registration", layout="wide")

st.title("Multi-modal Lunar Image Registration UI")
st.write("Smart India Hackathon 2026 - Problem Statement 26166")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Source Image (Chandrayaan-2)")
    source_file = st.file_uploader("Upload Source Image", type=["png", "jpg", "tif", "img"])

with col2:
    st.subheader("Reference Image (LRO NAC)")
    ref_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "tif", "img"])

method = st.selectbox("Registration Method", ["SIFT Baseline", "Phase Congruency (Illumination-Invariant)", "LoFTR (Deep Learning)"])

if st.button("Run Registration"):
    st.info("Running registration pipeline...")
