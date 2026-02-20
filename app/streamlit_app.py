"""
Streamlit application for semantic segmentation mask prediction.

Upload an image and get a predicted segmentation mask using the model
loaded from Azure Blob Storage.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path when running: streamlit run app/streamlit_app.py
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

import streamlit as st
import numpy as np
from PIL import Image

from dotenv import load_dotenv

# Load environment variables from .env
env_path = _project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.predictor import SegmentationPredictor
from src.utils import CATEGORY_NAMES, CATEGORY_COLORS

st.set_page_config(
    page_title="Segmentation Mask Prediction",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Semantic Segmentation - Mask Prediction")
st.markdown(
    "Upload an image to get a semantic segmentation mask (8 Cityscapes categories). "
    "The model is loaded from Azure Blob Storage (configure via .env)."
)


# File upload
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG or JPG image for segmentation",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Use session state to persist prediction across reruns
    # Invalidate cache when uploaded file changes (by name + size)
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if "last_file_key" not in st.session_state or st.session_state.last_file_key != file_key:
        st.session_state.last_file_key = file_key
        st.session_state.colored_mask = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    if st.button("Predict", type="primary"):
        with st.spinner("Loading model and running prediction..."):
            try:
                predictor = SegmentationPredictor(

                    azure_blob_name=os.environ.get("AZURE_MODEL_BLOB_NAME", "model/best_model.keras"),
                    azure_container_name=os.environ.get("AZURE_CONTAINER_NAME", "training-outputs"),
                )
                colored_mask = predictor.predict_to_colored_mask(image_bytes)
                st.session_state.colored_mask = colored_mask
            except FileNotFoundError as e:
                st.error(str(e))
                st.session_state.colored_mask = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.colored_mask = None
                raise

    colored_mask = st.session_state.colored_mask
    if colored_mask is not None:
        with col2:
            st.subheader("Predicted Mask")
            st.image(colored_mask, width="stretch")
    else:
        with col2:
            st.info("Click **Predict** to run segmentation.")

else:
    st.info("👆 Upload an image to get started.")

# Footer: legend with mask colors
st.sidebar.markdown("---")
st.sidebar.markdown("### Legend (mask colors)")
for cat_id, name in CATEGORY_NAMES.items():
    r, g, b = CATEGORY_COLORS.get(cat_id, [0, 0, 0])
    st.sidebar.markdown(
        f'<span style="display: inline-block; width: 14px; height: 14px; '
        f'background-color: rgb({r},{g},{b}); border: 1px solid #333; '
        f'margin-right: 8px; vertical-align: middle;"></span> **{name}**',
        unsafe_allow_html=True,
    )
