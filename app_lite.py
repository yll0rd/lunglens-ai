"""
LungLens AI - Lightweight Version for Streamlit Cloud

Simplified version without Grad-CAM (requires less dependencies).
Use this if the full app fails to deploy.
"""

import io
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "nih_outputs" / "mobile_export"
DEFAULT_THRESHOLD = 0.5791015625
EFFICIENTNET_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@st.cache_resource
def load_onnx_model():
    """Load ONNX classifier."""
    onnx_path = MODEL_DIR / "lunglens_effnet_b0.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")
    return ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNet-B0."""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Resize
    image = image.resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE), Image.Resampling.LANCZOS)

    # Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # NCHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


def run_inference(preprocessed_image: np.ndarray, ort_session, threshold: float) -> Tuple[float, str, str]:
    """Run ONNX inference."""
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    output = ort_session.run([output_name], {input_name: preprocessed_image})
    logits = output[0][0]

    # Convert to probability (assuming binary classification)
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    tumour_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    tumour_prob = np.clip(tumour_prob, 0.0, 1.0)

    # Classify
    if tumour_prob >= threshold:
        class_label = "tumour_positive"
        class_name = "High tumour suspicion"
    else:
        class_label = "tumour_negative"
        class_name = "Low tumour suspicion"

    return tumour_prob, class_label, class_name


def format_triage_note(tumour_prob: float, class_name: str, threshold: float) -> str:
    """Generate triage note."""
    if tumour_prob >= threshold:
        recommendation = (
            "Warning: Prioritise for radiologist review. "
            "Consider CT referral if clinically appropriate."
        )
    else:
        recommendation = (
            "OK: Routine radiologist review. Standard follow-up protocol."
        )

    return f"""
### AI Triage Summary

**AI Risk Score:** {tumour_prob:.4f} (threshold: {threshold:.4f})

**Classification:** {class_name}

**Recommendation:** {recommendation}

---
*This is a research prototype, not a medical device.*
"""


def main():
    """Main app."""
    st.set_page_config(page_title="LungLens AI", page_icon="", layout="wide")

    st.title("LungLens AI - Chest X-ray Triage")

    st.markdown("""
    Binary triage tool for tumour vs no tumour classification.
    **For research and demonstration only - not a diagnostic tool.**
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.01,
        )

        st.warning(
            "This prototype is for research only. Not a medical device. "
            "Must not be used as a standalone diagnostic tool."
        )

    # Upload
    st.header("Upload Chest X-ray")
    uploaded_file = st.file_uploader("Select image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Please upload a chest X-ray image")
        return

    # Load image
    try:
        image = Image.open(io.BytesIO(uploaded_file.getbuffer()))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_container_width=True)

    # Inference
    if st.button("Run LungLens AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                # Load model
                ort_session = load_onnx_model()

                # Preprocess
                preprocessed = preprocess_image(image)

                # Infer
                tumour_prob, class_label, class_name = run_inference(
                    preprocessed, ort_session, threshold
                )

            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.success("Analysis complete!")

        # Display results
        col_score, col_class = st.columns(2)

        with col_score:
            st.metric("AI Risk Score", f"{tumour_prob:.4f}")

        with col_class:
            status = "HIGH RISK" if tumour_prob >= threshold else "LOW RISK"
            st.metric("Classification", status)

        # Triage note
        note = format_triage_note(tumour_prob, class_name, threshold)
        st.markdown(note)


if __name__ == "__main__":
    main()
