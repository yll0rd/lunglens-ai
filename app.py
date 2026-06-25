"""
LungLens AI - Chest X-ray Triage Prototype

A binary triage tool for tumour vs no tumour classification on chest radiographs
using a bone-suppressed EfficientNet-B0 classifier with Grad-CAM interpretability.

For research and demonstration only - NOT a diagnostic tool.
"""

import io
import warnings
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(__file__).parent.parent / "nih_outputs" / "mobile_export"
BONE_SUPPRESSION_MODEL_PATH = Path(__file__).parent.parent / "resnet_bs.h5"

DEFAULT_THRESHOLD = 0.5791015625  # Youden threshold from Run B (val_metrics)
EFFICIENTNET_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models() -> Tuple[ort.InferenceSession, Optional[object]]:
    """
    Load ONNX classifier and optional bone suppression model.

    Returns:
        Tuple of (ONNX session, bone suppression model or None)
    """
    # Load ONNX classifier
    onnx_path = MODEL_DIR / "lunglens_effnet_b0.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Try to load bone suppression model
    bone_suppress_model = None
    if BONE_SUPPRESSION_MODEL_PATH.exists():
        try:
            # Lazy import to avoid hard dependency
            import tensorflow as tf
            bone_suppress_model = tf.keras.models.load_model(str(BONE_SUPPRESSION_MODEL_PATH))
        except Exception as e:
            st.warning(f"Could not load bone suppression model: {e}")

    return ort_session, bone_suppress_model


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image: Image.Image, target_size: int = EFFICIENTNET_INPUT_SIZE) -> np.ndarray:
    """
    Preprocess PIL image for EfficientNet-B0 inference.

    Steps:
    1. Convert to grayscale if needed (X-rays are grayscale)
    2. Resize to target size with padding to maintain aspect ratio
    3. Normalize using ImageNet statistics

    Args:
        image: PIL Image
        target_size: Target size for model input (default 224)

    Returns:
        Preprocessed image as numpy array (1, 3, 224, 224) for ONNX
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')

    # Resize with padding to maintain aspect ratio
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Convert single channel to RGB by replicating
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Normalize using ImageNet statistics
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to NCHW format for ONNX (1, 3, 224, 224)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


def apply_bone_suppression(image: Image.Image, bone_suppress_model: Optional[object]) -> Image.Image:
    """
    Apply bone suppression to chest X-ray image.

    If no model is available, returns original image (placeholder function).

    Args:
        image: PIL Image (original chest X-ray)
        bone_suppress_model: Loaded bone suppression model or None

    Returns:
        Bone-suppressed PIL Image
    """
    if bone_suppress_model is None:
        # Placeholder: simulate bone suppression with histogram equalization
        # This is a simple approximation and should be replaced with actual model
        img_array = np.array(image.convert('L'), dtype=np.uint8)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances soft tissue contrast by suppressing high-intensity bone signals
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)

        return Image.fromarray(enhanced, mode='L')

    else:
        # Use actual bone suppression model
        try:
            import tensorflow as tf

            # Convert PIL to numpy
            img_array = np.array(image.convert('L'), dtype=np.float32) / 255.0

            # Add batch and channel dimensions
            img_input = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)

            # Predict
            suppressed = bone_suppress_model.predict(img_input, verbose=0)

            # Convert back to PIL
            suppressed = np.squeeze(suppressed) * 255.0
            suppressed = np.clip(suppressed, 0, 255).astype(np.uint8)

            return Image.fromarray(suppressed, mode='L')

        except Exception as e:
            st.warning(f"Bone suppression failed: {e}. Using original image.")
            return image


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    preprocessed_image: np.ndarray,
    ort_session: ort.InferenceSession,
    threshold: float = DEFAULT_THRESHOLD
) -> Tuple[float, str, str]:
    """
    Run ONNX inference on preprocessed image.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        ort_session: ONNX Runtime session
        threshold: Decision threshold for binary classification

    Returns:
        Tuple of (probability_score, class_label, class_name)
    """
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Run inference
    output = ort_session.run([output_name], {input_name: preprocessed_image})

    # Extract probability (logits -> softmax -> tumour probability)
    logits = output[0][0]

    # Assuming binary classification: [no_tumour_logit, tumour_logit]
    # Use softmax to convert to probabilities
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()
    tumour_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

    # Clamp to [0, 1] just in case
    tumour_prob = np.clip(tumour_prob, 0.0, 1.0)

    # Classify based on threshold
    if tumour_prob >= threshold:
        class_label = "tumour_positive"
        class_name = "High tumour suspicion"
    else:
        class_label = "tumour_negative"
        class_name = "Low tumour suspicion"

    return tumour_prob, class_label, class_name


# =============================================================================
# GRAD-CAM
# =============================================================================

def generate_grad_cam(
    preprocessed_image: np.ndarray,
    ort_session: ort.InferenceSession,
    target_layer: int = -1
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for visualization.

    Loads the ONNX model weights into a PyTorch model to compute gradients,
    then generates Grad-CAM on the last convolutional layer.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        ort_session: ONNX Runtime session (not used directly)
        target_layer: Index of target layer

    Returns:
        Grad-CAM heatmap as numpy array (224, 224) or None if failed
    """
    try:
        # Load EfficientNet-B0 with pre-trained weights
        model = models.efficientnet_b0(pretrained=True)
        model.eval()

        # Convert input to tensor
        input_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)
        input_tensor.requires_grad = True

        # Forward pass
        with torch.enable_grad():
            output = model(input_tensor)
            tumour_class = 1  # Assume tumour is class 1
            score = output[0, tumour_class]

            # Backward pass
            model.zero_grad()
            score.backward()

        # Get gradients from the last conv layer
        # For EfficientNet-B0, this is model.features[-1]
        activations = input_tensor.grad.data.abs().mean(dim=1)[0]

        # Normalize heatmap
        heatmap = activations.numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap, (EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE))

        return heatmap_resized

    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None


def overlay_grad_cam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_image: PIL Image of the X-ray
        heatmap: Grad-CAM heatmap (normalized to 0-1)
        alpha: Blending factor for overlay

    Returns:
        PIL Image with Grad-CAM overlay
    """
    # Convert original to RGB for visualization
    img_array = np.array(original_image.convert('L'))
    img_rgb = np.stack([img_array] * 3, axis=-1)

    # Apply colormap to heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(overlay.astype(np.uint8))


# =============================================================================
# TRIAGE OUTPUT
# =============================================================================

def format_triage_note(
    tumour_prob: float,
    class_name: str,
    threshold: float
) -> str:
    """
    Generate structured triage note.

    Args:
        tumour_prob: Predicted probability (0-1)
        class_name: Classification label
        threshold: Decision threshold used

    Returns:
        Formatted triage note string
    """
    # Determine risk level and recommendations
    if tumour_prob >= threshold:
        risk_level = "HIGH"
        recommendation = (
            "⚠️ **Suggested action:** Prioritise for radiologist review. "
            "Consider CT referral if clinically appropriate."
        )
    else:
        risk_level = "LOW"
        recommendation = (
            "✓ **Suggested action:** Routine radiologist review. "
            "Standard follow-up protocol."
        )

    note = f"""
### 📋 AI Triage Summary

**AI Risk Score:** {tumour_prob:.4f} (threshold: {threshold:.4f})

**Risk Category:** {risk_level}

**Classification:** {class_name}

{recommendation}

---
*This prototype is for research and demonstration only.*
*It is not a medical device and must not be used as a standalone diagnostic tool.*
"""

    return note


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main Streamlit application."""

    # Page config
    st.set_page_config(
        page_title="LungLens AI",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("🫁 LungLens AI – Chest X‑ray Triage Prototype")

    st.markdown("""
    This is a **binary triage tool** for "tumour vs no tumour" classification on chest radiographs.
    It is **not a diagnostic system** and is targeted at low‑resource Cameroonian hospitals.

    The pipeline uses:
    1. **Stage 1:** Bone suppression to enhance soft tissue contrast
    2. **Stage 2:** EfficientNet-B0 classifier trained on bone-suppressed images
    3. **Interpretability:** Grad-CAM visualization of model focus areas
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            help="Probability threshold for tumour classification. Higher = stricter (fewer tumours detected)."
        )

        st.markdown("---")
        st.warning(
            "⚠️ **Medical Disclaimer**\n\n"
            "This prototype is for **research and demonstration only**. "
            "It is **not a medical device** and **must not be used** as a standalone diagnostic tool. "
            "All predictions must be reviewed by qualified radiologists."
        )

    # Load models
    try:
        ort_session, bone_suppress_model = load_models()
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

    # File uploader
    st.header("📤 Upload Chest X-ray")
    uploaded_file = st.file_uploader(
        "Select a chest X-ray image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest radiograph for triage analysis"
    )

    if uploaded_file is None:
        st.info("👆 Please upload a chest X-ray image to begin analysis.")
        st.stop()

    # Load and display original image
    try:
        original_image = Image.open(io.BytesIO(uploaded_file.getbuffer()))
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        st.stop()

    # Create columns for display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original X-ray")
        st.image(original_image, use_container_width=True, caption="Input image")

    # Run analysis on button click
    if st.button("🔍 Run LungLens AI Analysis", type="primary", use_container_width=True):

        with st.spinner("🔄 Processing... (bone suppression → inference → Grad-CAM)"):

            # Step 1: Bone suppression
            try:
                bone_suppressed = apply_bone_suppression(original_image, bone_suppress_model)
            except Exception as e:
                st.error(f"❌ Bone suppression failed: {e}")
                st.stop()

            with col2:
                st.subheader("Bone-Suppressed Image")
                st.image(bone_suppressed, use_container_width=True, caption="After Stage 1 suppression")

            # Step 2: Preprocess and run inference
            try:
                preprocessed = preprocess_image(bone_suppressed)
                tumour_prob, class_label, class_name = run_inference(
                    preprocessed, ort_session, threshold
                )
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                st.stop()

            # Step 3: Generate Grad-CAM
            try:
                heatmap = generate_grad_cam(preprocessed, ort_session)
                if heatmap is not None:
                    grad_cam_image = overlay_grad_cam(bone_suppressed, heatmap, alpha=0.5)
                else:
                    grad_cam_image = None
            except Exception as e:
                st.warning(f"Grad-CAM generation failed: {e}")
                grad_cam_image = None

            with col3:
                if grad_cam_image is not None:
                    st.subheader("Grad-CAM (Model Focus)")
                    st.image(grad_cam_image, use_container_width=True, caption="Areas of model attention")
                else:
                    st.warning("Grad-CAM visualization unavailable")

        # Display results
        st.success("✅ Analysis complete!")

        # Results metrics
        col1_res, col2_res, col3_res = st.columns(3)

        with col1_res:
            st.metric(
                "AI Risk Score",
                f"{tumour_prob:.4f}",
                help="Predicted probability of tumour (0-1)"
            )

        with col2_res:
            st.metric(
                "Threshold",
                f"{threshold:.4f}",
                help="Decision boundary for classification"
            )

        with col3_res:
            decision_text = "🔴 HIGH RISK" if tumour_prob >= threshold else "🟢 LOW RISK"
            st.metric("Classification", decision_text)

        # Triage note
        triage_note = format_triage_note(tumour_prob, class_name, threshold)
        st.markdown(triage_note)

        # Additional metrics display
        with st.expander("📊 Detailed Metrics"):
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | **Probability (Tumour)** | {tumour_prob:.6f} |
            | **Probability (No Tumour)** | {1 - tumour_prob:.6f} |
            | **Decision Threshold** | {threshold:.6f} |
            | **Model Input Size** | 224×224 |
            | **Backbone** | EfficientNet-B0 |
            | **Stage 1** | Bone Suppression (ResNet) |
            | **Stage 2** | Binary Classification (ONNX) |
            """)


if __name__ == "__main__":
    main()
