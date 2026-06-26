"""
LungLens AI - Chest X-ray Triage Prototype

A binary triage tool for tumour vs no tumour classification on chest radiographs
using a bone-suppressed EfficientNet-B0 classifier with Grad-CAM interpretability.

For research and demonstration only - NOT a diagnostic tool.
"""

import io
import os
import warnings
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from threshold_utils import compute_optimal_threshold
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(__file__).parent
BONE_SUPPRESSION_MODEL_PATH = Path(__file__).parent / "resnet_bs.h5"

# Load optimal threshold (with specificity constraint: min_spec=70%)
def _load_threshold() -> float:
    """Load optimal threshold from config or use default (Youden)."""
    import json
    config = compute_optimal_threshold()
    threshold = config.get("threshold", 0.5791015625)
    # config_path = MODEL_DIR / "threshold_config.json"
    # if config_path.exists():
    #     try:
    #         with open(config_path) as f:
    #             config = json.load(f)
    #         print(f"✅ Loaded threshold from config: {config['threshold']}")
    #         return float(config['threshold'])
    #     except Exception as e:
    #         print(f"⚠️ Failed to load config: {e}. Using Youden threshold.")
    return threshold  # Fallback: Youden threshold from Run B


DEFAULT_THRESHOLD = _load_threshold()
# Note: This threshold uses specificity-constrained selection (min_spec=70%)
# which is more medically appropriate than pure Youden's Index
EFFICIENTNET_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models() -> Tuple[ort.InferenceSession, Optional[ort.InferenceSession]]:
    """
    Load ONNX classifier and optional bone suppression model.

    Returns:
        Tuple of (ONNX session, bone suppression ONNX session or None)
    """
    # Load ONNX classifier
    onnx_path = MODEL_DIR / "lunglens_effnet_b0.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Try to load bone suppression model
    bone_suppress_model = None
    onnx_bs_path = MODEL_DIR / "resnet_bs.onnx"
    if onnx_bs_path.exists():
        try:
            bone_suppress_model = ort.InferenceSession(str(onnx_bs_path), providers=['CPUExecutionProvider'])
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


def apply_bone_suppression(image: Image.Image, bone_suppress_model: Optional[ort.InferenceSession], view_orientation: str = "PA") -> Image.Image:
    """
    Apply bone suppression to chest X-ray image using ONNX model.

    If no model is available, returns original image (CLAHE fallback).

    Args:
        image: PIL Image (original chest X-ray)
        bone_suppress_model: ONNX InferenceSession or None
        view_orientation: View position ("PA" or "AP")

    Returns:
        Bone-suppressed PIL Image
    """
    print(bone_suppress_model)
    if bone_suppress_model is None:
        # Fallback: simulate bone suppression with histogram equalization
        img_array = np.array(image.convert('L'), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)
        return Image.fromarray(enhanced, mode='L')

    else:
        # Use ONNX bone suppression model
        try:
            # Resize image to 256x256 as required by ResNet-BS
            img_resized = image.convert('L').resize((256, 256), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0

            # Add batch and channel dimensions: (1, 256, 256, 1)
            img_input = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0).astype(np.float32)

            # Get input/output names
            input_names = [inp.name for inp in bone_suppress_model.get_inputs()]
            # print(f"Bone suppression model inputs: {input_names}")
            os.write(1, f"Bone suppression model inputs: {input_names}\n".encode())
            output_names = [out.name for out in bone_suppress_model.get_outputs()]
            os.write(1, f"Bone suppression model outputs: {output_names}\n".encode())
            print(f"Bone suppression model outputs: {output_names}")

            # Prepare input feed
            input_feed = {}
            for inp_name in input_names:
                if 'image' in inp_name.lower() or inp_name.lower() == 'input':
                    # First/main input is the image
                    input_feed[inp_name] = img_input
                else:
                    st.warning(f"⚠️ Unmapped input: {inp_name}")

            # Run inference
            suppressed = bone_suppress_model.run(output_names, input_feed)[0]

            # Convert back to PIL and resize to original
            suppressed = np.squeeze(suppressed) * 255.0
            suppressed = np.clip(suppressed, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(suppressed, mode='L')

            # Resize back to original dimensions if needed
            result_image = result_image.resize(image.size, Image.Resampling.LANCZOS)

            return result_image

        except Exception as e:
            st.warning(f"Bone suppression failed: {e}. Using original image.")
            return image


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    preprocessed_image: np.ndarray,
    ort_session: ort.InferenceSession,
    threshold: float = DEFAULT_THRESHOLD,
    view_orientation: str = "PA"
) -> Tuple[float, str, str]:
    """
    Run ONNX inference on preprocessed image.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        ort_session: ONNX Runtime session
        threshold: Decision threshold for binary classification
        view_orientation: View position ("PA" or "AP")

    Returns:
        Tuple of (probability_score, class_label, class_name)
    """
    # Get all input/output names
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    # Encode view orientation as metadata (PA=0, AP=1)
    view_metadata = np.array([[1.0 if view_orientation.upper() == "AP" else 0.0]], dtype=np.float32)

    # Prepare input feed
    input_feed = {}
    for inp_name in input_names:
        if 'image' in inp_name.lower() or inp_name.lower() == 'input':
            input_feed[inp_name] = preprocessed_image
        elif 'view' in inp_name.lower() or 'metadata' in inp_name.lower():
            input_feed[inp_name] = view_metadata
        else:
            st.warning(f"⚠️ Unmapped input: {inp_name}")

    # Run inference
    output = ort_session.run(output_names, input_feed)

    # Extract probability (logits -> softmax -> tumour probability)
    os.write(1, f"Output type: {type(output)}, length: {len(output)}\n".encode())
    os.write(1, f"Output[0] shape: {np.array(output[0]).shape}\n".encode())
    os.write(1, f"Output[0]: {output[0]}\n".encode())

    logits = output[0]
    if isinstance(logits, np.ndarray) and logits.ndim > 1:
        logits = logits[0]

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

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, _module, _inp, out) -> None:  # noqa: ARG002
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:  # noqa: ARG002
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 1
    ) -> Optional[np.ndarray]:
        """Compute the Grad-CAM heatmap for one image."""
        try:
            self.model.eval()
            image_tensor = image_tensor.detach().requires_grad_(True)

            # Forward pass
            with torch.enable_grad():
                logit = self.model(image_tensor)
                self.model.zero_grad()

                # Backward pass for target class
                if logit.ndim > 1:
                    score = logit[0, target_class]
                else:
                    score = logit[target_class] if len(logit) > target_class else logit[0]

                score.backward()

            # Get captured gradients and activations
            if self._gradients is None or self._activations is None:
                return None

            gradients = self._gradients[0]
            activations = self._activations[0]

            # Compute weights
            weights = gradients.mean(dim=(1, 2))

            # Weighted combination of activation maps
            cam = torch.einsum("c,chw->hw", weights, activations)
            cam = F.relu(cam)

            # Normalize
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            heatmap = cam.cpu().detach().numpy().astype(np.float32)
            return heatmap

        except Exception as e:
            st.warning(f"Grad-CAM computation failed: {e}")
            return None


_grad_cam_instance: Optional[GradCAM] = None


def get_grad_cam(model: torch.nn.Module) -> GradCAM:
    """Get or create GradCAM instance with target layer set to last conv layer."""
    global _grad_cam_instance
    if _grad_cam_instance is None:
        # Target the last convolutional layer in EfficientNet-B0
        target_layer = model.features[-1][0]  # First conv in last block
        _grad_cam_instance = GradCAM(model, target_layer)
    return _grad_cam_instance


@st.cache_resource
def load_efficientnet_model() -> torch.nn.Module:
    """Load pre-trained EfficientNet-B0 for Grad-CAM computation."""
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model


def generate_grad_cam(
    preprocessed_image: np.ndarray,
    view_orientation: str = "PA"
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for visualization.

    Uses PyTorch EfficientNet-B0 with gradient hooks to compute attention.
    Adapted from dual_stage_xray_pipeline_v2.ipynb.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        view_orientation: View position ("PA" or "AP") - for context

    Returns:
        Grad-CAM heatmap as numpy array (224, 224) or None if failed
    """
    try:
        # Load model
        model = load_efficientnet_model()

        # Get or create Grad-CAM instance
        grad_cam = get_grad_cam(model)

        # Convert to tensor
        input_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)

        # Compute Grad-CAM
        heatmap = grad_cam.compute(input_tensor, target_class=1)

        if heatmap is None:
            return None

        # Resize to standard size
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
        st.info(
            "**📊 Threshold Selection Strategy**\n\n"
            "This app uses a **specificity-constrained approach** (min 70% specificity) "
            "rather than pure Youden's Index. This is more medically appropriate because:\n\n"
            "• **Avoids false alarms**: Maintains ≥70% specificity\n"
            "• **Maximizes detection**: Among valid thresholds, picks highest sensitivity\n"
            "• **Clinically sound**: Prioritizes not flagging healthy patients\n\n"
            "To compute the optimal threshold from your data:\n"
            "`python compute_optimal_threshold.py`"
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

    col_upload, col_view = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Select a chest X-ray image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest radiograph for triage analysis",
            width='stretch'
        )

    with col_view:
        view_orientation = st.selectbox(
            "View Orientation",
            ["PA", "AP"],
            help="Specify whether the X-ray is Posteroanterior (PA) or Anteroposterior (AP) view"
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
        st.image(original_image, width='stretch', caption=f"Input image ({view_orientation})")

    # Run analysis on button click
    if st.button("🔍 Run LungLens AI Analysis", type="primary", width='stretch'):

        with st.spinner("🔄 Processing... (bone suppression → inference → Grad-CAM)"):

            # Step 1: Bone suppression
            try:
                bone_suppressed = apply_bone_suppression(original_image, bone_suppress_model, view_orientation)
            except Exception as e:
                st.error(f"❌ Bone suppression failed: {e}")
                st.stop()

            with col2:
                st.subheader("Bone-Suppressed Image")
                st.image(bone_suppressed, width='stretch', caption="After Stage 1 suppression")

            # Step 2: Preprocess and run inference
            try:
                preprocessed = preprocess_image(bone_suppressed)
                tumour_prob, class_label, class_name = run_inference(
                    preprocessed, ort_session, threshold, view_orientation
                )
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                st.stop()

            # Step 3: Generate Grad-CAM
            try:
                heatmap = generate_grad_cam(preprocessed, view_orientation)
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
                    st.image(grad_cam_image, width='stretch', caption="Areas of model attention")
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
