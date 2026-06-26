"""
LungLens AI - Chest X-ray Triage Prototype

A binary triage tool for tumour vs no tumour classification on chest radiographs
using a bone-suppressed EfficientNet-B0 classifier with Grad-CAM interpretability.

For research and demonstration only - NOT a diagnostic tool.
"""

import io
import json
import time
import warnings
from datetime import datetime
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

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION AND CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent

def _load_threshold() -> float:
    """Load optimal threshold from config or use default (Youden).

    Threshold computed using specificity-constrained selection (min_spec=70%)
    from training data. See threshold_utils.compute_optimal_threshold().
    """
    config_path = MODEL_DIR / "threshold_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            threshold = float(config.get('threshold', 0.5791015625))
            print(f"✅ Loaded threshold from config: {threshold}")
            return threshold
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Failed to load config ({type(e).__name__}): {e}. Using fallback.")

    fallback_threshold = 0.5791015625
    print(f"📌 Using fallback threshold: {fallback_threshold}")
    return fallback_threshold


DEFAULT_THRESHOLD = _load_threshold()
EFFICIENTNET_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
BONE_SUPPRESSION_INPUT_SIZE = 256


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models() -> Tuple[ort.InferenceSession, Optional[ort.InferenceSession]]:
    """Load ONNX classifier and optional bone suppression model.

    Returns:
        Tuple of (classifier ONNX session, bone suppression ONNX session or None)
    """
    onnx_path = MODEL_DIR / "lunglens_effnet_b0.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    bone_suppress_model = None
    onnx_bs_path = MODEL_DIR / "resnet_bs.onnx"
    if onnx_bs_path.exists():
        try:
            bone_suppress_model = ort.InferenceSession(str(onnx_bs_path), providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Warning: Could not load bone suppression model: {e}")

    return ort_session, bone_suppress_model


@st.cache_resource
def load_efficientnet_model() -> torch.nn.Module:
    """Load pre-trained EfficientNet-B0 for Grad-CAM computation."""
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, target_size: int = EFFICIENTNET_INPUT_SIZE) -> np.ndarray:
    """Preprocess PIL image for EfficientNet-B0 inference.

    Steps:
    1. Convert to grayscale if needed (X-rays are grayscale)
    2. Resize to target size
    3. Normalize using ImageNet statistics

    Args:
        image: PIL Image
        target_size: Target size for model input (default 224)

    Returns:
        Preprocessed image as numpy array (1, 3, 224, 224) for ONNX
    """
    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    img_array = np.array(image, dtype=np.float32) / 255.0

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# BONE SUPPRESSION
# ──────────────────────────────────────────────────────────────────────────────

def apply_bone_suppression(
    image: Image.Image,
    bone_suppress_model: Optional[ort.InferenceSession],
    view_orientation: str = "PA"
) -> Image.Image:
    """Apply bone suppression to chest X-ray image using ONNX model.

    If no model is available, applies CLAHE as fallback.

    Args:
        image: PIL Image (original chest X-ray)
        bone_suppress_model: ONNX InferenceSession or None
        view_orientation: View position ("PA" or "AP") - for context

    Returns:
        Bone-suppressed PIL Image
    """
    if bone_suppress_model is None:
        img_array = np.array(image.convert('L'), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)
        return Image.fromarray(enhanced, mode='L')

    try:
        img_resized = image.convert('L').resize((BONE_SUPPRESSION_INPUT_SIZE, BONE_SUPPRESSION_INPUT_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        img_input = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0).astype(np.float32)

        input_names = [inp.name for inp in bone_suppress_model.get_inputs()]
        output_names = [out.name for out in bone_suppress_model.get_outputs()]

        input_feed = {}
        for inp_name in input_names:
            if 'image' in inp_name.lower() or inp_name.lower() == 'input':
                input_feed[inp_name] = img_input

        suppressed = bone_suppress_model.run(output_names, input_feed)[0]

        suppressed = np.squeeze(suppressed) * 255.0
        suppressed = np.clip(suppressed, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(suppressed, mode='L')

        result_image = result_image.resize(image.size, Image.Resampling.LANCZOS)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result_array = np.array(result_image, dtype=np.uint8)
        enhanced = clahe.apply(result_array)
        result_image = Image.fromarray(enhanced, mode='L')

        return result_image

    except Exception as e:
        print(f"Bone suppression failed: {e}. Using CLAHE fallback.")
        img_array = np.array(image.convert('L'), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)
        return Image.fromarray(enhanced, mode='L')


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    preprocessed_image: np.ndarray,
    ort_session: ort.InferenceSession,
    threshold: float = DEFAULT_THRESHOLD,
    view_orientation: str = "PA"
) -> Tuple[float, str, str]:
    """Run ONNX inference on preprocessed image.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        ort_session: ONNX Runtime session
        threshold: Decision threshold for binary classification
        view_orientation: View position ("PA" or "AP")

    Returns:
        Tuple of (probability_score, class_label, class_name)
    """
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    view_metadata = np.array([[1.0 if view_orientation.upper() == "AP" else 0.0]], dtype=np.float32)

    input_feed = {'image': preprocessed_image, 'view_metadata': view_metadata}

    output = ort_session.run(output_names, input_feed)

    logits = output[0]
    if isinstance(logits, np.ndarray) and logits.ndim > 1:
        logits = logits[0]

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    if logits_tensor.ndim == 0:
        tumour_prob = float(torch.sigmoid(logits_tensor))
    else:
        tumour_prob = float(torch.sigmoid(logits_tensor[0]))

    if tumour_prob >= threshold:
        class_label = "tumour_positive"
        class_name = "HIGH SUSPICION"
    else:
        class_label = "tumour_negative"
        class_name = "LOW SUSPICION"

    return tumour_prob, class_label, class_name


# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────────────────────

def generate_grad_cam(
    preprocessed_image: np.ndarray,
    view_orientation: str = "PA"
) -> Optional[np.ndarray]:
    """Generate Grad-CAM heatmap for visualization.

    Uses PyTorch EfficientNet-B0 with gradient hooks to compute attention.
    Hooks are registered and removed within this function to prevent accumulation.

    Args:
        preprocessed_image: Preprocessed image array (1, 3, 224, 224)
        view_orientation: View position ("PA" or "AP") - for context

    Returns:
        Grad-CAM heatmap as numpy array (224, 224) or None if failed
    """
    try:
        model = load_efficientnet_model()
        model.eval()

        target_layer = model.features[-1][0]

        activations = None
        gradients = None

        def save_activation(_module, _inp, out):
            nonlocal activations
            activations = out.detach()

        def save_gradient(_module, _grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        hook1 = target_layer.register_forward_hook(save_activation)
        hook2 = target_layer.register_full_backward_hook(save_gradient)

        try:
            input_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)
            input_tensor = input_tensor.detach().requires_grad_(True)

            with torch.enable_grad():
                logit = model(input_tensor)
                model.zero_grad()

                if logit.ndim > 1:
                    score = logit[0, 1]
                else:
                    score = logit[1] if len(logit) > 1 else logit[0]

                score.backward()

            if gradients is None or activations is None:
                return None

            grad_vals = gradients[0]
            act_vals = activations[0]

            weights = grad_vals.mean(dim=(1, 2))

            cam = torch.einsum("c,chw->hw", weights, act_vals)
            cam = F.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            heatmap = cam.cpu().detach().numpy().astype(np.float32)

            heatmap_resized = cv2.resize(heatmap, (EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE))

            return heatmap_resized

        finally:
            hook1.remove()
            hook2.remove()

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None


def overlay_grad_cam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> Image.Image:
    """Overlay Grad-CAM heatmap on bone-suppressed image.

    Args:
        original_image: PIL Image of the bone-suppressed X-ray
        heatmap: Grad-CAM heatmap (normalized to 0-1)
        alpha: Blending factor for overlay

    Returns:
        PIL Image with Grad-CAM overlay
    """
    original_image = original_image.resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE), Image.Resampling.LANCZOS)

    img_array = np.array(original_image.convert('L'))
    img_rgb = np.stack([img_array] * 3, axis=-1)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(overlay.astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# TRIAGE OUTPUT FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

def format_triage_note(
    tumour_prob: float,
    class_name: str,
    threshold: float,
    uploaded_filename: str,
    view_orientation: str,
    bone_suppress_available: bool
) -> Tuple[str, str]:
    """Generate structured triage report.

    Args:
        tumour_prob: Predicted probability (0-1)
        class_name: Classification label
        threshold: Decision threshold used
        uploaded_filename: Name of uploaded image file
        view_orientation: View position (PA or AP)
        bone_suppress_available: Whether bone suppression model was available

    Returns:
        Tuple of (plain_text_note, markdown_note)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    is_high_risk = tumour_prob >= threshold

    plain_text_note = f"""--- LUNGLENS AI TRIAGE REPORT ---
Date/Time: {timestamp}
Image file: {uploaded_filename}
View orientation: {view_orientation}

PIPELINE:
- Stage 1: Bone Suppression — {"ONNX model active" if bone_suppress_available else "CLAHE fallback"}
- Stage 2: EfficientNet-B0 classifier (ONNX, Run B)
- Threshold: {threshold:.4f} (Youden-J, min specificity 70%)

RESULT:
- AI Risk Score: {tumour_prob:.4f}
- Classification: {class_name}
- No-Tumour Probability: {1 - tumour_prob:.4f}

SUGGESTED TRIAGE ACTION:
"""

    if is_high_risk:
        plain_text_note += """Prioritise for radiologist review within same session.
Consider CT referral if clinically appropriate and available.
Document as flagged by AI triage tool for radiologist follow-up."""
    else:
        plain_text_note += """Routine radiologist review at next available appointment.
AI screening does not exclude tumour — clinical judgment required.
Standard follow-up protocol."""

    plain_text_note += """

DISCLAIMER:
This output was generated by LungLens AI, a research prototype. It is not a medical
device, has not been clinically validated in Cameroon, and must not substitute for
radiologist review. University of Bamenda, Department of Computer Engineering, 2026.
"""

    markdown_note = f"""## 📋 LungLens AI Triage Report

**Date/Time:** {timestamp}
**Image:** {uploaded_filename}
**View:** {view_orientation}

### Pipeline
- **Stage 1:** Bone Suppression — {"ONNX model active" if bone_suppress_available else "CLAHE fallback"}
- **Stage 2:** EfficientNet-B0 classifier (ONNX, Run B)
- **Threshold:** {threshold:.4f} (Youden-J, min specificity 70%)

### Results
| Metric | Value |
|--------|-------|
| **AI Risk Score** | {tumour_prob:.4f} |
| **Classification** | {class_name} |
| **No-Tumour Probability** | {1 - tumour_prob:.4f} |

### Suggested Triage Action
"""

    if is_high_risk:
        markdown_note += """⚠️ **HIGH RISK**

Prioritise for radiologist review within same session. Consider CT referral if clinically appropriate and available. Document as flagged by AI triage tool for radiologist follow-up."""
    else:
        markdown_note += """✓ **LOW RISK**

Routine radiologist review at next available appointment. AI screening does not exclude tumour — clinical judgment required. Standard follow-up protocol."""

    markdown_note += """

---
*This output was generated by LungLens AI, a research prototype. It is not a medical device, has not been clinically validated in Cameroon, and must not substitute for radiologist review. University of Bamenda, Department of Computer Engineering, 2026.*
"""

    return plain_text_note, markdown_note


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LungLens AI – Lung Tumour Triage",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header Banner
    st.markdown("""
    # 🫁 LungLens AI
    ## AI-Assisted Chest X-ray Triage for Lung Tumour Screening
    *Designed for low-resource regional hospitals where CT access is limited — EfficientNet-B0 classifier on bone-suppressed radiographs.*
    """)

    st.error("""
    ⚠️ **RESEARCH PROTOTYPE ONLY — Not a medical device. Not approved for clinical use.
    All outputs must be reviewed by a qualified radiologist before any clinical decision is made.**
    """)

    # Load models early
    bone_suppress_available = False
    try:
        ort_session, bone_suppress_model = load_models()
        bone_suppress_available = bone_suppress_model is not None
    except FileNotFoundError as e:
        st.error(f"❌ Classifier not found — cannot run. {e}")
        st.stop()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Model Status & Controls")

        st.subheader("📊 Loaded Models")
        st.success("✅ ONNX Classifier loaded (18.17 MB)")
        if bone_suppress_available:
            st.success("✅ Bone Suppression model loaded (ONNX)")
        else:
            st.warning("⚠️ Bone Suppression: CLAHE fallback active")

        st.markdown("---")

        st.subheader("📐 View Orientation")
        view_orientation = st.selectbox(
            "Select view type:",
            ["PA (Posteroanterior)", "AP (Anteroposterior)"],
            help="PA is the standard view. AP is common in bedridden patients. Note: this model was trained on PA views; AP predictions may be less reliable."
        )
        view_orientation = view_orientation.split()[0]

        st.markdown("---")

        st.subheader("⚙️ Decision Threshold")
        threshold = st.slider(
            "Probability threshold for tumour classification",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            help="Probability threshold for tumour classification. Higher = stricter (fewer tumours detected)."
        )

        if threshold < DEFAULT_THRESHOLD:
            st.caption("📉 Lower threshold → higher sensitivity, more false positives. More patients flagged for review.")
        elif threshold > DEFAULT_THRESHOLD:
            st.caption("📈 Higher threshold → higher specificity, fewer false positives. Risk of missing true tumours.")
        else:
            st.caption("✓ Optimal threshold (Youden-J, min specificity 70%)")

        st.markdown("---")

        with st.expander("📊 Validated Performance (Run B, Test Set)"):
            st.markdown("""
| Metric       | Value  |
|--------------|--------|
| AUROC        | 0.7334 |
| Sensitivity  | 0.696  |
| Specificity  | 0.635  |
| NPV          | 0.9092 |
| PPV          | 0.2841 |
| Threshold    | 0.5791 |

*Evaluated on NIH ChestX-ray14, patient-stratified split (n=17,512 test images).*

**Note:** These metrics reflect benchmark data. Real-world performance in Cameroonian hospitals may differ due to domain shift.
            """)

        with st.expander("ℹ️ About the Pipeline"):
            st.markdown("""
**Stage 1: Bone Suppression**
- ResNet-BS pretrained on JSRT/BSE_JSRT dataset
- Attenuates rib and clavicular shadows to enhance soft-tissue visibility

**Stage 2: EfficientNet-B0 Classification**
- Binary classifier trained on NIH ChestX-ray14 (Mass + Nodule = positive class)
- ONNX export, 18.17 MB
- Metadata fusion: PA/AP view flag fused into classifier head

**Grad-CAM Interpretability**
- Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
- Overlaid on bone-suppressed input to show model focus areas
            """)

        st.markdown("---")

        st.error("""
⚠️ **NOT A MEDICAL DEVICE**

Research prototype only. University of Bamenda, 2026.
Do not use for clinical decisions.
        """)

    # Load models (already done in sidebar, but keep reference)
    # ort_session and bone_suppress_model are available from sidebar block

    # Upload Section
    st.header("📤 Upload Chest X-ray")

    col_upload, col_view_info = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Select a chest X-ray image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest radiograph for triage analysis"
        )

    if uploaded_file is not None:
        st.caption(f"📄 File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    if uploaded_file is None:
        st.info("👆 Please upload a chest X-ray image to begin analysis.")
        st.stop()

    # Load and display original image
    try:
        original_image = Image.open(io.BytesIO(uploaded_file.getbuffer()))
        img_width, img_height = original_image.size
        st.caption(f"Image dimensions: {img_width}×{img_height} pixels")
        if original_image.mode != 'L':
            st.info("ℹ️ This image appears to be in colour. It will be converted to grayscale for analysis, as the model was trained on grayscale chest radiographs.")
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        st.stop()

    # Pipeline Visualization
    st.header("🔬 Pipeline Visualization")

    col1, _, _ = st.columns(3)

    with col1:
        st.subheader("Original X-ray")
        st.image(original_image, use_column_width=True)
        st.caption(f"Input image ({view_orientation})\n\nRaw chest radiograph as uploaded.")

    # Initialize result variables
    bone_suppressed: Optional[Image.Image] = None
    preprocessed: Optional[np.ndarray] = None
    tumour_prob: float = 0.0
    class_name: str = ""
    grad_cam_image: Optional[Image.Image] = None
    inference_time_ms: float = 0.0

    # Run analysis on button click
    if st.button("🔍 Run LungLens AI Analysis", type="primary", use_container_width=True):

        progress_container = st.container()

        # Step 1: Bone suppression
        progress_container.info("⏳ Stage 1: Applying bone suppression...")
        try:
            bone_suppressed = apply_bone_suppression(original_image, bone_suppress_model, view_orientation)
        except Exception as e:
            st.error(f"❌ Bone suppression failed: {e}")
            st.stop()

        # Step 2: Preprocess and run inference
        progress_container.info("⏳ Stage 2: Running EfficientNet-B0 classifier...")
        inference_start = time.time()
        try:
            preprocessed = preprocess_image(bone_suppressed)
            tumour_prob, _, class_name = run_inference(
                preprocessed, ort_session, threshold, view_orientation
            )
            inference_time_ms = (time.time() - inference_start) * 1000
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

        # Step 3: Generate Grad-CAM for all images
        progress_container.info("⏳ Generating Grad-CAM interpretability map...")
        try:
            if preprocessed is not None and bone_suppressed is not None:
                heatmap = generate_grad_cam(preprocessed, view_orientation)
                if heatmap is not None:
                    grad_cam_image = overlay_grad_cam(bone_suppressed, heatmap, alpha=0.4)
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")

        # Display results in pipeline columns
        col2_res, col3_res = st.columns(2)

        with col2_res:
            st.subheader("Bone-Suppressed Image")
            st.image(bone_suppressed, use_column_width=True)
            st.caption("After Stage 1 suppression\n\nBone shadows attenuated; soft tissues enhanced.")

        with col3_res:
            st.subheader("Grad-CAM (Model Focus)")
            if grad_cam_image is not None:
                st.image(grad_cam_image, use_column_width=True)
                st.caption("Areas of model attention\n\nRed/yellow = high model focus; blue = low focus.")
            else:
                st.warning("⚠️ Grad-CAM visualization unavailable for this image.")

        progress_container.success("✅ Analysis complete!")

        # Risk Score and Classification Panel
        st.header("⚠️ Risk Assessment")

        is_high_risk = tumour_prob >= threshold

        if is_high_risk:
            st.markdown("""
<div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 8px; text-align: center;">
<h2>🔴 HIGH TUMOUR SUSPICION</h2>
</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background-color: #44aa44; color: white; padding: 20px; border-radius: 8px; text-align: center;">
<h2>🟢 LOW TUMOUR SUSPICION</h2>
</div>
            """, unsafe_allow_html=True)

        col_risk1, col_risk2, col_risk3 = st.columns(3)
        with col_risk1:
            st.metric("AI Risk Score", f"{tumour_prob:.4f}", help="Predicted probability of tumour (0-1)")
        with col_risk2:
            st.metric("Threshold Applied", f"{threshold:.4f}", help="Decision boundary for classification")
        with col_risk3:
            st.metric("No-Tumour Probability", f"{1 - tumour_prob:.4f}", help="Complement probability (1 - risk score)")

        st.markdown("### Risk Gauge")
        st.progress(tumour_prob, text=f"Risk: {tumour_prob*100:.1f}% | Threshold: {threshold*100:.1f}%")

        st.info(f"""
**NPV Context (Important for triage):**

At the validated threshold, this model has an NPV of 90.9%, meaning roughly 9 in 10 low-suspicion results are true negatives. However, clinical judgment must always be applied.
        """)

        # Structured Triage Output
        st.header("📋 Triage Report")

        plain_note, markdown_note = format_triage_note(
            tumour_prob, class_name, threshold, uploaded_file.name, view_orientation, bone_suppress_available
        )

        expander_label = "📋 Copy Triage Report (HIGH RISK)" if is_high_risk else "📋 Copy Triage Report"
        with st.expander(expander_label, expanded=is_high_risk):
            st.markdown(markdown_note)
            st.code(plain_note, language=None)
            st.button("📋 Copy to Clipboard", help="Copy plain text version to clipboard")

        # Detailed Metrics
        with st.expander("📊 Pipeline Details"):
            st.markdown("### Inference Details")
            st.markdown(f"""
| Parameter              | Value                         |
|------------------------|-------------------------------|
| Model                  | EfficientNet-B0 (ONNX)        |
| ONNX opset             | 17                            |
| ONNX file size         | 18.17 MB                      |
| Input size             | 224 × 224 px (3 channels)     |
| Bone suppression model | ResNet-BS (JSRT/BSE_JSRT)     |
| Stage 1 input size     | 256 × 256 px (1 channel)      |
| Threshold              | {threshold:.6f}               |
| View orientation       | {view_orientation}            |
| Inference time         | {inference_time_ms:.1f} ms    |
            """)

            st.markdown("### Subgroup Context")
            st.info("""
At threshold 0.5791 (Run B), subgroup-stratified specificity:
- vs No Finding (normal lungs): 0.651
- vs Other Pathology (non-tumour disease): 0.632

This gap indicates the model may produce more false positives for patients with non-tumour lung disease (e.g., TB, effusion) than for patients with normal lungs.
            """)

            st.markdown("### Training Data Reference")
            st.caption("""
Trained on NIH ChestX-ray14 (112,120 images, 30,805 patients). Positive class: Mass + Nodule labels (n=11,175). Patient-stratified split; zero patient overlap across train/validation/test.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
LungLens AI v1.0 — University of Bamenda, 2026 | Research Prototype | <a href="https://github.com/yll0rd/lunglens-ai">GitHub Repository</a>
</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
