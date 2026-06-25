"""
Configuration file for LungLens AI Streamlit app.

This module centralizes all configurable parameters used by the application.
"""

from pathlib import Path
from typing import Tuple

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (assuming app is in streamlit_app/ folder)
APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent

# Model paths
MODEL_DIR = PROJECT_DIR / "nih_outputs" / "mobile_export"
ONNX_MODEL_PATH = MODEL_DIR / "lunglens_effnet_b0.onnx"
BONE_SUPPRESSION_MODEL_PATH = PROJECT_DIR / "resnet_bs.h5"

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# EfficientNet-B0 configuration
EFFICIENTNET_INPUT_SIZE = 224  # Input resolution for EfficientNet-B0
EFFICIENTNET_CHANNELS = 3      # RGB (grayscale replicated to 3 channels)

# ImageNet normalization statistics (used for preprocessing)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# =============================================================================
# CLASSIFICATION THRESHOLDS
# =============================================================================

# Default threshold (Youden-J index from validation data of Run B)
# This maximizes sensitivity + specificity - 1
DEFAULT_THRESHOLD = 0.5791015625

# Threshold range for slider in UI
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.01

# =============================================================================
# MODEL PERFORMANCE METRICS (Run B - Bone-Suppressed EfficientNet-B0)
# =============================================================================

# Validation metrics
VAL_AUROC = 0.7394
VAL_SENSITIVITY = 0.6456  # Recall for tumour-positive cases
VAL_SPECIFICITY = 0.7014  # Recall for tumour-negative cases
VAL_THRESHOLD = 0.5791015625

# Test metrics
TEST_AUROC = 0.7334
TEST_SENSITIVITY = 0.6960
TEST_SPECIFICITY = 0.6346
TEST_THRESHOLD = 0.5791015625

# Sample sizes
VAL_TUMOUR_POSITIVE = 1219
VAL_TUMOUR_NEGATIVE = 5686
TEST_TUMOUR_POSITIVE = 3020
TEST_TUMOUR_NEGATIVE = 14492

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Streamlit page config
STREAMLIT_PAGE_TITLE = "LungLens AI"
STREAMLIT_PAGE_ICON = "🫁"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

# Grad-CAM configuration
GRADCAM_ALPHA = 0.5  # Blending factor for heatmap overlay (0-1)
GRADCAM_COLORMAP = "jet"  # Colormap for heatmap visualization

# Bone suppression (placeholder) configuration
CLAHE_CLIP_LIMIT = 2.0  # Contrast limiting threshold for CLAHE
CLAHE_TILE_GRID_SIZE = (8, 8)  # Tile grid size for CLAHE

# =============================================================================
# CLASSIFICATION LABELS
# =============================================================================

CLASS_LABELS = {
    "tumour_positive": {
        "name": "High tumour suspicion",
        "emoji": "🔴",
        "risk_level": "HIGH",
        "recommendation": (
            "⚠️ **Suggested action:** Prioritise for radiologist review. "
            "Consider CT referral if clinically appropriate."
        )
    },
    "tumour_negative": {
        "name": "Low tumour suspicion",
        "emoji": "🟢",
        "risk_level": "LOW",
        "recommendation": (
            "✓ **Suggested action:** Routine radiologist review. "
            "Standard follow-up protocol."
        )
    }
}

# =============================================================================
# MEDICAL DISCLAIMER
# =============================================================================

MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer**\n\n"
    "This prototype is for **research and demonstration only**. "
    "It is **not a medical device** and **must not be used** as a standalone diagnostic tool. "
    "All predictions must be reviewed by qualified radiologists."
)

# =============================================================================
# SUPPORTED FILE FORMATS
# =============================================================================

SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]

# =============================================================================
# ONNX RUNTIME CONFIGURATION
# =============================================================================

# Execution providers for ONNX Runtime (in order of preference)
ONNX_EXECUTION_PROVIDERS = ['CPUExecutionProvider']

# Option to use GPU if available (uncomment if CUDA is installed)
# ONNX_EXECUTION_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Enable debug logging
DEBUG_MODE = False

# Suppress warnings
SUPPRESS_WARNINGS = True
