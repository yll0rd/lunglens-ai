# LungLens AI – Chest X-ray Triage Prototype

A Streamlit-based binary triage tool for "tumour vs no tumour" classification on chest radiographs, designed for low-resource healthcare settings.

## Features

✅ **Two-Stage Pipeline**
- **Stage 1:** Bone suppression (ResNet-based enhancement of soft tissue)
- **Stage 2:** Binary classification with EfficientNet-B0 (ONNX-exported)

✅ **Interpretability**
- Grad-CAM visualization showing where the model focuses
- Visual overlay on bone-suppressed images

✅ **Structured Triage Output**
- AI risk score (0–1 probability)
- Binary classification (High/Low tumour suspicion)
- Actionable recommendations for radiologists

✅ **Configurable Decision Threshold**
- Slider to adjust sensitivity/specificity trade-off
- Default: Youden-J threshold (0.579) from validation data

✅ **Medical Safety**
- Clear disclaimer that this is for research/demonstration only
- Not a diagnostic device
- Emphasizes radiologist review requirement

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. **Navigate to the streamlit_app directory:**
   ```bash
   cd streamlit_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

### Workflow

1. **Upload** a chest X-ray image (JPG or PNG)
2. **View** the original radiograph
3. **Click** "Run LungLens AI Analysis"
4. **Review**:
   - Bone-suppressed version (Stage 1)
   - Grad-CAM heatmap (model focus areas)
   - AI risk score and classification
   - Triage recommendations

### Configuration

Use the sidebar to:
- Adjust the decision threshold (0–1 slider)
- Read the medical disclaimer
- Understand threshold sensitivity

## Model Architecture

| Component | Details |
|-----------|---------|
| **Input** | Chest X-ray (any size, converted to grayscale) |
| **Stage 1** | Bone suppression (ResNet, `resnet_bs.h5`) |
| **Preprocessing** | Resize to 224×224, ImageNet normalization |
| **Stage 2** | EfficientNet-B0 classifier (ONNX, `lunglens_effnet_b0.onnx`) |
| **Output** | Probability score (0–1) → Binary class (threshold-dependent) |

## Model Paths

Ensure the following files exist in your project structure:

```
project/
├── nih_outputs/
│   └── mobile_export/
│       └── lunglens_effnet_b0.onnx        ← Stage 2 classifier
├── resnet_bs.h5                          ← Stage 1 (optional)
└── streamlit_app/
    ├── app.py
    ├── requirements.txt
    └── README.md
```

If the bone suppression model is unavailable, the app will use CLAHE (Contrast Limited Adaptive Histogram Equalization) as a placeholder.

## Validation Metrics (Run B)

- **AUROC:** 0.7394 (validation), 0.7334 (test)
- **Sensitivity:** 0.646 (validation), 0.696 (test)
- **Specificity:** 0.701 (validation), 0.635 (test)
- **Youden Threshold:** 0.5791
- **Backbone:** EfficientNet-B0
- **Input:** Bone-suppressed chest X-rays

## Disclaimers

⚠️ **This prototype is for research and demonstration only.**
- Not a medical device
- Not FDA/CE approved
- Must not be used as a standalone diagnostic tool
- All predictions require radiologist review
- For low-resource healthcare settings only (demonstration)

## File Structure

```
app.py
├── load_models()           Load ONNX classifier and bone suppression model
├── preprocess_image()      Resize, normalize for EfficientNet-B0
├── apply_bone_suppression() Apply ResNet bone suppression or CLAHE placeholder
├── run_inference()         ONNX inference with threshold-based classification
├── generate_grad_cam()     Grad-CAM heatmap for interpretability
├── overlay_grad_cam()      Blend heatmap with original image
├── format_triage_note()    Generate structured triage output
└── main()                  Streamlit UI
```

## Performance Notes

- **Processing time:** ~2–5 seconds (depending on hardware and bone suppression model)
- **Memory:** ~1–2 GB (model weights + inference)
- **GPU support:** Not implemented (CPU-based ONNX runtime)

## Future Enhancements

- GPU inference support via CUDA/TensorRT
- Multi-class classification (tumour subtypes)
- Batch processing for multiple images
- Integration with PACS systems
- Calibration for different X-ray equipment
- Uncertainty quantification (Bayesian approaches)

## License

Research and demonstration only. Use at your own risk.
