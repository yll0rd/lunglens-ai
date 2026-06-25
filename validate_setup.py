"""
Validation script for LungLens AI Streamlit app setup.

Checks that all required models, dependencies, and configuration are in place.
Run this before launching the app to ensure everything is working.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        msg = f"[OK] Python {version.major}.{version.minor}.{version.micro}"
        return True, msg
    else:
        msg = f"[FAIL] Python {version.major}.{version.minor} (requires 3.8+)"
        return False, msg


def check_model_files() -> Tuple[bool, List[str]]:
    """Check that all required model files exist."""
    messages = []
    project_dir = Path(__file__).parent.parent

    # Check ONNX model
    onnx_path = project_dir / "nih_outputs" / "mobile_export" / "lunglens_effnet_b0.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        messages.append(f"[OK] ONNX model: {onnx_path.name} ({size_mb:.1f} MB)")
        onnx_ok = True
    else:
        messages.append(f"[FAIL] ONNX model not found: {onnx_path}")
        onnx_ok = False

    # Check bone suppression model
    bs_path = project_dir / "resnet_bs.h5"
    if bs_path.exists():
        size_mb = bs_path.stat().st_size / (1024 * 1024)
        messages.append(f"[OK] Bone suppression model: {bs_path.name} ({size_mb:.1f} MB)")
        bs_ok = True
    else:
        messages.append(f"[WARN] Bone suppression model not found: {bs_path} (will use CLAHE fallback)")
        bs_ok = True  # Fallback is available, so not critical

    return onnx_ok, messages


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check that all required Python packages are installed."""
    messages = []
    dependencies = {
        'streamlit': 'Streamlit framework',
        'numpy': 'Numerical computing',
        'PIL': 'Image processing (Pillow)',
        'cv2': 'OpenCV for image enhancement',
        'onnxruntime': 'ONNX model inference',
        'torch': 'PyTorch (for Grad-CAM)',
        'torchvision': 'PyTorch vision utilities',
    }

    all_ok = True
    for module, description in dependencies.items():
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            messages.append(f"[OK] {module:<15} v{version:<10} - {description}")
        except ImportError:
            messages.append(f"[FAIL] {module:<15} - {description} (not installed)")
            all_ok = False

    return all_ok, messages


def check_configuration() -> Tuple[bool, List[str]]:
    """Check configuration file exists and is valid."""
    messages = []
    config_path = Path(__file__).parent / "config.py"

    if config_path.exists():
        try:
            # Try to import config
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)

            messages.append(f"[OK] Configuration file: {config_path.name}")
            messages.append(f"  - ONNX model path: {config.ONNX_MODEL_PATH}")
            messages.append(f"  - Default threshold: {config.DEFAULT_THRESHOLD}")
            messages.append(f"  - Input size: {config.EFFICIENTNET_INPUT_SIZE}x{config.EFFICIENTNET_INPUT_SIZE}")
            return True, messages
        except Exception as e:
            messages.append(f"[FAIL] Configuration file error: {e}")
            return False, messages
    else:
        messages.append(f"[FAIL] Configuration file not found: {config_path}")
        return False, messages


def check_app_file() -> Tuple[bool, List[str]]:
    """Check that app.py exists and has valid syntax."""
    messages = []
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        messages.append(f"[FAIL] app.py not found: {app_path}")
        return False, messages

    try:
        import py_compile
        py_compile.compile(str(app_path), doraise=True)
        messages.append(f"[OK] app.py exists and syntax is valid")
        return True, messages
    except py_compile.PyCompileError as e:
        messages.append(f"[FAIL] app.py syntax error: {e}")
        return False, messages


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("LungLens AI – Setup Validation")
    print("=" * 70 + "\n")

    all_passed = True
    checks = [
        ("Python Version", check_python_version),
        ("Model Files", check_model_files),
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
        ("App File", check_app_file),
    ]

    for check_name, check_func in checks:
        print(f"\n[*] {check_name}:")
        print("-" * 70)

        try:
            if check_name == "Python Version":
                passed, msg = check_func()
                print(msg)
                all_passed = all_passed and passed
            else:
                passed, messages = check_func()
                for msg in messages:
                    print(msg)
                all_passed = all_passed and passed
        except Exception as e:
            print(f"[X] Error during check: {e}")
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] All checks passed! Ready to run Streamlit app.")
        print("\nTo start the app, run:")
        print("  streamlit run app.py")
    else:
        print("[WARN] Some checks failed. See details above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Verify model files exist in project directory")
        print("  3. Check Python version (requires 3.8+)")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
