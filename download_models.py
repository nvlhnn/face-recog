"""
Download Models
===============
Downloads required model files based on the configured engine.

- OpenCV: Downloads YuNet + SFace ONNX models
- InsightFace: Models are auto-downloaded by the library (ArcFace)
"""

import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

OPENCV_MODELS = {
    # YuNet Face Detector (~230KB)
    "face_detection_yunet_2023mar.onnx": 
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    
    # SFace Face Recognizer (~37MB)
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def download_models(all_engines: bool = False):
    """Download required models based on FACE_ENGINE env var or all if requested."""
    engine = os.getenv("FACE_ENGINE", "opencv").lower().strip()
    
    if all_engines or engine == "opencv":
        _download_opencv_models()
    
    if all_engines or engine == "insightface":
        _download_insightface_models()
    
    if not all_engines and engine not in ["opencv", "insightface"]:
        print(f"⚠️  Unknown engine: {engine}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def _download_insightface_models():
    """Trigger InsightFace model download by initializing the app."""
    model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
    model_root = os.path.join(MODELS_DIR, 'insightface')
    print(f"⬇ Pre-downloading InsightFace models (pack: {model_name}) into {model_root}...")
    try:
        from insightface.app import FaceAnalysis
        # Use root parameter to force download to /app/models/insightface
        app = FaceAnalysis(name=model_name, root=model_root, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print(f"✓ InsightFace models ({model_name}) are ready")
    except Exception as e:
        print(f"✗ Failed to pre-download InsightFace models: {e}")
        # Only failing on production builds is critical
        if os.getenv("ENV", "staging").lower() == "production":
            raise


def _download_opencv_models():
    """Download OpenCV YuNet + SFace models if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for filename, url in OPENCV_MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            continue
        
        print(f"⬇ Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            raise

    print("✅ OpenCV models ready!")


if __name__ == "__main__":
    # Check if --all flag is passed
    download_all = "--all" in sys.argv
    download_models(all_engines=download_all)
