"""
Download OpenCV Face Models
===========================
Downloads YuNet (face detection) and SFace (face recognition) models.
Run this once before starting the server.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

MODELS = {
    # YuNet Face Detector (~230KB)
    "face_detection_yunet_2023mar.onnx": 
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    
    # SFace Face Recognizer (~37MB)
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def download_models():
    """Download required models if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for filename, url in MODELS.items():
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

    print("\n✅ All models ready!")
    return MODELS_DIR

if __name__ == "__main__":
    download_models()
