"""
Face Recognition API - Lightweight Edition
==========================================
Using OpenCV + SFace (No TensorFlow!)
RAM: ~150MB | Speed: ~100-200ms per request
"""

import os

# Ensure models are downloaded
from download_models import download_models
download_models()

# Ensure data directories exist
os.makedirs('data', exist_ok=True)

from app import create_app
from app.extensions import logger

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Face Recognition API (OpenCV + SFace) on port {port}...")
    logger.info("Lightweight mode: ~150MB RAM, ~100ms per request")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
