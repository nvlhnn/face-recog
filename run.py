"""
Face Recognition API
====================
Multi-engine face recognition service.
Supports: OpenCV (SFace), InsightFace (ArcFace)
Configure via FACE_ENGINE environment variable.
"""

import os

# Ensure models are downloaded (engine-aware)
from download_models import download_models
download_models()

# Ensure data directories exist
os.makedirs('data', exist_ok=True)

from app import create_app
from app.extensions import logger

app = create_app()

if __name__ == '__main__':
    engine = os.getenv("FACE_ENGINE", "opencv")
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Face Recognition API (engine: {engine}) on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
