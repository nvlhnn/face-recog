"""
Image Utilities
===============
High-level face processing utilities that delegate to the configured engine.
Engine selection is controlled via the FACE_ENGINE environment variable.

Supported engines: opencv, insightface
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple

from app.extensions import logger
from app.engines import get_engine

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_image_file(file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image file."""
    if not file:
        return False, "No image file provided"

    filename = file.filename.lower()
    if '.' not in filename:
        return False, "Invalid file format"

    ext = filename.rsplit('.', 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '{ext}' not allowed"

    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        return False, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB"

    return True, None


def _is_anti_spoofing_enabled() -> bool:
    """Check if anti-spoofing is enabled via env var."""
    return os.getenv('ANTI_SPOOFING', 'false').lower().strip() in ('true', '1', 'yes')


class ImageProcessor:
    """
    Face processing utilities that delegate to the configured engine.
    
    This class provides a stable API regardless of which engine is active.
    All methods are static for backward compatibility.
    """

    @staticmethod
    def _decode_image(image_file) -> Optional[np.ndarray]:
        """Decode uploaded file to OpenCV BGR image."""
        image_file.seek(0)
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def get_face_encoding(image_file, check_liveness: bool = None) -> Optional[np.ndarray]:
        """
        Extract face encoding from an uploaded image file.
        
        If anti-spoofing is enabled (ANTI_SPOOFING=true or check_liveness=True),
        a liveness check is performed before encoding extraction.
        Raises ValueError if the image fails the spoof check.
        
        Args:
            image_file: Uploaded file object
            check_liveness: Override for anti-spoofing. None = use env var.
        
        Returns:
            1-D numpy array (face embedding), or None if no face found.
            
        Raises:
            ValueError: If anti-spoofing check fails (spoof detected).
        """
        try:
            img = ImageProcessor._decode_image(image_file)
            if img is None:
                logger.warning("Could not decode image")
                return None

            engine = get_engine()
            detection = engine.detect_face(img)

            if not detection.face_found:
                logger.warning("No face detected in image")
                return None

            # Anti-spoofing check
            should_check = check_liveness if check_liveness is not None else _is_anti_spoofing_enabled()
            if should_check:
                liveness = engine.check_liveness(img, detection)
                if not liveness.is_live:
                    msg = liveness.message or "Spoof detected"
                    logger.warning(f"Anti-spoof check failed: {msg} (score: {liveness.score:.3f})")
                    raise ValueError(f"Liveness check failed: {msg}")
                logger.info(f"Anti-spoof check passed (score: {liveness.score:.3f})")

            encoding = engine.extract_encoding(img, detection)
            return encoding

        except ValueError:
            raise  # Re-raise spoof detection errors
        except Exception as e:
            logger.error(f"Error extracting encoding: {e}")
            raise

    @staticmethod
    def compare_encodings(
        known_encoding: np.ndarray,
        unknown_encoding: np.ndarray,
        tolerance: float = None
    ) -> Tuple[bool, float, float]:
        """
        Compare two face encodings.
        
        Args:
            known_encoding: Stored face encoding
            unknown_encoding: New face encoding to compare
            tolerance: Match threshold (uses engine default if None)
        
        Returns:
            (is_match, distance, confidence_percent)
        """
        engine = get_engine()

        if tolerance is None:
            tolerance = engine.default_threshold()

        result = engine.compare_encodings(known_encoding, unknown_encoding, tolerance)
        return result.is_match, result.distance, result.confidence

    @staticmethod
    def analyze_face_attributes(image_file) -> Optional[dict]:
        """
        Analyze face attributes (eyes, smile, etc.).
        
        Delegates to the configured engine. Results vary by engine:
        - opencv: eyes_open, smiling, landmarks
        - insightface: age, gender, landmarks
        
        Returns:
            Dict with analysis results, or error dict.
        """
        try:
            img = ImageProcessor._decode_image(image_file)
            if img is None:
                return {"face_detected": False, "error": "Could not decode image"}

            engine = get_engine()
            detection = engine.detect_face(img)

            if not detection.face_found:
                return {"face_detected": False, "error": "No face detected"}

            result = engine.analyze_attributes(img, detection)
            if result is None:
                return {
                    "face_detected": True,
                    "message": f"Attribute analysis not supported by {engine.name()} engine"
                }
            return result

        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            return {"face_detected": False, "error": str(e)}

    @staticmethod
    def check_liveness(image_file) -> dict:
        """
        Perform standalone anti-spoofing / liveness check.
        
        Can be called independently from the /liveness endpoint.
        
        Returns:
            Dict with is_live, score, and details.
        """
        try:
            img = ImageProcessor._decode_image(image_file)
            if img is None:
                return {"face_detected": False, "is_live": False, "error": "Could not decode image"}

            engine = get_engine()
            detection = engine.detect_face(img)

            if not detection.face_found:
                return {"face_detected": False, "is_live": False, "error": "No face detected"}

            liveness = engine.check_liveness(img, detection)

            return {
                "face_detected": True,
                "is_live": liveness.is_live,
                "liveness_score": round(liveness.score, 4),
                "details": liveness.details,
                "message": liveness.message,
            }

        except Exception as e:
            logger.error(f"Error checking liveness: {e}")
            return {"face_detected": False, "is_live": False, "error": str(e)}
