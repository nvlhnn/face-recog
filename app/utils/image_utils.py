"""
Image Utilities (OpenCV + SFace)
================================
Lightweight face recognition using OpenCV DNN.
RAM: ~150MB | Speed: ~100-200ms per verification
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple
from app.extensions import logger

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Lazy-loaded models
_detector = None
_recognizer = None


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


def get_detector():
    """Get face detector (lazy load)."""
    global _detector
    if _detector is None:
        if not os.path.exists(DETECTOR_MODEL):
            raise FileNotFoundError(f"Model not found: {DETECTOR_MODEL}. Run 'python download_models.py' first.")
        _detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
        logger.info("YuNet face detector loaded")
    return _detector


def get_recognizer():
    """Get face recognizer (lazy load)."""
    global _recognizer
    if _recognizer is None:
        if not os.path.exists(RECOGNIZER_MODEL):
            raise FileNotFoundError(f"Model not found: {RECOGNIZER_MODEL}. Run 'python download_models.py' first.")
        _recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        logger.info("SFace recognizer loaded")
    return _recognizer


class ImageProcessor:
    """OpenCV-based face processing utilities."""
    
    @staticmethod
    def _decode_image(image_file) -> Optional[np.ndarray]:
        """Decode uploaded file to OpenCV image."""
        image_file.seek(0)
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    
    @staticmethod
    def _detect_face(img: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest face in an image with multiple attempts."""
        detector = get_detector()
        
        h, w = img.shape[:2]
        
        # Lower the score threshold for more lenient detection (default is 0.9)
        detector.setScoreThreshold(0.5)
        
        # Try detection at original size first
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        
        # If no face found, try with resized image (sometimes helps)
        if faces is None or len(faces) == 0:
            # Try with standard size
            scale = min(640 / w, 640 / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h))
                detector.setInputSize((new_w, new_h))
                _, faces = detector.detect(resized)
                
                # Scale face coordinates back to original
                if faces is not None and len(faces) > 0:
                    faces = faces / scale
        
        if faces is None or len(faces) == 0:
            return None
        
        # Return largest face (by area)
        if len(faces) > 1:
            areas = [(f[2] * f[3]) for f in faces]
            largest_idx = np.argmax(areas)
            return faces[largest_idx]
        
        return faces[0]
    
    @staticmethod
    def get_face_encoding(image_file) -> Optional[np.ndarray]:
        """
        Extract face encoding (128-dim feature vector) from uploaded image.
        """
        try:
            img = ImageProcessor._decode_image(image_file)
            if img is None:
                logger.warning("Could not decode image")
                return None
            
            face = ImageProcessor._detect_face(img)
            if face is None:
                logger.warning("No face detected in image")
                return None
            
            # Align and crop face, then extract features
            recognizer = get_recognizer()
            aligned_face = recognizer.alignCrop(img, face)
            encoding = recognizer.feature(aligned_face)
            
            # Flatten to 1D array
            encoding = encoding.flatten()
            logger.info(f"Extracted encoding: {len(encoding)} dimensions")
            return encoding
            
        except Exception as e:
            logger.error(f"Error extracting encoding: {e}")
            raise
    
    @staticmethod
    def compare_encodings(known_encoding: np.ndarray, unknown_encoding: np.ndarray, tolerance: float = 0.363) -> Tuple[bool, float, float]:
        """
        Compare two face encodings using cosine similarity.
        
        Args:
            known_encoding: Stored face encoding
            unknown_encoding: New face encoding to compare
            tolerance: Match threshold (default 0.363 for SFace cosine)
        
        Returns:
            (is_match, distance, confidence_percent)
        """
        recognizer = get_recognizer()
        
        # Ensure same type (float32) and reshape for OpenCV
        feat1 = known_encoding.astype(np.float32).reshape(1, -1)
        feat2 = unknown_encoding.astype(np.float32).reshape(1, -1)
        
        # Cosine similarity score (higher = more similar)
        cosine_score = recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        
        # Convert to distance (0 = identical, 1 = different)
        distance = 1 - cosine_score
        
        # Confidence percentage
        confidence = max(0, min(100, cosine_score * 100 / tolerance * 0.363))
        
        is_match = cosine_score >= tolerance
        return bool(is_match), float(distance), float(confidence)
