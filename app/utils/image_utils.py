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

    @staticmethod
    def analyze_face_attributes(image_file) -> Optional[dict]:
        """
        Analyze face attributes: eyes open/closed, smile detection.
        
        YuNet returns 5 landmarks:
        - [4-5]: Right eye (x, y)
        - [6-7]: Left eye (x, y)
        - [8-9]: Nose tip (x, y)
        - [10-11]: Right mouth corner (x, y)
        - [12-13]: Left mouth corner (x, y)
        
        Returns dict with:
        - eyes_open: bool (True if eyes appear open)
        - smiling: bool (True if mouth appears in smile)
        - face_detected: bool
        """
        try:
            img = ImageProcessor._decode_image(image_file)
            if img is None:
                return {"face_detected": False, "error": "Could not decode image"}
            
            face = ImageProcessor._detect_face(img)
            if face is None:
                return {"face_detected": False, "error": "No face detected"}
            
            # Extract landmarks from YuNet detection
            # face format: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y, 
            #               nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
            
            x, y, w, h = face[0:4]
            right_eye = (face[4], face[5])
            left_eye = (face[6], face[7])
            nose = (face[8], face[9])
            right_mouth = (face[10], face[11])
            left_mouth = (face[12], face[13])
            
            # Calculate face proportions for analysis
            face_width = w
            face_height = h
            
            # Eye openness: Check eye region brightness/contrast
            # Simple heuristic: eyes are typically in upper 1/3 of face
            eye_distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
            
            # Mouth width for smile detection
            mouth_width = np.sqrt((left_mouth[0] - right_mouth[0])**2 + (left_mouth[1] - right_mouth[1])**2)
            
            # ===== IMPROVED SMILE DETECTION =====
            # Method 1: Mouth width ratio (works for big smiles with teeth)
            smile_ratio = float(mouth_width / eye_distance) if eye_distance > 0 else 0
            
            # Method 2: Mouth corner elevation (works for subtle smiles)
            # When smiling, mouth corners tend to be closer to the eyes (raised)
            # Calculate average mouth corner height relative to nose
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            nose_to_mouth = float(mouth_center_y - nose[1])  # Distance from nose to mouth
            nose_to_eyes = float(nose[1] - (left_eye[1] + right_eye[1]) / 2)  # Distance from eyes to nose
            
            # Ratio of nose-to-mouth vs nose-to-eyes (smaller = mouth corners raised = smile)
            vertical_ratio = nose_to_mouth / nose_to_eyes if nose_to_eyes > 0 else 1.0
            
            # Combined smile detection:
            # - Wide smile (teeth): mouth_ratio > 1.15
            # - Subtle smile (no teeth): vertical_ratio < 0.75 (mouth corners raised)
            wide_smile = smile_ratio > 1.10
            subtle_smile = vertical_ratio < 0.80 and smile_ratio > 0.95
            
            smiling = wide_smile or subtle_smile
            
            # Confidence based on strongest signal
            if wide_smile:
                smile_confidence = min((smile_ratio - 0.9) / 0.4 * 100, 100)
            elif subtle_smile:
                smile_confidence = min((0.85 - vertical_ratio) / 0.15 * 80, 80)
            else:
                smile_confidence = max(0, (smile_ratio - 0.8) / 0.3 * 50)
            
            # For eyes, we can check if eye points are above nose
            # (basic check - real eye state detection needs more landmarks)
            eyes_level = (left_eye[1] + right_eye[1]) / 2
            nose_level = nose[1]
            eyes_open = eyes_level < nose_level  # Eyes should be above nose
            
            return {
                "face_detected": True,
                "eyes_open": bool(eyes_open),
                "smiling": bool(smiling),
                "smile_confidence": float(round(min(float(smile_ratio) / 1.5 * 100, 100), 1)),
                "landmarks": {
                    "right_eye": [float(right_eye[0]), float(right_eye[1])],
                    "left_eye": [float(left_eye[0]), float(left_eye[1])],
                    "nose": [float(nose[0]), float(nose[1])],
                    "right_mouth": [float(right_mouth[0]), float(right_mouth[1])],
                    "left_mouth": [float(left_mouth[0]), float(left_mouth[1])]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            return {"face_detected": False, "error": str(e)}

