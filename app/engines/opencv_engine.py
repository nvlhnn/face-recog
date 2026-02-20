"""
OpenCV Engine (YuNet + SFace)
=============================
Face recognition using OpenCV's DNN module.
RAM: ~150MB | Speed: ~100-200ms per verification

This is the default, lightweight engine.
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict

from app.core.entities import FaceDetection, FaceMatch
from app.engines.base import FaceEngineBase, DetectionResult, ComparisonResult
from app.extensions import logger

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")


class OpenCVEngine(FaceEngineBase):
    """OpenCV-based face recognition using YuNet detector + SFace recognizer."""

    def __init__(self):
        self._detector = None
        self._recognizer = None

    def name(self) -> str:
        return "opencv"

    def initialize(self) -> None:
        """Load YuNet and SFace models."""
        if not os.path.exists(DETECTOR_MODEL):
            raise FileNotFoundError(
                f"Model not found: {DETECTOR_MODEL}. Run 'python download_models.py' first."
            )
        if not os.path.exists(RECOGNIZER_MODEL):
            raise FileNotFoundError(
                f"Model not found: {RECOGNIZER_MODEL}. Run 'python download_models.py' first."
            )

        self._detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
        logger.info("YuNet face detector loaded")

        self._recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        logger.info("SFace recognizer loaded")

    def _ensure_initialized(self):
        """Lazy-initialize models on first use."""
        if self._detector is None or self._recognizer is None:
            self.initialize()

    def detect_face(self, image: np.ndarray) -> FaceDetection:
        """Detect the largest face using YuNet."""
        self._ensure_initialized()

        h, w = image.shape[:2]

        # Lower the score threshold for more lenient detection
        self._detector.setScoreThreshold(0.5)

        # Try detection at original size first
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)

        # If no face found, try with resized image
        if faces is None or len(faces) == 0:
            scale = min(640 / w, 640 / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(image, (new_w, new_h))
                self._detector.setInputSize((new_w, new_h))
                _, faces = self._detector.detect(resized)

                # Scale face coordinates back to original
                if faces is not None and len(faces) > 0:
                    faces = faces / scale

        if faces is None or len(faces) == 0:
            return FaceDetection(face_found=False)

        # Select largest face (by area)
        if len(faces) > 1:
            areas = [(f[2] * f[3]) for f in faces]
            largest_idx = np.argmax(areas)
            face = faces[largest_idx]
        else:
            face = faces[0]

        # Extract landmarks from YuNet detection
        landmarks = None
        if len(face) >= 14:
            landmarks = {
                "right_eye": [float(face[4]), float(face[5])],
                "left_eye": [float(face[6]), float(face[7])],
                "nose": [float(face[8]), float(face[9])],
                "right_mouth": [float(face[10]), float(face[11])],
                "left_mouth": [float(face[12]), float(face[13])],
            }

        score = float(face[14]) if len(face) > 14 else 0.0

        return DetectionResult(
            face_found=True,
            bbox=(float(face[0]), float(face[1]), float(face[2]), float(face[3])),
            landmarks=landmarks,
            confidence=score,
            raw=face,  # Keep raw face data for alignCrop
        ).to_domain()


    def extract_encoding(self, image: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """Extract 128-dim face encoding using SFace."""
        self._ensure_initialized()

        if not detection.face_found or detection.raw_data is None:
            return None

        try:
            # SFace alignCrop expects the raw detection result from detect()
            aligned_face = self._recognizer.alignCrop(image, detection.raw_data)
            encoding = self._recognizer.feature(aligned_face)
            encoding = encoding.flatten()
            logger.info(f"Extracted encoding: {len(encoding)} dimensions")
            return encoding
        except Exception as e:
            logger.error(f"Error extracting encoding: {e}")
            return None

    def compare_encodings(
        self,
        encoding_a: np.ndarray,
        encoding_b: np.ndarray,
        threshold: float
    ) -> FaceMatch:
        """Compare two encodings using SFace cosine similarity."""
        self._ensure_initialized()

        try:
            # Ensure we are working with float32 vectors
            feat1 = encoding_a.astype(np.float32).reshape(1, -1)
            feat2 = encoding_b.astype(np.float32).reshape(1, -1)

            # Safeguard: SFace uses 128-dimensional vectors
            if feat1.shape[1] != 128 or feat2.shape[1] != 128:
                logger.error(
                    f"Invalid encoding dimensions for OpenCV: {feat1.shape[1]} or {feat2.shape[1]}. "
                    "Expected 128."
                )
                return ComparisonResult(is_match=False, distance=1.0, confidence=0.0).to_domain()

            # Cosine similarity score (higher = more similar)
            cosine_score = self._recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)

            # Convert to distance (0 = identical, 1 = different)
            distance = 1 - cosine_score

            # Confidence percentage
            confidence = max(0, min(100, cosine_score * 100 / threshold * 0.363))

            is_match = cosine_score >= threshold
            return ComparisonResult(
                is_match=bool(is_match),
                distance=float(distance),
                confidence=float(confidence),
            ).to_domain()
        except Exception as e:
            logger.error(f"Error during OpenCV comparison: {e}")
            return ComparisonResult(is_match=False, distance=1.0, confidence=0.0).to_domain()



    def analyze_attributes(self, image: np.ndarray, detection: FaceDetection) -> Optional[Dict]:
        """Analyze face attributes using YuNet landmarks."""
        if not detection.face_found or detection.landmarks is None:
            return {"face_detected": False, "error": "No face detected"}

        landmarks = detection.landmarks
        right_eye = landmarks["right_eye"]
        left_eye = landmarks["left_eye"]
        nose = landmarks["nose"]
        right_mouth = landmarks["right_mouth"]
        left_mouth = landmarks["left_mouth"]

        # Eye distance
        eye_distance = np.sqrt(
            (left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2
        )

        # Mouth width
        mouth_width = np.sqrt(
            (left_mouth[0] - right_mouth[0]) ** 2 + (left_mouth[1] - right_mouth[1]) ** 2
        )

        # Smile detection
        smile_ratio = float(mouth_width / eye_distance) if eye_distance > 0 else 0

        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        nose_to_mouth = float(mouth_center_y - nose[1])
        nose_to_eyes = float(nose[1] - (left_eye[1] + right_eye[1]) / 2)
        vertical_ratio = nose_to_mouth / nose_to_eyes if nose_to_eyes > 0 else 1.0

        wide_smile = smile_ratio > 1.10
        subtle_smile = vertical_ratio < 0.80 and smile_ratio > 0.95
        smiling = wide_smile or subtle_smile

        if wide_smile:
            smile_confidence = min((smile_ratio - 0.9) / 0.4 * 100, 100)
        elif subtle_smile:
            smile_confidence = min((0.85 - vertical_ratio) / 0.15 * 80, 80)
        else:
            smile_confidence = max(0, (smile_ratio - 0.8) / 0.3 * 50)

        # Eye openness
        eyes_level = (left_eye[1] + right_eye[1]) / 2
        nose_level = nose[1]
        eyes_open = eyes_level < nose_level

        return {
            "face_detected": True,
            "eyes_open": bool(eyes_open),
            "smiling": bool(smiling),
            "smile_confidence": float(round(min(float(smile_ratio) / 1.5 * 100, 100), 1)),
            "landmarks": landmarks,
        }

    def default_threshold(self) -> float:
        return 0.363
