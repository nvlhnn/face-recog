"""
InsightFace Engine
==================
Face recognition using the InsightFace library (ArcFace model).
Higher accuracy than OpenCV/SFace, but heavier (~500MB RAM).

Install: pip install insightface onnxruntime
"""

import numpy as np
import os
from typing import Optional, Dict

from app.core.entities import FaceDetection, FaceMatch
from app.engines.base import FaceEngineBase, DetectionResult, ComparisonResult
from app.extensions import logger


class InsightFaceEngine(FaceEngineBase):
    """InsightFace-based face recognition using ArcFace."""

    def __init__(self, model_name: str = "buffalo_l"):
        self._model_name = model_name
        self._app = None

    def name(self) -> str:
        return "insightface"

    def initialize(self) -> None:
        """Load InsightFace model pack."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is not installed. Run: pip install insightface onnxruntime"
            )

        # Set model root to /app/models/insightface
        model_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'insightface')
        
        self._app = FaceAnalysis(
            name=self._model_name, 
            root=model_root,
            providers=["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info(f"InsightFace loaded (model: {self._model_name}, root: {model_root})")

    def _ensure_initialized(self):
        if self._app is None:
            self.initialize()

    def detect_face(self, image: np.ndarray) -> FaceDetection:
        """Detect the largest face using InsightFace."""
        self._ensure_initialized()

        faces = self._app.get(image)

        if not faces:
            return FaceDetection(face_found=False)

        # Select largest face by bbox area
        if len(faces) > 1:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        else:
            face = faces[0]

        bbox = face.bbox  # [x1, y1, x2, y2]
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y

        landmarks = None
        if face.kps is not None:
            # InsightFace returns 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
            kps = face.kps.tolist()
            landmarks = {
                "left_eye": kps[0],
                "right_eye": kps[1],
                "nose": kps[2],
                "left_mouth": kps[3],
                "right_mouth": kps[4],
            }

        score = float(face.det_score) if hasattr(face, "det_score") else 0.0

        return DetectionResult(
            face_found=True,
            bbox=(float(x), float(y), float(w), float(h)),
            landmarks=landmarks,
            confidence=score,
            raw=face,
        ).to_domain()


    def extract_encoding(self, image: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """Extract 512-dim face embedding using ArcFace."""
        if not detection.face_found or detection.raw_data is None:
            return None

        face = detection.raw_data
        if face.embedding is None:
            return None

        encoding = face.embedding.flatten()
        logger.info(f"Extracted encoding: {len(encoding)} dimensions")
        return encoding


    def compare_encodings(
        self,
        encoding_a: np.ndarray,
        encoding_b: np.ndarray,
        threshold: float
    ) -> FaceMatch:
        """Compare two embeddings using cosine similarity."""
        try:
            # Safeguard: Ensure both encodings have the same dimension (512 for InsightFace)
            if encoding_a.shape[0] != 512 or encoding_b.shape[0] != 512:
                logger.error(
                    f"Invalid encoding dimensions for InsightFace: {encoding_a.shape} or {encoding_b.shape}. "
                    "Expected (512,)."
                )
                return ComparisonResult(is_match=False, distance=1.0, confidence=0.0).to_domain()

            # Normalize
            a = encoding_a / (np.linalg.norm(encoding_a) + 1e-6)
            b = encoding_b / (np.linalg.norm(encoding_b) + 1e-6)

            # Cosine similarity
            cosine_sim = float(np.dot(a, b))

            # Distance (0 = identical, 2 = opposite)
            distance = 1.0 - cosine_sim

            # Confidence (map similarity to 0-100)
            confidence = max(0, min(100, cosine_sim / threshold * 100 * threshold))

            is_match = cosine_sim >= threshold
            return ComparisonResult(
                is_match=bool(is_match),
                distance=float(distance),
                confidence=float(confidence),
            ).to_domain()
        except Exception as e:
            logger.error(f"Error during InsightFace comparison: {e}")
            return ComparisonResult(is_match=False, distance=1.0, confidence=0.0).to_domain()



    def analyze_attributes(self, image: np.ndarray, detection: FaceDetection) -> Optional[Dict]:
        """Analyze attributes using InsightFace."""
        if not detection.face_found or detection.raw_data is None:
            return {"face_detected": False, "error": "No face detected"}

        face = detection.raw_data

        result = {"face_detected": True}

        # InsightFace provides age and gender natively
        if hasattr(face, "age") and face.age is not None:
            result["age"] = int(face.age)
        if hasattr(face, "gender") and face.gender is not None:
            result["gender"] = "male" if face.gender == 1 else "female"

        # Landmarks-based smile/eyes detection (if landmarks available)
        if detection.landmarks:
            result["landmarks"] = detection.landmarks

        return result

    def default_threshold(self) -> float:
        return 0.4
