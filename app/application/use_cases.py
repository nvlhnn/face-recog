"""
Application Use Cases
=====================
Orchestrates business logic flows.
"""

from typing import Optional, Dict, Tuple
import numpy as np

from app.core.entities import User, FaceDetection
from app.core.ports import FaceEnginePort, FaceRepositoryPort


class FaceRecognitionUseCases:
    """Use cases for face registration and verification."""

    def __init__(self, engine: FaceEnginePort, repository: FaceRepositoryPort):
        self.engine = engine
        self.repository = repository

    def register_user(self, user_id: str, image: np.ndarray, anti_spoof: bool) -> Tuple[bool, str]:
        """Process and register a new face."""
        # 1. Detect Face
        detection = self.engine.detect_face(image)
        if not detection.face_found:
            return False, "No face detected"

        # 2. Optional anti-spoofing
        if anti_spoof:
            liveness = self.engine.liveness_check(image, detection)
            if not liveness.is_live:
                return False, f"Spoof detected: {liveness.message}"

        # 3. Extract Encoding
        encoding = self.engine.extract_encoding(image, detection)
        if encoding is None:
            return False, "Failed to extract face features"

        # 4. Save to Repository
        user = User(user_id=user_id, engine=self.engine.name(), encoding=encoding)
        self.repository.save(user)
        
        return True, "User registered successfully"

    def verify_user(self, user_id: str, image: np.ndarray, anti_spoof: bool, tolerance: float = None) -> Dict:
        """Verify a user against stored records."""
        # 1. Get stored user
        stored_user = self.repository.get_by_id(user_id, self.engine.name())
        if not stored_user:
            return {"matched": False, "message": "User not found"}

        # 2. Detect face in live image
        detection = self.engine.detect_face(image)
        if not detection.face_found:
            return {"matched": False, "message": "No face detected in photo"}

        # 3. Anti-spoofing
        if anti_spoof:
            liveness = self.engine.liveness_check(image, detection)
            if not liveness.is_live:
                return {"matched": False, "message": liveness.message}

        # 4. Extract live encoding
        current_encoding = self.engine.extract_encoding(image, detection)
        if current_encoding is None:
            return {"matched": False, "message": "Extraction failed"}

        # 5. Compare
        threshold = tolerance or self.engine.get_default_threshold()
        match = self.engine.compare_encodings(stored_user.encoding, current_encoding, threshold)

        return {
            "matched": match.is_match,
            "distance": round(match.distance, 4),
            "confidence": round(match.confidence, 1),
            "engine": self.engine.name()
        }

    def analyze_face(self, image: np.ndarray) -> Dict:
        """Extract attributes without registration."""
        detection = self.engine.detect_face(image)
        if not detection.face_found:
            return {"face_detected": False, "message": "No face detected"}
            
        attrs = self.engine.analyze_attributes(image, detection)
        return {"face_detected": True, "engine": self.engine.name(), "attributes": attrs}

    def delete_user(self, user_id: str) -> bool:
        """Delete user data."""
        return self.repository.delete(user_id, self.engine.name())

    def list_users(self, page: int, per_page: int) -> Dict:
        """Paginated list of users."""
        return self.repository.list_paginated(self.engine.name(), page, per_page)

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user details."""
        return self.repository.get_by_id(user_id, self.engine.name())

