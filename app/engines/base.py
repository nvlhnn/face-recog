"""
Face Engine Base
================
Abstract base class defining the interface for all face recognition engines.
Implementations: OpenCV (SFace), InsightFace
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


from app.core.entities import FaceDetection, FaceMatch, LivenessResult as DomainLivenessResult
from app.core.ports import FaceEnginePort


@dataclass
class DetectionResult:
    """Internal Engine Result (Legacy compatibility)"""
    face_found: bool
    bbox: Optional[Tuple[float, float, float, float]] = None
    landmarks: Optional[Dict[str, List[float]]] = None
    confidence: float = 0.0
    raw: Any = None
    
    def to_domain(self) -> FaceDetection:
        return FaceDetection(
            face_found=self.face_found,
            bbox=self.bbox,
            landmarks=self.landmarks,
            confidence=self.confidence,
            raw_data=self.raw
        )


@dataclass
class ComparisonResult:
    """Internal Engine Result (Legacy compatibility)"""
    is_match: bool
    distance: float
    confidence: float
    
    def to_domain(self) -> FaceMatch:
        return FaceMatch(
            is_match=self.is_match,
            distance=self.distance,
            confidence=self.confidence
        )


class FaceEngineBase(FaceEnginePort, ABC):
    """
    Abstract base class for face recognition engines.
    
    All engines must implement:
    - detect_face: Find faces in an image
    - extract_encoding: Get a face embedding/encoding vector
    - compare_encodings: Compare two encoding vectors
    - analyze_attributes: (Optional) Detect smile, eyes open, etc.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of this engine (e.g. 'opencv', 'insightface')."""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        Load models and prepare the engine. Called once at startup.
        """
        ...

    @abstractmethod
    def detect_face(self, image: np.ndarray) -> FaceDetection:
        """
        Detect the largest/most prominent face in an image.
        Returns a Domain FaceDetection entity.
        """
        ...

    @abstractmethod
    def extract_encoding(self, image: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract a face encoding/embedding vector from a detected face.
        """
        ...

    @abstractmethod
    def compare_encodings(
        self,
        encoding_a: np.ndarray,
        encoding_b: np.ndarray,
        threshold: float
    ) -> FaceMatch:
        """
        Compare two face encodings. Returns Domain FaceMatch entity.
        """
        ...

    def analyze_attributes(self, image: np.ndarray, detection: FaceDetection) -> Optional[Dict]:
        """
        Analyze face attributes. Default is not supported.
        """
        return None

    def get_default_threshold(self) -> float:
        return self.default_threshold()

    def liveness_check(self, image: np.ndarray, detection: FaceDetection) -> DomainLivenessResult:
        """Default implementation using the texture-based AntiSpoofChecker."""
        from app.engines.anti_spoof import AntiSpoofChecker
        
        threshold = float(os.getenv('ANTI_SPOOFING_THRESHOLD', '0.5'))
        checker = AntiSpoofChecker(threshold=threshold)
        
        # Internal check wants an internal DetectionResult (for now or just bbox)
        # But we can just pass bbox to it.
        result = checker.check(image, face_bbox=detection.bbox)
        
        return DomainLivenessResult(
            is_live=result.is_live,
            score=result.score,
            details=result.details,
            message=result.message
        )


    def check_liveness(
        self,
        image: np.ndarray,
        detection: 'DetectionResult',
        threshold: float = None
    ) -> 'LivenessResult':
        """
        Perform anti-spoofing / liveness check on a face image.
        
        Default implementation uses the texture-based AntiSpoofChecker.
        Engines can override with their own anti-spoofing methods.
        
        Args:
            image: BGR numpy array (OpenCV format)
            detection: Result from detect_face
            threshold: Liveness score threshold (0.0-1.0). Uses ANTI_SPOOFING_THRESHOLD env var or 0.5.
            
        Returns:
            LivenessResult with is_live, score, and details.
        """
    @abstractmethod
    def default_threshold(self) -> float:
        """Return the default match threshold for this engine."""
        ...


