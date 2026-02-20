"""
Domain Ports
============
Interfaces that the application layer uses to communicate with adapters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import numpy as np

from app.core.entities import FaceDetection, FaceMatch, LivenessResult, User


class FaceEnginePort(ABC):
    """Interface for face recognition engines (OpenCV, InsightFace)."""
    
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect_face(self, image: np.ndarray) -> FaceDetection: ...

    @abstractmethod
    def extract_encoding(self, image: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]: ...

    @abstractmethod
    def compare_encodings(self, enc_a: np.ndarray, enc_b: np.ndarray, threshold: float) -> FaceMatch: ...
    
    @abstractmethod
    def liveness_check(self, image: np.ndarray, detection: FaceDetection) -> LivenessResult: ...

    @abstractmethod
    def analyze_attributes(self, image: np.ndarray, detection: FaceDetection) -> Optional[Dict]: ...

    @abstractmethod
    def get_default_threshold(self) -> float: ...


class FaceRepositoryPort(ABC):
    """Interface for user and encoding storage (SQLite, etc.)."""

    @abstractmethod
    def save(self, user: User) -> None: ...

    @abstractmethod
    def get_by_id(self, user_id: str, engine: str) -> Optional[User]: ...

    @abstractmethod
    def delete(self, user_id: str, engine: str) -> bool: ...

    @abstractmethod
    def list_paginated(self, engine: str, page: int, per_page: int) -> Dict: ...
