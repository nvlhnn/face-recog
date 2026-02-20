"""
Domain Entities
===============
Pure business models with no dependencies.
"""

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


@dataclass
class FaceDetection:
    """A detected face in an image."""
    face_found: bool
    bbox: Optional[tuple] = None  # (x, y, w, h)
    landmarks: Optional[Dict[str, List[float]]] = None
    confidence: float = 0.0
    raw_data: Any = None


@dataclass
class FaceMatch:
    """Result of a face comparison."""
    is_match: bool
    distance: float
    confidence: float  # 0-100 percentage


@dataclass
class LivenessResult:
    """Result from anti-spoofing check."""
    is_live: bool
    score: float
    details: Dict[str, float]
    message: Optional[str] = None


class LivenessChallenge(str, Enum):
    """Supported liveness challenge types."""
    LOOK_STRAIGHT = "look_straight"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    SMILE = "smile"
    OPEN_MOUTH = "open_mouth"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"


# Challenges available for random selection (excludes look_straight which is always first)
RANDOM_CHALLENGES = [
    LivenessChallenge.LOOK_LEFT,
    LivenessChallenge.LOOK_RIGHT,
    LivenessChallenge.SMILE,
    LivenessChallenge.OPEN_MOUTH,
    LivenessChallenge.LOOK_UP,
    LivenessChallenge.LOOK_DOWN,
]


@dataclass
class ChallengeResult:
    """Result of validating a single challenge."""
    challenge: str
    passed: bool
    score: float  # 0.0 - 1.0, how well the challenge was met
    frames_passed: int  # how many frames passed
    frames_total: int  # total frames evaluated
    message: Optional[str] = None


@dataclass
class LivenessSession:
    """Tracks a liveness detection session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    challenges: List[str] = field(default_factory=list)
    current_step: int = 0
    results: List[ChallengeResult] = field(default_factory=list)
    baseline_landmarks: Optional[Dict[str, List[float]]] = None
    anti_spoof_score: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    completed: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if session has exceeded timeout."""
        from app.config import Config
        timeout = getattr(Config, 'LIVENESS_SESSION_TIMEOUT', 120)
        return (time.time() - self.created_at) > timeout

    @property
    def current_challenge(self) -> Optional[str]:
        """Get the current challenge to validate."""
        if self.current_step < len(self.challenges):
            return self.challenges[self.current_step]
        return None

    @property
    def all_passed(self) -> bool:
        """Check if all challenges have been passed."""
        return (
            len(self.results) == len(self.challenges)
            and all(r.passed for r in self.results)
        )


@dataclass
class User:
    """Registered user entity."""
    user_id: str
    engine: str
    encoding: Optional[np.ndarray] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
