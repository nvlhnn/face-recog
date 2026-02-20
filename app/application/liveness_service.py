"""
Liveness Service
================
Orchestrates challenge-response liveness detection sessions.

Flow:
1. Client calls start_session() → gets session_id + challenges
2. Client sends frames for each challenge → validate_step()
3. After all challenges pass → get_result()

Sessions are stored in-memory with configurable timeout.
"""

import os
import random
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.entities import (
    FaceDetection,
    LivenessSession,
    LivenessChallenge,
    ChallengeResult,
    RANDOM_CHALLENGES,
)
from app.core.ports import FaceEnginePort
from app.engines.liveness_validator import LivenessValidator
from app.extensions import logger


# In-memory session store (thread-safe)
_sessions: Dict[str, LivenessSession] = {}
_sessions_lock = threading.Lock()


def _get_config(key: str, default, cast=None):
    """Helper to get config from env with type casting."""
    val = os.getenv(key, str(default))
    if cast is not None:
        return cast(val)
    return val


class LivenessService:
    """
    Manages liveness detection sessions with challenge-response flow.

    Configuration (via environment variables):
        LIVENESS_CHALLENGE_COUNT: Number of random challenges (default: 3)
        LIVENESS_FRAMES_PER_CHALLENGE: Expected frames per challenge (default: 3)
        LIVENESS_CONSENSUS_THRESHOLD: Fraction of frames that must pass (default: 0.6)
        LIVENESS_SESSION_TIMEOUT: Session timeout in seconds (default: 120)
        ANTI_SPOOFING: Enable texture-based anti-spoof on best frame (default: false)
        ANTI_SPOOFING_THRESHOLD: Threshold for texture anti-spoof (default: 0.5)
    """

    def __init__(self, engine: FaceEnginePort):
        self.engine = engine
        self.validator = LivenessValidator()

    @property
    def challenge_count(self) -> int:
        return int(_get_config('LIVENESS_CHALLENGE_COUNT', 3))

    @property
    def consensus_threshold(self) -> float:
        return float(_get_config('LIVENESS_CONSENSUS_THRESHOLD', 0.6))

    @property
    def anti_spoof_enabled(self) -> bool:
        return _get_config('ANTI_SPOOFING', 'false').lower().strip() in ('true', '1', 'yes')

    def start_session(self) -> LivenessSession:
        """
        Create a new liveness session with randomized challenges.

        Returns a LivenessSession with:
        - challenges[0] = "look_straight" (always first, for baseline)
        - challenges[1..N] = N random challenges from the pool
        """
        # Pick N random challenges (no duplicates)
        count = min(self.challenge_count, len(RANDOM_CHALLENGES))
        random_picks = random.sample(RANDOM_CHALLENGES, count)

        # Build challenge list: look_straight first, then random picks
        challenges = [LivenessChallenge.LOOK_STRAIGHT.value]
        challenges.extend([c.value for c in random_picks])

        session = LivenessSession(challenges=challenges)

        # Store session
        with _sessions_lock:
            # Cleanup expired sessions while we're here
            self._cleanup_expired()
            _sessions[session.session_id] = session

        logger.info(
            f"Liveness session started: {session.session_id}, "
            f"challenges: {challenges}"
        )

        return session

    def get_session(self, session_id: str) -> Optional[LivenessSession]:
        """Retrieve an active session by ID."""
        with _sessions_lock:
            session = _sessions.get(session_id)
            if session and session.is_expired:
                del _sessions[session_id]
                return None
            return session

    def validate_step(
        self,
        session_id: str,
        images: List[np.ndarray],
    ) -> Tuple[bool, Dict]:
        """
        Validate the current challenge step with provided image frames.

        Args:
            session_id: Active session ID
            images: List of BGR numpy arrays (multiple frames for consensus)

        Returns:
            (success, result_dict)
        """
        session = self.get_session(session_id)
        if session is None:
            return False, {"error": "Session not found or expired"}

        if session.completed:
            return False, {"error": "Session already completed"}

        current_challenge = session.current_challenge
        if current_challenge is None:
            return False, {"error": "All challenges already completed"}

        # Detect faces in all frames
        detections: List[FaceDetection] = []
        for img in images:
            detection = self.engine.detect_face(img)
            detections.append(detection)

        # Filter to valid detections for logging
        valid_count = sum(1 for d in detections if d.face_found)
        logger.info(
            f"Session {session_id}: validating '{current_challenge}', "
            f"{valid_count}/{len(images)} frames have faces"
        )

        # STEP 0: Baseline (look_straight)
        if current_challenge == LivenessChallenge.LOOK_STRAIGHT.value:
            success, landmarks, message = self.validator.validate_baseline(detections)

            if not success:
                return False, {
                    "challenge": current_challenge,
                    "passed": False,
                    "message": message,
                    "next_challenge": current_challenge,  # retry same step
                    "step": session.current_step,
                    "total_steps": len(session.challenges),
                }

            # Store baseline landmarks
            session.baseline_landmarks = landmarks

            # Run anti-spoof check on the best frame if enabled
            if self.anti_spoof_enabled:
                best_detection = max(
                    [d for d in detections if d.face_found],
                    key=lambda d: d.confidence,
                )
                best_idx = detections.index(best_detection)
                liveness = self.engine.liveness_check(images[best_idx], best_detection)
                session.anti_spoof_score = liveness.score

                if not liveness.is_live:
                    result = ChallengeResult(
                        challenge=current_challenge,
                        passed=False,
                        score=liveness.score,
                        frames_passed=0,
                        frames_total=len(detections),
                        message=f"Anti-spoof failed: {liveness.message}",
                    )
                    session.results.append(result)
                    session.completed = True

                    return False, {
                        "challenge": current_challenge,
                        "passed": False,
                        "message": result.message,
                        "anti_spoof_score": round(liveness.score, 4),
                        "step": session.current_step,
                        "total_steps": len(session.challenges),
                    }

            # Baseline passed
            result = ChallengeResult(
                challenge=current_challenge,
                passed=True,
                score=1.0,
                frames_passed=valid_count,
                frames_total=len(detections),
                message="Baseline captured",
            )
            session.results.append(result)
            session.current_step += 1

            return True, {
                "challenge": current_challenge,
                "passed": True,
                "message": "Baseline captured successfully",
                "next_challenge": session.current_challenge,
                "step": session.current_step,
                "total_steps": len(session.challenges),
            }

        # STEPS 1-N: Challenge validation
        if session.baseline_landmarks is None:
            return False, {"error": "Baseline not captured. Complete look_straight first."}

        challenge_result = self.validator.validate_challenge(
            challenge=current_challenge,
            detections=detections,
            baseline_landmarks=session.baseline_landmarks,
            consensus_threshold=self.consensus_threshold,
        )

        session.results.append(challenge_result)

        if challenge_result.passed:
            session.current_step += 1

            # Check if all challenges completed
            if session.current_step >= len(session.challenges):
                session.completed = True
                return True, {
                    "challenge": current_challenge,
                    "passed": True,
                    "score": challenge_result.score,
                    "message": "All challenges completed!",
                    "next_challenge": None,
                    "step": session.current_step,
                    "total_steps": len(session.challenges),
                    "session_complete": True,
                }

            return True, {
                "challenge": current_challenge,
                "passed": True,
                "score": challenge_result.score,
                "message": challenge_result.message,
                "next_challenge": session.current_challenge,
                "step": session.current_step,
                "total_steps": len(session.challenges),
            }
        else:
            # Challenge failed — client can retry the same step
            return False, {
                "challenge": current_challenge,
                "passed": False,
                "score": challenge_result.score,
                "message": challenge_result.message,
                "next_challenge": current_challenge,  # retry
                "step": session.current_step,
                "total_steps": len(session.challenges),
            }

    def get_result(self, session_id: str) -> Tuple[bool, Dict]:
        """
        Get the final result of a completed liveness session.

        Returns:
            (success, result_dict)
        """
        session = self.get_session(session_id)
        if session is None:
            return False, {"error": "Session not found or expired"}

        if not session.completed:
            return False, {
                "error": "Session not completed",
                "current_step": session.current_step,
                "total_steps": len(session.challenges),
                "next_challenge": session.current_challenge,
            }

        all_passed = session.all_passed

        # Calculate overall score
        if session.results:
            overall_score = sum(r.score for r in session.results) / len(session.results)
        else:
            overall_score = 0.0

        result = {
            "session_id": session.session_id,
            "verified": all_passed,
            "overall_score": round(overall_score, 4),
            "challenges": [
                {
                    "challenge": r.challenge,
                    "passed": r.passed,
                    "score": r.score,
                    "frames": f"{r.frames_passed}/{r.frames_total}",
                }
                for r in session.results
            ],
            "engine": self.engine.name(),
        }

        if session.anti_spoof_score is not None:
            result["anti_spoof_score"] = round(session.anti_spoof_score, 4)

        # Cleanup session after result retrieval
        with _sessions_lock:
            _sessions.pop(session_id, None)

        return True, result

    def _cleanup_expired(self):
        """Remove expired sessions (called within lock)."""
        expired = [
            sid for sid, s in _sessions.items() if s.is_expired
        ]
        for sid in expired:
            del _sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired liveness sessions")
