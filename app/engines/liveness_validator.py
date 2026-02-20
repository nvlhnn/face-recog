"""
Liveness Validator
==================
Validates facial landmarks against liveness challenges.

Uses relative geometry (compared to a baseline "look_straight" frame)
to determine if the user is performing the requested action.

Works with both OpenCV (YuNet) and InsightFace engines since both
produce 5-point landmarks: right_eye, left_eye, nose, right_mouth, left_mouth.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from app.core.entities import (
    FaceDetection,
    ChallengeResult,
    LivenessChallenge,
)
from app.extensions import logger


class LivenessValidator:
    """
    Validates facial landmarks against liveness challenges.

    All checks are RELATIVE to a baseline frame (look_straight),
    which normalizes for different face shapes, distances, and
    engine accuracy differences.
    """

    # --- Thresholds (tuned for 5-point landmarks) ---
    # These are ratios relative to inter-eye distance for scale invariance

    # Head turn: how much the nose must shift horizontally relative to baseline
    TURN_THRESHOLD = 0.08

    # Head tilt: how much the vertical nose-to-eye ratio must change
    TILT_UP_THRESHOLD = 0.10    # nose moves closer to eyes
    TILT_DOWN_THRESHOLD = 0.12  # nose moves farther from eyes

    # Smile: how much the mouth-width/eye-distance ratio must increase
    SMILE_THRESHOLD = 1.08  # multiplier over baseline ratio

    # Open mouth: mouth corners spread vertically
    OPEN_MOUTH_THRESHOLD = 1.15  # multiplier over baseline vertical mouth spread

    def _extract_geometry(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Extract normalized geometric features from 5-point landmarks.

        Returns a dict of features that can be compared across frames.
        All values are normalized by inter-eye distance for scale invariance.
        """
        right_eye = np.array(landmarks["right_eye"])
        left_eye = np.array(landmarks["left_eye"])
        nose = np.array(landmarks["nose"])
        right_mouth = np.array(landmarks["right_mouth"])
        left_mouth = np.array(landmarks["left_mouth"])

        # Inter-eye distance (used for normalization)
        eye_dist = np.linalg.norm(left_eye - right_eye)
        if eye_dist < 1e-6:
            eye_dist = 1.0  # prevent division by zero

        # Eye midpoint
        eye_mid = (left_eye + right_eye) / 2.0

        # Horizontal nose offset from eye midpoint (normalized)
        nose_offset_x = (nose[0] - eye_mid[0]) / eye_dist

        # Vertical nose-to-eye-midpoint distance (normalized)
        nose_to_eyes_y = (nose[1] - eye_mid[1]) / eye_dist

        # Mouth width (normalized)
        mouth_width = np.linalg.norm(left_mouth - right_mouth) / eye_dist

        # Mouth center
        mouth_mid = (left_mouth + right_mouth) / 2.0

        # Vertical mouth spread (distance between mouth corners in Y)
        mouth_vertical = abs(left_mouth[1] - right_mouth[1]) / eye_dist

        # Nose to mouth vertical distance (normalized)
        nose_to_mouth_y = (mouth_mid[1] - nose[1]) / eye_dist

        # Smile ratio: mouth_width / eye_distance (classic metric)
        smile_ratio = float(mouth_width)

        return {
            "eye_dist": float(eye_dist),
            "nose_offset_x": float(nose_offset_x),
            "nose_to_eyes_y": float(nose_to_eyes_y),
            "mouth_width": float(mouth_width),
            "mouth_vertical": float(mouth_vertical),
            "nose_to_mouth_y": float(nose_to_mouth_y),
            "smile_ratio": float(smile_ratio),
        }

    def validate_challenge(
        self,
        challenge: str,
        detections: List[FaceDetection],
        baseline_landmarks: Dict[str, List[float]],
        consensus_threshold: float = 0.6,
    ) -> ChallengeResult:
        """
        Validate multiple frames against a liveness challenge.

        Args:
            challenge: The challenge type (from LivenessChallenge enum)
            detections: List of FaceDetection results from multiple frames
            baseline_landmarks: Landmarks from the "look_straight" baseline frame
            consensus_threshold: Fraction of frames that must pass (e.g., 0.6 = 2/3)

        Returns:
            ChallengeResult with pass/fail and details
        """
        if not detections:
            return ChallengeResult(
                challenge=challenge,
                passed=False,
                score=0.0,
                frames_passed=0,
                frames_total=0,
                message="No frames provided",
            )

        # Extract baseline geometry
        baseline_geo = self._extract_geometry(baseline_landmarks)

        # Check each frame
        frames_passed = 0
        scores = []
        check_fn = self._get_check_function(challenge)

        if check_fn is None:
            return ChallengeResult(
                challenge=challenge,
                passed=False,
                score=0.0,
                frames_passed=0,
                frames_total=len(detections),
                message=f"Unknown challenge: {challenge}",
            )

        for detection in detections:
            if not detection.face_found or detection.landmarks is None:
                scores.append(0.0)
                continue

            current_geo = self._extract_geometry(detection.landmarks)
            passed, score = check_fn(current_geo, baseline_geo)

            if passed:
                frames_passed += 1
            scores.append(score)

        total_frames = len(detections)
        avg_score = float(np.mean(scores)) if scores else 0.0
        consensus_met = (frames_passed / total_frames) >= consensus_threshold if total_frames > 0 else False

        return ChallengeResult(
            challenge=challenge,
            passed=consensus_met,
            score=round(avg_score, 4),
            frames_passed=frames_passed,
            frames_total=total_frames,
            message=f"{'Passed' if consensus_met else 'Failed'}: {frames_passed}/{total_frames} frames",
        )

    def validate_baseline(self, detections: List[FaceDetection]) -> Tuple[bool, Optional[Dict[str, List[float]]], str]:
        """
        Validate the baseline (look_straight) frame.

        Selects the best detection (highest confidence) and returns its landmarks
        as the baseline for subsequent challenges.

        Returns:
            (success, baseline_landmarks, message)
        """
        valid_detections = [d for d in detections if d.face_found and d.landmarks is not None]

        if not valid_detections:
            return False, None, "No face detected in baseline frames"

        # Pick the detection with highest confidence
        best = max(valid_detections, key=lambda d: d.confidence)

        # Basic sanity: ensure landmarks make geometric sense
        landmarks = best.landmarks
        geo = self._extract_geometry(landmarks)

        # Nose should be roughly centered (within ±0.2 of eye midpoint)
        if abs(geo["nose_offset_x"]) > 0.25:
            return False, None, "Face not centered. Please look straight at the camera."

        # Nose should be below eyes
        if geo["nose_to_eyes_y"] < 0:
            return False, None, "Face orientation unclear. Please face the camera directly."

        logger.info(f"Baseline captured: nose_offset={geo['nose_offset_x']:.3f}, "
                    f"nose_to_eyes={geo['nose_to_eyes_y']:.3f}, "
                    f"smile_ratio={geo['smile_ratio']:.3f}")

        return True, landmarks, "Baseline captured successfully"

    def _get_check_function(self, challenge: str):
        """Map challenge name to validation function."""
        mapping = {
            LivenessChallenge.LOOK_LEFT.value: self._check_look_left,
            LivenessChallenge.LOOK_RIGHT.value: self._check_look_right,
            LivenessChallenge.SMILE.value: self._check_smile,
            LivenessChallenge.OPEN_MOUTH.value: self._check_open_mouth,
            LivenessChallenge.LOOK_UP.value: self._check_look_up,
            LivenessChallenge.LOOK_DOWN.value: self._check_look_down,
        }
        return mapping.get(challenge)

    # ==========================================
    # Individual Challenge Checks
    # Each returns (passed: bool, score: float)
    # ==========================================

    def _check_look_left(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Head turn LEFT: nose shifts toward the RIGHT side of the image
        relative to eye midpoint (from the user's perspective, turning left
        means the nose moves to camera's right).

        With 5-point landmarks, when the user turns left:
        - nose_offset_x decreases (nose moves left relative to eyes)
        """
        delta = baseline["nose_offset_x"] - current["nose_offset_x"]
        score = min(1.0, max(0.0, delta / (self.TURN_THRESHOLD * 2)))
        passed = delta > self.TURN_THRESHOLD
        return passed, score

    def _check_look_right(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Head turn RIGHT: nose shifts toward the LEFT side of the image
        (from user's perspective, turning right means nose moves to camera's left).

        nose_offset_x increases.
        """
        delta = current["nose_offset_x"] - baseline["nose_offset_x"]
        score = min(1.0, max(0.0, delta / (self.TURN_THRESHOLD * 2)))
        passed = delta > self.TURN_THRESHOLD
        return passed, score

    def _check_smile(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Smile: mouth width increases relative to baseline.
        """
        if baseline["smile_ratio"] < 1e-6:
            return False, 0.0

        ratio = current["smile_ratio"] / baseline["smile_ratio"]
        score = min(1.0, max(0.0, (ratio - 1.0) / (self.SMILE_THRESHOLD - 1.0)))
        passed = ratio >= self.SMILE_THRESHOLD
        return passed, score

    def _check_open_mouth(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Open mouth: the vertical distance between nose and mouth center increases,
        and/or mouth vertical spread changes.

        We use nose_to_mouth_y as the primary metric.
        """
        if baseline["nose_to_mouth_y"] < 1e-6:
            return False, 0.0

        ratio = current["nose_to_mouth_y"] / baseline["nose_to_mouth_y"]
        score = min(1.0, max(0.0, (ratio - 1.0) / (self.OPEN_MOUTH_THRESHOLD - 1.0)))
        passed = ratio >= self.OPEN_MOUTH_THRESHOLD
        return passed, score

    def _check_look_up(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Look UP: nose-to-eyes vertical distance DECREASES
        (nose moves closer to eyes).
        """
        if baseline["nose_to_eyes_y"] < 1e-6:
            return False, 0.0

        delta = baseline["nose_to_eyes_y"] - current["nose_to_eyes_y"]
        score = min(1.0, max(0.0, delta / self.TILT_UP_THRESHOLD))
        passed = delta > self.TILT_UP_THRESHOLD
        return passed, score

    def _check_look_down(self, current: Dict, baseline: Dict) -> Tuple[bool, float]:
        """
        Look DOWN: nose-to-eyes vertical distance INCREASES
        (nose moves farther from eyes).
        """
        if baseline["nose_to_eyes_y"] < 1e-6:
            return False, 0.0

        delta = current["nose_to_eyes_y"] - baseline["nose_to_eyes_y"]
        score = min(1.0, max(0.0, delta / self.TILT_DOWN_THRESHOLD))
        passed = delta > self.TILT_DOWN_THRESHOLD
        return passed, score
