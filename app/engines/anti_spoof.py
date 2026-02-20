"""
Anti-Spoofing Module
====================
Texture-based liveness detection using OpenCV.
Works with any face engine — no additional dependencies required.

Techniques used:
1. Laplacian Variance — Detects blurriness (printed photos/screens are often less sharp)
2. Color Distribution — Screens emit unnatural color distributions
3. Moiré Pattern Detection — Detects screen capture artifacts
4. Edge Density Analysis — Real faces have richer edge patterns
5. Specular Reflection — Screens often show specular highlights

Each check returns a score (0.0 = likely spoof, 1.0 = likely real).
The final score is a weighted average.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from app.extensions import logger


@dataclass
class LivenessResult:
    """Result from anti-spoofing check."""
    is_live: bool
    score: float  # 0.0 (spoof) to 1.0 (live)
    details: dict  # Per-check scores
    message: Optional[str] = None


class AntiSpoofChecker:
    """
    Texture-based anti-spoofing using image quality analysis.
    
    This is a passive liveness check — it analyzes a single image
    without requiring user interaction (blink, head turn, etc.).
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Minimum liveness score to consider as live (0.0-1.0).
                       Higher = stricter. Default: 0.5
        """
        self.threshold = threshold

    def check(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> LivenessResult:
        """
        Perform anti-spoofing analysis on a face image.
        """
        try:
            # Crop to face region if bbox provided
            if face_bbox is not None:
                x, y, w, h = [int(v) for v in face_bbox]
                # Add padding around face (20%)
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                face_roi = image[y1:y2, x1:x2]
            else:
                face_roi = image

            if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                return LivenessResult(
                    is_live=False, score=0.0,
                    details={}, message="Face region too small or empty"
                )

            # Run all checks
            sharpness = self._check_sharpness(face_roi)
            color = self._check_color_distribution(face_roi)
            moire = self._check_moire_pattern(face_roi)
            edges = self._check_edge_density(face_roi)
            reflection = self._check_specular_reflection(face_roi)

            # Weighted average
            weights = {
                "sharpness": 0.30,
                "color": 0.25,
                "moire": 0.15,
                "edges": 0.15,
                "reflection": 0.15,
            }

            scores = {
                "sharpness": sharpness,
                "color": color,
                "moire": moire,
                "edges": edges,
                "reflection": reflection,
            }

            final_score = sum(scores[k] * weights[k] for k in weights)
            final_score = round(final_score, 4)

            is_live = final_score >= self.threshold

            message = None
            if not is_live:
                weakest = min(scores, key=scores.get)
                reasons = {
                    "sharpness": "Image appears too blurry or unnaturally sharp",
                    "color": "Unnatural color distribution detected",
                    "moire": "Moiré pattern detected",
                    "edges": "Unusual edge pattern",
                    "reflection": "Specular reflection detected",
                }
                message = f"Spoof detected: {reasons.get(weakest, 'Low liveness score')}"

            logger.info(f"Anti-spoof check: score={final_score:.3f}, live={is_live}")

            return LivenessResult(
                is_live=is_live,
                score=final_score,
                details={k: round(v, 4) for k, v in scores.items()},
                message=message,
            )
        except Exception as e:
            logger.error(f"Internal error in AntiSpoofChecker: {e}")
            return LivenessResult(
                is_live=False, 
                score=0.0, 
                details={}, 
                message=f"Liveness check error: {str(e)}"
            )

    def _check_sharpness(self, face_roi: np.ndarray) -> float:
        """
        Check image sharpness using Laplacian variance.
        
        Real faces captured by camera have natural sharpness.
        Printed photos or screens are often blurrier or have
        unnatural sharpness patterns.
        
        Returns: 0.0 (blurry/fake) to 1.0 (naturally sharp)
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Typical ranges:
        # - Printed photo: 10-80
        # - Screen display: 50-200
        # - Real face (camera): 80-500+
        # Score: penalize both too low (print) and extremely high (edited)
        if laplacian_var < 15:
            return 0.1  # Very blurry — likely a print
        elif laplacian_var < 50:
            return 0.3
        elif laplacian_var < 100:
            return 0.6
        elif laplacian_var < 800:
            return 1.0  # Natural range
        else:
            return 0.7  # Unusually sharp — possible post-processing

    def _check_color_distribution(self, face_roi: np.ndarray) -> float:
        """
        Analyze color channel distribution.
        
        Screens emit light differently than natural scenes:
        - Screen: narrow color peaks, high saturation uniformity
        - Real: broader color variance, natural skin tone gradients
        
        Returns: 0.0 (unnatural) to 1.0 (natural)
        """
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Check saturation variance (screens have more uniform saturation)
        sat_std = float(np.std(s))
        # Check value (brightness) variance
        val_std = float(np.std(v))
        # Check hue variance
        hue_std = float(np.std(h))

        score = 0.0

        # Saturation should have natural variance (not too uniform, not too wild)
        if 15 < sat_std < 60:
            score += 0.4
        elif 10 < sat_std < 80:
            score += 0.2

        # Brightness should have moderate variance
        if 20 < val_std < 70:
            score += 0.3
        elif 10 < val_std < 90:
            score += 0.15

        # Hue should have some variance (skin is not perfectly uniform)
        if 5 < hue_std < 30:
            score += 0.3
        elif 3 < hue_std < 50:
            score += 0.15

        return min(1.0, score)

    def _check_moire_pattern(self, face_roi: np.ndarray) -> float:
        """
        Detect moiré patterns that appear when photographing screens.
        
        Uses frequency domain analysis (FFT) to find repetitive
        high-frequency patterns typical of screen capture.
        
        Returns: 0.0 (moiré detected) to 1.0 (clean)
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        target_size = 128
        gray = cv2.resize(gray, (target_size, target_size))

        # Apply FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Log magnitude spectrum
        magnitude = np.log1p(magnitude)

        # Analyze high-frequency content
        center = target_size // 2
        # High-frequency ring (outer region)
        mask = np.zeros_like(magnitude, dtype=bool)
        y, x = np.ogrid[:target_size, :target_size]
        outer_ring = ((x - center) ** 2 + (y - center) ** 2) > (center * 0.6) ** 2
        mask[outer_ring] = True

        high_freq_energy = float(np.mean(magnitude[mask]))
        total_energy = float(np.mean(magnitude))

        if total_energy == 0:
            return 0.5

        ratio = high_freq_energy / total_energy

        # High ratio = lots of high-freq patterns = possible moiré
        if ratio > 0.85:
            return 0.2  # Strong moiré pattern
        elif ratio > 0.75:
            return 0.5
        elif ratio > 0.65:
            return 0.7
        else:
            return 1.0  # Clean image

    def _check_edge_density(self, face_roi: np.ndarray) -> float:
        """
        Analyze edge density and distribution.
        
        Real faces have complex, organic edge patterns.
        Printed photos may have different edge characteristics
        (paper texture, print dots).
        
        Returns: 0.0 (suspicious) to 1.0 (natural)
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Edge density (percentage of edge pixels)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # Real faces typically have moderate edge density
        if 0.03 < edge_density < 0.20:
            return 1.0  # Natural range
        elif 0.02 < edge_density < 0.25:
            return 0.6
        elif edge_density < 0.02:
            return 0.3  # Too smooth (possible blurred print)
        else:
            return 0.4  # Too many edges (possible screen artifacts)

    def _check_specular_reflection(self, face_roi: np.ndarray) -> float:
        """
        Detect specular reflections typical of screen glare.
        
        Screens often produce bright specular highlights that
        don't appear on real faces in normal conditions.
        
        Returns: 0.0 (screen glare detected) to 1.0 (natural)
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Find very bright spots (potential screen reflections)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.count_nonzero(bright_mask)) / gray.size

        # Check for large contiguous bright areas (screen glare)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_bright_areas = 0
        if contours:
            face_area = gray.shape[0] * gray.shape[1]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > face_area * 0.01:  # >1% of face area
                    large_bright_areas += 1

        if large_bright_areas > 2:
            return 0.2  # Multiple large bright areas = screen
        elif large_bright_areas > 0:
            return 0.5
        elif bright_ratio > 0.05:
            return 0.6
        else:
            return 1.0  # Normal brightness distribution
