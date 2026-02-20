"""
Engines Package
===============
Face recognition engine abstraction layer.

Supported engines:
- opencv     : OpenCV DNN (YuNet + SFace) — lightweight, ~150MB RAM
- insightface: InsightFace (ArcFace)      — high accuracy, ~500MB RAM

Configure via FACE_ENGINE environment variable (default: opencv).
"""

import os
from typing import Optional

from app.engines.base import FaceEngineBase, DetectionResult, ComparisonResult

# Singleton engine instance
_engine_instance: Optional[FaceEngineBase] = None


def get_engine() -> FaceEngineBase:
    """
    Get the configured face engine (singleton).
    
    Engine is selected via the FACE_ENGINE environment variable:
    - 'opencv'      → OpenCV + SFace (default)
    - 'insightface' → InsightFace + ArcFace
    
    Returns:
        Initialized FaceEngineBase instance.
    """
    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    engine_name = os.getenv("FACE_ENGINE", "opencv").lower().strip()

    if engine_name == "opencv":
        from app.engines.opencv_engine import OpenCVEngine
        _engine_instance = OpenCVEngine()

    elif engine_name == "insightface":
        from app.engines.insightface_engine import InsightFaceEngine
        model = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
        _engine_instance = InsightFaceEngine(model_name=model)

    else:
        raise ValueError(
            f"Unknown face engine: '{engine_name}'. "
            f"Supported: opencv, insightface"
        )

    # Initialize (lazy — will load models on first use)
    from app.extensions import logger
    logger.info(f"Face engine configured: {_engine_instance.name()}")

    return _engine_instance


def reset_engine():
    """Reset the singleton engine (useful for testing)."""
    global _engine_instance
    _engine_instance = None


__all__ = [
    "FaceEngineBase",
    "DetectionResult",
    "ComparisonResult",
    "get_engine",
    "reset_engine",
]

# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == "LivenessResult":
        from app.engines.anti_spoof import LivenessResult
        return LivenessResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
