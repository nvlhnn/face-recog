"""
Application Layer
=================
Use cases and factory functions.
"""

from app.application.use_cases import FaceRecognitionUseCases
from app.engines import get_engine
from app.repositories.face_repository import FaceRepository


def get_face_use_cases() -> FaceRecognitionUseCases:
    """Dependency injection factory for face use cases."""
    engine = get_engine()
    repository = FaceRepository()
    return FaceRecognitionUseCases(engine=engine, repository=repository)
