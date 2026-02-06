"""
Face Service
============
Business logic for encoding-based face recognition.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from app.extensions import logger, db
from app.repositories import FaceRepository
from app.utils.image_utils import ImageProcessor


@dataclass
class VerificationResult:
    matched: bool
    user_id: str
    distance: float
    confidence: float = 0.0
    message: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            'status': 'success',
            'matched': self.matched,
            'user_id': self.user_id,
            'distance': self.distance,
            'confidence': round(self.confidence, 1)
        }
        if self.message:
            result['message'] = self.message
        return result


class FaceService:
    def __init__(self):
        self.repository = FaceRepository()
        self.tolerance = 0.363  # SFace cosine similarity threshold

    def register_face(self, user_id: str, image_file) -> Tuple[bool, str]:
        """
        Extract face encoding and store in database.
        """
        try:
            # 1. Extract encoding from image
            encoding = ImageProcessor.get_face_encoding(image_file)
            
            if encoding is None:
                return False, "No face detected in image"
            
            # 2. Store encoding in database
            existing = self.repository.exists(user_id)
            self.repository.save_encoding(user_id, encoding)
            
            msg = "Face updated" if existing else "Face registered"
            logger.info(f"{msg} for user {user_id}")
            return True, msg
            
        except ValueError as e:
            # Spoof detection
            return False, str(e)
        except Exception as e:
            logger.error(f"Register error: {e}")
            return False, str(e)

    def verify_face(self, user_id: str, image_file, tolerance: float = None) -> Tuple[bool, VerificationResult]:
        """
        Compare uploaded image encoding against stored encoding.
        """
        tolerance = tolerance or self.tolerance
        
        try:
            # 1. Get stored encoding
            stored_encoding = self.repository.get_encoding(user_id)
            
            if stored_encoding is None:
                return False, VerificationResult(
                    matched=False,
                    user_id=user_id,
                    distance=1.0,
                    confidence=0.0,
                    message="User ID not found"
                )
            
            # 2. Extract encoding from uploaded image
            unknown_encoding = ImageProcessor.get_face_encoding(image_file)
            
            if unknown_encoding is None:
                return False, VerificationResult(
                    matched=False,
                    user_id=user_id,
                    distance=0.0,
                    confidence=0.0,
                    message="No face detected in uploaded image"
                )
            
            # 3. Compare encodings (now returns confidence too)
            is_match, distance, confidence = ImageProcessor.compare_encodings(
                stored_encoding, 
                unknown_encoding, 
                tolerance
            )
            
            # 4. Log attendance if matched
            if is_match:
                self.repository.log_attendance(user_id, "check_in", distance)
            
            return True, VerificationResult(
                matched=is_match,
                user_id=user_id,
                distance=round(distance, 4),
                confidence=confidence
            )
            
        except ValueError as e:
            # Spoof detection
            return False, VerificationResult(
                matched=False,
                user_id=user_id,
                distance=0.0,
                confidence=0.0,
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Verify error: {e}")
            return False, VerificationResult(
                matched=False,
                user_id=user_id,
                distance=0.0,
                confidence=0.0,
                message=f"Error: {str(e)}"
            )

    def delete_face(self, user_id: str) -> Tuple[bool, str]:
        """Delete user's face encoding from database."""
        deleted = self.repository.delete(user_id)
        if deleted:
            return True, "Face data deleted"
        return False, "User not found"
        
    def list_users(self, page, per_page):
        return self.repository.find_all_paginated(page, per_page)

    def get_user(self, user_id: str):
        """Get user info (without encoding)."""
        return self.repository.get_user_info(user_id)
