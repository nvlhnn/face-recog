"""
Face Repository
===============
SQLite Data Access for face encodings.
"""

import json
import numpy as np
from typing import Optional, List, Tuple
from app.extensions import db, logger


class FaceRepository:
    
    def exists(self, user_id: str) -> bool:
        with db.get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?", 
                (user_id,)
            ).fetchone()
            return result is not None

    def save_encoding(self, user_id: str, encoding: np.ndarray):
        """Save face encoding to database (as JSON string)."""
        encoding_json = json.dumps(encoding.tolist())
        
        with db.get_connection() as conn:
            if self.exists(user_id):
                conn.execute(
                    "UPDATE users SET encoding = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (encoding_json, user_id)
                )
            else:
                conn.execute(
                    "INSERT INTO users (user_id, encoding) VALUES (?, ?)",
                    (user_id, encoding_json)
                )
            conn.commit()

    def get_encoding(self, user_id: str) -> Optional[np.ndarray]:
        """Retrieve face encoding from database."""
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT encoding FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if row and row['encoding']:
                return np.array(json.loads(row['encoding']))
            return None

    def get_user_info(self, user_id: str) -> Optional[dict]:
        """Get user info without encoding."""
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT user_id, created_at, updated_at FROM users WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if row:
                return {
                    "user_id": row['user_id'],
                    "created_at": str(row['created_at']),
                    "updated_at": str(row['updated_at'])
                }
            return None

    def log_attendance(self, user_id: str, type: str, distance: float):
        """Record a successful verification."""
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO attendance_logs (user_id, type, confidence_score) VALUES (?, ?, ?)",
                (user_id, type, distance)
            )
            conn.commit()
            logger.info(f"Attendance logged for {user_id}")

    def delete(self, user_id: str) -> bool:
        with db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE user_id = ?", 
                (user_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def find_all_paginated(self, page: int, per_page: int):
        with db.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            offset = (page - 1) * per_page
            rows = conn.execute(
                "SELECT user_id, created_at FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", 
                (per_page, offset)
            ).fetchall()
            
            data = [{"user_id": r['user_id'], "created_at": str(r['created_at'])} for r in rows]
            
            return {
                "users": data,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page
            }
