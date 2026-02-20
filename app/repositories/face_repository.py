"""
Face Repository
===============
SQLite Data Access for face encodings.
"""

import json
import numpy as np
from typing import Optional, Dict
from flask import current_app
from app.core.entities import User
from app.core.ports import FaceRepositoryPort
from app.extensions import db, logger


class FaceRepository(FaceRepositoryPort):
    
    @property
    def engine(self) -> str:
        """Get the current active engine name from config."""
        return current_app.config.get('FACE_ENGINE', 'opencv')

    def exists(self, user_id: str) -> bool:
        with db.get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? AND engine = ?", 
                (user_id, self.engine)
            ).fetchone()
            return result is not None

    def save(self, user: User):
        """Save face encoding to database for the specific engine."""
        encoding_json = json.dumps(user.encoding.tolist())
        
        with db.get_connection() as conn:
            if self.exists(user.user_id):
                conn.execute(
                    "UPDATE users SET encoding = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE user_id = ? AND engine = ?",
                    (encoding_json, user.user_id, user.engine)
                )
            else:
                conn.execute(
                    "INSERT INTO users (user_id, engine, encoding) VALUES (?, ?, ?)",
                    (user.user_id, user.engine, encoding_json)
                )
            conn.commit()

    def get_by_id(self, user_id: str, engine: str) -> Optional[User]:
        """Retrieve User entity from database."""
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ? AND engine = ?",
                (user_id, engine)
            ).fetchone()
            
            if row:
                return User(
                    user_id=row['user_id'],
                    engine=row['engine'],
                    encoding=np.array(json.loads(row['encoding'])),
                    created_at=str(row['created_at']),
                    updated_at=str(row['updated_at'])
                )
            return None

    def get_user_info(self, user_id: str) -> Optional[dict]:
        """Get user info without encoding for the current engine."""
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT user_id, created_at, updated_at FROM users WHERE user_id = ? AND engine = ?",
                (user_id, self.engine)
            ).fetchone()
            
            if row:
                return {
                    "user_id": row['user_id'],
                    "engine": self.engine,
                    "created_at": str(row['created_at']),
                    "updated_at": str(row['updated_at'])
                }
            return None

    def delete(self, user_id: str) -> bool:
        """Delete user and engine-specific encoding."""
        with db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM users WHERE user_id = ? AND engine = ?", 
                (user_id, self.engine)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_paginated(self, engine: str, page: int, per_page: int) -> Dict:
        """List all users registered with the specified engine."""
        with db.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM users WHERE engine = ?", (engine,)
            ).fetchone()[0]
            
            offset = (page - 1) * per_page
            rows = conn.execute(
                "SELECT user_id, created_at FROM users WHERE engine = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?", 
                (engine, per_page, offset)
            ).fetchall()
            
            data = [{"user_id": r['user_id'], "created_at": str(r['created_at'])} for r in rows]
            
            return {
                "users": data,
                "total": total,
                "engine": engine,
                "page": page,
                "per_page": per_page,
                "pages": (total + per_page - 1) // per_page
            }
