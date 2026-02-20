"""
Extensions Module
=================
SQLite database, authentication, and utilities
"""

import logging
import sqlite3
import os
import threading
from functools import wraps
from contextlib import contextmanager
from flask import request, jsonify, current_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================
# Authentication
# ===================

def require_auth(f):
    """Decorator to require ACCESS_TOKEN authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = current_app.config.get('ACCESS_TOKEN')
        
        # If no token configured, allow all requests
        if not token:
            return f(*args, **kwargs)
        
        # Check Authorization header
        auth_header = request.headers.get('Authorization', '')
        
        if auth_header.startswith('Bearer '):
            provided_token = auth_header[7:]
        else:
            # Also check X-Access-Token header
            provided_token = request.headers.get('X-Access-Token', '')
        
        if not provided_token:
            return jsonify({
                "status": "error",
                "error": "Missing access token. Use 'Authorization: Bearer <token>' header."
            }), 401
        
        if provided_token != token:
            logger.warning(f"Invalid access token attempt from {request.remote_addr}")
            return jsonify({
                "status": "error", 
                "error": "Invalid access token"
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function


# ===================
# Rate Limiting
# ===================

# Simple in-memory rate limiter
_rate_limit_store = {}
_rate_limit_lock = threading.Lock()

def check_rate_limit(ip: str, limit: int = 100, window: int = 60) -> bool:
    """
    Check if IP is within rate limit.
    Returns True if allowed, False if rate limited.
    """
    import time
    current_time = time.time()
    
    with _rate_limit_lock:
        if ip not in _rate_limit_store:
            _rate_limit_store[ip] = []
        
        # Clean old entries
        _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if current_time - t < window]
        
        if len(_rate_limit_store[ip]) >= limit:
            return False
        
        _rate_limit_store[ip].append(current_time)
        return True


def rate_limit(f):
    """Decorator to apply rate limiting."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Parse rate limit from config (e.g., "100 per minute")
        rate_config = current_app.config.get('RATE_LIMIT', '100 per minute')
        try:
            parts = rate_config.split()
            limit = int(parts[0])
            # Default to 60 seconds window
            window = 60
            if 'hour' in rate_config.lower():
                window = 3600
            elif 'day' in rate_config.lower():
                window = 86400
        except:
            limit, window = 100, 60
        
        ip = request.remote_addr
        if not check_rate_limit(ip, limit, window):
            logger.warning(f"Rate limit exceeded for {ip}")
            return jsonify({
                "status": "error",
                "error": "Rate limit exceeded. Please try again later."
            }), 429
        
        return f(*args, **kwargs)
    return decorated_function


# ===================
# Database
# ===================

class Database:
    def __init__(self):
        self.db_path = None
        self._local = threading.local()
    
    def init_app(self, app):
        self.db_path = app.config['DB_PATH']
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self._create_tables()
        logger.info(f"SQLite DB ready at: {self.db_path}")
    
    def _create_tables(self):
        with self.get_connection() as conn:
            # Users Table with encoding storage and engine tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT,
                    engine TEXT,
                    encoding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, engine)
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_engine ON users(engine)')
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e


db = Database()
