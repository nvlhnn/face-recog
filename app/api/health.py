"""
Health API
==========
Health check endpoints
"""

import os
from flask import Blueprint, jsonify
from datetime import datetime

from app.extensions import db
from app.engines import get_engine

health_bp = Blueprint('health', __name__)


@health_bp.route('/', methods=['GET'])
def home():
    """API Information
    ---
    tags:
      - Health
    summary: Get API information and available endpoints
    responses:
      200:
        description: API is running
    """
    anti_spoofing = os.getenv('ANTI_SPOOFING', 'false').lower() in ('true', '1', 'yes')
    return jsonify({
        "status": "running",
        "service": "Face Recognition API",
        "version": "2.0.0",
        "engine": get_engine().name(),
        "anti_spoofing": anti_spoofing,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "register": "POST /register - Register a new face",
            "verify": "POST /verify - Verify a face against a user",
            "delete": "POST /delete - Delete a user's face data",
            "analyze": "POST /analyze - Analyze face attributes (response varies by engine)",
            "liveness": "POST /liveness - Anti-spoofing / liveness check",
            "list": "GET /users - List all registered users",
            "health": "GET /health - Health check",
            "docs": "GET /apidocs - API documentation"
        }
    }), 200


@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health Check
    ---
    tags:
      - Health
    summary: Check API and database connectivity status
    responses:
      200:
        description: All systems healthy
      503:
        description: Database unhealthy
    """
    anti_spoofing = os.getenv('ANTI_SPOOFING', 'false').lower() in ('true', '1', 'yes')
    engine = get_engine()
    health = {
        "api": "healthy",
        "database": "unknown",
        "engine": engine.name(),
        "anti_spoofing": anti_spoofing,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
        health["database"] = "healthy"
    except Exception as e:
        health["database"] = f"unhealthy: {str(e)}"
        return jsonify(health), 503
    
    return jsonify(health), 200
