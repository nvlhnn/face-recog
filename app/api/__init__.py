"""
API Package
===========
API routes and controllers
"""

from flask import Flask


def register_blueprints(app: Flask):
    """Register all API blueprints."""
    from app.api.health import health_bp
    from app.api.face import face_bp
    from app.api.users import users_bp
    from app.api.liveness import liveness_bp
    
    app.register_blueprint(health_bp)
    app.register_blueprint(face_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(liveness_bp)
