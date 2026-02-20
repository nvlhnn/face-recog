"""
Face Recognition Flask API
==========================
Application factory and initialization
"""

from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from app.config import Config
from app.extensions import db
from app.api import register_blueprints


def create_app(config_class=Config):
    """Application factory pattern."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    CORS(app)
    init_swagger(app)
    
    # Initialize SQLite database
    db.init_app(app)
    
    # Register blueprints (routes)
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app


def init_swagger(app):
    """Initialize Swagger documentation."""
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }
    
    swagger_template = {
        "info": {
            "title": "Face Recognition API",
            "description": "REST API for face registration, verification, and user management. Supports multiple engines: OpenCV, InsightFace.",
            "version": "2.0.0",
        },
        "basePath": "/",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "Enter: **Bearer &lt;your-token&gt;**"
            }
        },
        "security": [
            {"Bearer": []}
        ],
        "tags": [
            {"name": "Health", "description": "Health check endpoints"},
            {"name": "Face", "description": "Face registration and verification"},
            {"name": "Users", "description": "User management"}
        ]
    }
    
    Swagger(app, config=swagger_config, template=swagger_template)


def register_error_handlers(app):
    """Register error handlers."""
    from flask import jsonify
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "error": "Endpoint not found"}), 404
    
    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"status": "error", "error": "Method not allowed"}), 405
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"status": "error", "error": "Internal server error"}), 500
