"""
Users API
=========
User management endpoints
"""

from flask import Blueprint, request, jsonify, current_app

from app.services import FaceService
from app.extensions import require_auth, rate_limit

users_bp = Blueprint('users', __name__)


def get_face_service() -> FaceService:
    """Get FaceService instance."""
    return FaceService()


@users_bp.route('/users', methods=['GET'])
@require_auth
@rate_limit
def list_users():
    """List Users
    ---
    tags:
      - Users
    summary: List all registered users with pagination
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
      - name: page
        in: query
        type: integer
        required: false
        default: 1
        description: Page number (default 1)
    responses:
      200:
        description: List of users with pagination info
      401:
        description: Missing access token
      429:
        description: Rate limit exceeded
      500:
        description: Server error
    """
    page = request.args.get('page', 1, type=int)
    per_page = current_app.config.get('ITEMS_PER_PAGE', 10)
    
    try:
        service = get_face_service()
        result = service.list_users(page, per_page)
        
        return jsonify({
            "status": "success",
            **result
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@users_bp.route('/users/<user_id>', methods=['GET'])
@require_auth
@rate_limit
def get_user(user_id):
    """Get User
    ---
    tags:
      - Users
    summary: Get information about a specific registered user
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
      - name: user_id
        in: path
        type: string
        required: true
        description: User ID to look up
    responses:
      200:
        description: User found
      404:
        description: User not found
      429:
        description: Rate limit exceeded
      500:
        description: Server error
    """
    try:
        service = get_face_service()
        user = service.get_user(user_id)
        
        if user:
            return jsonify({
                "status": "success",
                "user": user
            }), 200
        else:
            return jsonify({
                "status": "fail",
                "message": "User not found"
            }), 404
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
