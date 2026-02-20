"""
Users API
=========
User management endpoints.
Following Hexagonal Architecture / Clean Architecture.
"""

from flask import Blueprint, request, jsonify, current_app
from app.application import get_face_use_cases
from app.extensions import require_auth, rate_limit

users_bp = Blueprint('users', __name__)


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
      - name: page
        in: query
        type: integer
        required: false
        default: 1
    responses:
      200:
        description: List of users with pagination info
    """
    page = request.args.get('page', 1, type=int)
    per_page = current_app.config.get('ITEMS_PER_PAGE', 10)
    
    try:
        use_cases = get_face_use_cases()
        result = use_cases.list_users(page, per_page)
        
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
      - name: user_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: User found
      404:
        description: Not found
    """
    try:
        use_cases = get_face_use_cases()
        user = use_cases.get_user(user_id)
        
        if user:
            return jsonify({
                "status": "success",
                "user": {
                    "user_id": user.user_id,
                    "engine": user.engine,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at
                }
            }), 200
        else:
            return jsonify({
                "status": "fail",
                "message": "User not found"
            }), 404
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
