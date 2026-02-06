"""
Face API
========
Face registration, verification, and deletion endpoints
"""

from flask import Blueprint, request, jsonify, current_app

from app.services import FaceService
from app.utils import validate_image_file
from app.extensions import require_auth, rate_limit

face_bp = Blueprint('face', __name__)


def get_face_service() -> FaceService:
    """Get FaceService instance."""
    return FaceService()


@face_bp.route('/register', methods=['POST'])
@require_auth
@rate_limit
def register_face():
    """Register Face
    ---
    tags:
      - Face
    summary: Register a new face for a user
    description: Register a new face for a user. If user_id already exists, the face encoding will be updated.
    consumes:
      - multipart/form-data
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
      - name: user_id
        in: formData
        type: string
        required: true
        description: Unique identifier for the user
      - name: image
        in: formData
        type: file
        required: true
        description: Image file containing the face (jpg, png, gif, bmp)
    responses:
      201:
        description: Face registered successfully
      400:
        description: Bad request - missing user_id, invalid image, or no face detected
      401:
        description: Missing access token
      403:
        description: Invalid access token
      429:
        description: Rate limit exceeded
      500:
        description: Server error
    """
    user_id = request.form.get('user_id')
    file = request.files.get('image')
    
    # Validate inputs
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    try:
        service = get_face_service()
        success, message = service.register_face(user_id, file)
        
        if success:
            return jsonify({
                "status": "success",
                "message": message,
                "user_id": user_id
            }), 201
        else:
            return jsonify({
                "status": "error",
                "error": message
            }), 400
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@face_bp.route('/verify', methods=['POST'])
@require_auth
@rate_limit
def verify_face():
    """Verify Face
    ---
    tags:
      - Face
    summary: Verify a face against a specific registered user
    consumes:
      - multipart/form-data
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
      - name: user_id
        in: formData
        type: string
        required: true
        description: User ID to verify against
      - name: image
        in: formData
        type: file
        required: true
        description: Image file containing the face to verify
      - name: tolerance
        in: formData
        type: number
        required: false
        description: Match tolerance (0.3-0.7, default 0.6). Lower is stricter.
    responses:
      200:
        description: Verification completed. Returns matched, distance, and confidence.
      400:
        description: Bad request - missing user_id, invalid image, or no face detected
      404:
        description: User ID not found
      429:
        description: Rate limit exceeded
      500:
        description: Server error
    """
    user_id = request.form.get('user_id')
    file = request.files.get('image')
    tolerance = request.form.get('tolerance', type=float)
    
    # Validate inputs
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    # Validate tolerance range
    if tolerance is not None:
        tolerance = max(0.3, min(0.7, tolerance))
    
    try:
        service = get_face_service()
        success, result = service.verify_face(user_id, file, tolerance)
        
        if not success and result.message and "not found" in result.message:
            return jsonify({
                "status": "fail",
                "message": result.message
            }), 404
        
        if not success and result.message and "No face" in result.message:
            return jsonify({
                "status": "fail",
                "message": result.message
            }), 400
        
        return jsonify(result.to_dict()), 200
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@face_bp.route('/delete', methods=['POST'])
@require_auth
@rate_limit
def delete_face():
    """Delete Face
    ---
    tags:
      - Face
    summary: Delete a user's face data from the system
    consumes:
      - multipart/form-data
      - application/json
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
      - name: user_id
        in: formData
        type: string
        required: true
        description: User ID to delete
    responses:
      200:
        description: Face data deleted successfully
      400:
        description: Missing user_id
      404:
        description: User ID not found
      429:
        description: Rate limit exceeded
      500:
        description: Server error
    """
    # Accept both form data and JSON
    if request.is_json:
        user_id = request.json.get('user_id')
    else:
        user_id = request.form.get('user_id')
    
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    try:
        service = get_face_service()
        success, message = service.delete_face(user_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": message,
                "user_id": user_id
            }), 200
        else:
            return jsonify({
                "status": "fail",
                "message": message
            }), 404
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
