"""
Face API
========
Face registration, verification, and deletion endpoints.
Following Hexagonal Architecture / Clean Architecture.
"""

from flask import Blueprint, request, jsonify, current_app
from app.application import get_face_use_cases
from app.utils.image_utils import ImageProcessor, validate_image_file, _is_anti_spoofing_enabled
from app.extensions import require_auth, rate_limit

face_bp = Blueprint('face', __name__)


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
    
    # 1. Validate inputs
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    try:
        # 2. Decode Image (Converting interface format to domain format np.ndarray)
        image = ImageProcessor._decode_image(file)
        if image is None:
            return jsonify({"status": "error", "error": "Could not decode image"}), 400

        # 3. Call Use Case
        use_cases = get_face_use_cases()
        anti_spoof = _is_anti_spoofing_enabled()
        
        success, message = use_cases.register_user(user_id, image, anti_spoof)
        
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
    
    # 1. Validate inputs
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    if tolerance is not None:
        tolerance = max(0.3, min(0.7, tolerance))
    
    try:
        # 2. Decode Image
        image = ImageProcessor._decode_image(file)
        if image is None:
            return jsonify({"status": "error", "error": "Could not decode image"}), 400

        # 3. Call Use Case
        use_cases = get_face_use_cases()
        anti_spoof = _is_anti_spoofing_enabled()
        
        result = use_cases.verify_user(user_id, image, anti_spoof, tolerance)
        
        if not result.get("matched") and "not found" in result.get("message", ""):
            return jsonify({
                "status": "fail",
                "message": result.get("message")
            }), 404
        
        if not result.get("matched") and "No face" in result.get("message", ""):
            return jsonify({
                "status": "fail",
                "message": result.get("message")
            }), 400
        
        return jsonify({"status": "success", **result}), 200
        
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
    parameters:
      - name: user_id
        in: formData
        type: string
        required: true
    responses:
      200:
        description: Deleted
      404:
        description: Not found
    """
    if request.is_json:
        user_id = request.json.get('user_id')
    else:
        user_id = request.form.get('user_id')
    
    if not user_id:
        return jsonify({"status": "error", "error": "Missing user_id"}), 400
    
    try:
        use_cases = get_face_use_cases()
        success = use_cases.delete_user(user_id)
        
        if success:
            return jsonify({"status": "success", "message": "Face data deleted", "user_id": user_id}), 200
        return jsonify({"status": "fail", "message": "User not found"}), 404
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@face_bp.route('/analyze', methods=['POST'])
@require_auth
@rate_limit
def analyze_face():
    """Analyze Face Attributes
    ---
    tags:
      - Face
    summary: Analyze face attributes (engine-dependent)
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Analysis completed
    """
    file = request.files.get('image')
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    try:
        image = ImageProcessor._decode_image(file)
        if image is None:
            return jsonify({"status": "error", "error": "Could not decode image"}), 400

        use_cases = get_face_use_cases()
        result = use_cases.analyze_face(image)
        
        if result.get("face_detected"):
            return jsonify({"status": "success", **result}), 200
        return jsonify({"status": "fail", **result}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@face_bp.route('/liveness', methods=['POST'])
@require_auth
@rate_limit
def check_liveness():
    """Check Liveness (Anti-Spoofing)
    ---
    tags:
      - Face
    summary: Perform anti-spoofing / liveness detection
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Result
    """
    file = request.files.get('image')
    
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return jsonify({"status": "error", "error": error_msg}), 400
    
    try:
        image = ImageProcessor._decode_image(file)
        if image is None:
            return jsonify({"status": "error", "error": "Could not decode image"}), 400

        use_cases = get_face_use_cases()
        # Direct engine call via use case logic
        detection = use_cases.engine.detect_face(image)
        if not detection.face_found:
            return jsonify({"status": "fail", "face_detected": False, "message": "No face detected"}), 400
            
        liveness = use_cases.engine.liveness_check(image, detection)
        
        return jsonify({
            "status": "success",
            "engine": use_cases.engine.name(),
            "face_detected": True,
            "is_live": liveness.is_live,
            "liveness_score": round(liveness.score, 4),
            "details": liveness.details,
            "message": liveness.message
        }), 200
            
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
