"""
Liveness API
============
Challenge-response liveness detection endpoints.

Flow:
1. POST /liveness/start          → Get session_id + challenges
2. POST /liveness/validate       → Validate current challenge (with multi-frame images)
3. GET  /liveness/result/<id>    → Get final session result
"""

from flask import Blueprint, request, jsonify
from app.application import get_face_use_cases
from app.application.liveness_service import LivenessService
from app.utils.image_utils import ImageProcessor
from app.engines import get_engine
from app.extensions import require_auth, rate_limit, logger

liveness_bp = Blueprint('liveness', __name__, url_prefix='/liveness')


def _get_liveness_service() -> LivenessService:
    """Factory for LivenessService."""
    engine = get_engine()
    return LivenessService(engine=engine)


@liveness_bp.route('/start', methods=['POST'])
@require_auth
@rate_limit
def start_liveness():
    """Start Liveness Session
    ---
    tags:
      - Liveness
    summary: Start a new challenge-response liveness detection session
    description: |
      Creates a new liveness session and returns a set of randomized challenges
      the user must complete. The first challenge is always "look_straight" (baseline),
      followed by N random challenges (configurable via LIVENESS_CHALLENGE_COUNT).

      Flow:
      1. Call this endpoint to get session_id + challenges
      2. For each challenge, capture 3-5 frames and POST to /liveness/validate
      3. After all challenges pass, GET /liveness/result/<session_id>
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: Bearer token (if ACCESS_TOKEN is configured)
    responses:
      200:
        description: Session started successfully
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            session_id:
              type: string
              example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            challenges:
              type: array
              items:
                type: string
              example: ["look_straight", "look_left", "smile", "look_down"]
            total_steps:
              type: integer
              example: 4
            instructions:
              type: object
    """
    try:
        service = _get_liveness_service()
        session = service.start_session()

        # Human-readable instructions for each challenge
        instructions = {
            "look_straight": "Look directly at the camera",
            "look_left": "Slowly turn your head to the LEFT",
            "look_right": "Slowly turn your head to the RIGHT",
            "smile": "Give a natural smile",
            "open_mouth": "Open your mouth wide",
            "look_up": "Tilt your head UP slightly",
            "look_down": "Tilt your head DOWN slightly",
        }

        return jsonify({
            "status": "success",
            "session_id": session.session_id,
            "challenges": session.challenges,
            "total_steps": len(session.challenges),
            "instructions": {
                c: instructions.get(c, c) for c in session.challenges
            },
        }), 200

    except Exception as e:
        logger.error(f"Error starting liveness session: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@liveness_bp.route('/validate', methods=['POST'])
@require_auth
@rate_limit
def validate_liveness():
    """Validate Liveness Challenge
    ---
    tags:
      - Liveness
    summary: Submit frames to validate the current liveness challenge
    description: |
      Send multiple image frames (3-5 recommended) for the current challenge.
      The server uses multi-frame consensus to determine if the challenge is met.

      - If the challenge passes, the session advances to the next step.
      - If it fails, you can retry the same step.
      - The response always tells you what the next_challenge is.
    consumes:
      - multipart/form-data
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
      - name: session_id
        in: formData
        type: string
        required: true
        description: Session ID from /liveness/start
      - name: images
        in: formData
        type: file
        required: true
        description: Multiple image files (send 3-5 frames). Use field name "images" for each.
    responses:
      200:
        description: Challenge validated (passed or failed with retry info)
      400:
        description: Bad request - missing session_id, no valid images
      404:
        description: Session not found or expired
      500:
        description: Server error
    """
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "error": "Missing session_id"}), 400

    # Get multiple image files
    image_files = request.files.getlist('images')
    if not image_files:
        # Also try single 'image' field for backward compat
        single = request.files.get('image')
        if single:
            image_files = [single]

    if not image_files:
        return jsonify({"status": "error", "error": "No image files provided. Use field name 'images'."}), 400

    try:
        # Decode all images
        images = []
        for f in image_files:
            img = ImageProcessor._decode_image(f)
            if img is not None:
                images.append(img)

        if not images:
            return jsonify({"status": "error", "error": "Could not decode any images"}), 400

        # Validate
        service = _get_liveness_service()
        success, result = service.validate_step(session_id, images)

        if "error" in result and "not found" in result["error"].lower():
            return jsonify({"status": "error", **result}), 404

        if "error" in result:
            return jsonify({"status": "error", **result}), 400

        status_code = 200

        return jsonify({
            "status": "success" if success else "fail",
            **result,
        }), status_code

    except Exception as e:
        logger.error(f"Error validating liveness: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@liveness_bp.route('/result/<session_id>', methods=['GET'])
@require_auth
def get_liveness_result(session_id: str):
    """Get Liveness Result
    ---
    tags:
      - Liveness
    summary: Get the final result of a completed liveness session
    description: |
      After all challenges have been completed, call this endpoint
      to get the final verification result. The session is cleaned up
      after this call.
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
      - name: session_id
        in: path
        type: string
        required: true
        description: Session ID from /liveness/start
    responses:
      200:
        description: Final liveness result
        schema:
          type: object
          properties:
            verified:
              type: boolean
            overall_score:
              type: number
            challenges:
              type: array
            engine:
              type: string
      400:
        description: Session not completed yet
      404:
        description: Session not found or expired
    """
    try:
        service = _get_liveness_service()
        success, result = service.get_result(session_id)

        if not success and "not found" in result.get("error", "").lower():
            return jsonify({"status": "error", **result}), 404

        if not success:
            return jsonify({"status": "error", **result}), 400

        return jsonify({"status": "success", **result}), 200

    except Exception as e:
        logger.error(f"Error getting liveness result: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@liveness_bp.route('/session/<session_id>', methods=['GET'])
@require_auth
def get_session_status(session_id: str):
    """Get Session Status
    ---
    tags:
      - Liveness
    summary: Check the current status of a liveness session
    parameters:
      - name: session_id
        in: path
        type: string
        required: true
    responses:
      200:
        description: Session status
      404:
        description: Session not found
    """
    try:
        service = _get_liveness_service()
        session = service.get_session(session_id)

        if session is None:
            return jsonify({"status": "error", "error": "Session not found or expired"}), 404

        return jsonify({
            "status": "success",
            "session_id": session.session_id,
            "challenges": session.challenges,
            "current_step": session.current_step,
            "current_challenge": session.current_challenge,
            "total_steps": len(session.challenges),
            "completed": session.completed,
            "results_so_far": [
                {
                    "challenge": r.challenge,
                    "passed": r.passed,
                    "score": r.score,
                }
                for r in session.results
            ],
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@liveness_bp.route('/check', methods=['POST'])
@require_auth
@rate_limit
def check_liveness_stateless():
    """Stateless Liveness Check
    ---
    tags:
      - Liveness
    summary: One-shot liveness check (no session needed)
    description: |
      Stateless endpoint for validating a single liveness challenge.
      Send a baseline image (looking straight) and a challenge image in one request.

      Supported challenges: look_left, look_right, smile, open_mouth, look_up, look_down
    consumes:
      - multipart/form-data
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
      - name: baseline
        in: formData
        type: file
        required: true
        description: Baseline image (user looking straight at camera)
      - name: challenge_image
        in: formData
        type: file
        required: true
        description: Challenge image (user performing the action)
      - name: challenge
        in: formData
        type: string
        required: true
        description: "Challenge type: look_left, look_right, smile, open_mouth, look_up, look_down"
    responses:
      200:
        description: Challenge validation result
      400:
        description: Bad request
      500:
        description: Server error
    """
    from app.core.entities import LivenessChallenge
    from app.engines.liveness_validator import LivenessValidator

    # Get inputs
    challenge = request.form.get('challenge', '').strip().lower()
    baseline_file = request.files.get('baseline')
    challenge_file = request.files.get('challenge_image')

    # Validate challenge type
    valid_challenges = [c.value for c in LivenessChallenge if c != LivenessChallenge.LOOK_STRAIGHT]
    if not challenge:
        return jsonify({"status": "error", "error": "Missing 'challenge' field"}), 400
    if challenge not in valid_challenges:
        return jsonify({
            "status": "error",
            "error": f"Invalid challenge: '{challenge}'. Valid: {valid_challenges}",
        }), 400

    # Validate files
    if not baseline_file:
        return jsonify({"status": "error", "error": "Missing 'baseline' image"}), 400
    if not challenge_file:
        return jsonify({"status": "error", "error": "Missing 'challenge_image' image"}), 400

    try:
        # Decode images
        baseline_img = ImageProcessor._decode_image(baseline_file)
        challenge_img = ImageProcessor._decode_image(challenge_file)

        if baseline_img is None:
            return jsonify({"status": "error", "error": "Could not decode baseline image"}), 400
        if challenge_img is None:
            return jsonify({"status": "error", "error": "Could not decode challenge image"}), 400

        engine = get_engine()
        validator = LivenessValidator()

        # Detect face in baseline
        baseline_detection = engine.detect_face(baseline_img)
        if not baseline_detection.face_found or baseline_detection.landmarks is None:
            return jsonify({"status": "fail", "error": "No face detected in baseline image"}), 400

        # Anti-spoof check on baseline (if enabled)
        import os
        anti_spoof = os.getenv('ANTI_SPOOFING', 'false').lower().strip() in ('true', '1', 'yes')
        anti_spoof_result = None
        if anti_spoof:
            liveness = engine.liveness_check(baseline_img, baseline_detection)
            anti_spoof_result = {
                "is_live": liveness.is_live,
                "score": round(liveness.score, 4),
            }
            if not liveness.is_live:
                return jsonify({
                    "status": "fail",
                    "passed": False,
                    "message": f"Anti-spoof failed: {liveness.message}",
                    "anti_spoof": anti_spoof_result,
                }), 200

        # Detect face in challenge image
        challenge_detection = engine.detect_face(challenge_img)
        if not challenge_detection.face_found or challenge_detection.landmarks is None:
            return jsonify({"status": "fail", "error": "No face detected in challenge image"}), 400

        # Validate challenge
        result = validator.validate_challenge(
            challenge=challenge,
            detections=[challenge_detection],
            baseline_landmarks=baseline_detection.landmarks,
            consensus_threshold=0.5,  # single frame, so 1/1 must pass
        )

        response = {
            "status": "success" if result.passed else "fail",
            "challenge": challenge,
            "passed": result.passed,
            "score": result.score,
            "message": result.message,
            "engine": engine.name(),
        }

        if anti_spoof_result:
            response["anti_spoof"] = anti_spoof_result

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in stateless liveness check: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

