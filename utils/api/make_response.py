from flask import jsonify

def make_response(data=None, message="", status="success"):
    """
    Unified JSON response format.
    """
    return jsonify({
        "status": status,
        "data": data or {},
        "message": message
    })