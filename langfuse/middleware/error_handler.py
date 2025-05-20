from typing import Tuple, Dict
from flask import jsonify, Response

def error_handler(error: Exception) -> Tuple[Response, int]:
    error_message: str = str(error)
    status_code: int = getattr(error, 'code', 500)
    
    response: Dict[str, Dict[str, str]] = {
        'error': {
            'message': error_message,
            'type': error.__class__.__name__
        }
    }
    
    return jsonify(response), status_code 