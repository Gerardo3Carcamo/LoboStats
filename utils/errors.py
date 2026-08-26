from typing import Any

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException


class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "validation_error",
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return jsonify({"error": {"code": error.code, "message": error.message, "details": error.details}}), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return jsonify({"error": {"code": error.name.lower().replace(" ", "_"), "message": error.description, "details": None}}), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.exception("Error no controlado", exc_info=error)
        return jsonify({"error": {"code": "internal_error", "message": "Ocurrió un error interno.", "details": None}}), 500
