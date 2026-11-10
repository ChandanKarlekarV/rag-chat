"""Domain errors raised by services and translated to the API error shape by the HTTP layer."""


class AppError(Exception):
    """An expected failure with a stable code and a friendly, actionable message."""

    status = 400

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        if status is not None:
            self.status = status


class NotFound(AppError):
    """Missing resource, or one owned by someone else (never reveal which)."""

    status = 404


class Unauthorized(AppError):
    """Not signed in or session expired."""

    status = 401


class Conflict(AppError):
    """Duplicate resource (e.g. same PDF uploaded twice)."""

    status = 409


class TooMany(AppError):
    """Rate limit or quota exceeded."""

    status = 429
