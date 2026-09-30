from typing import Annotated

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def validate_institutional_email(email: str) -> str:
    settings = get_settings()
    normalized = email.strip().lower()
    _, separator, domain = normalized.rpartition("@")
    if not separator or domain != settings.allowed_email_domain.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Se requiere una cuenta @{settings.allowed_email_domain}",
        )
    return normalized


async def get_current_email(
    x_user_email: Annotated[str | None, Header()] = None,
) -> str:
    settings = get_settings()
    if settings.is_development:
        return validate_institutional_email(x_user_email or settings.dev_user_email)

    # Production remains closed until the Google Workspace OIDC verifier is configured.
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="La autenticacion institucional aun no esta configurada",
    )
