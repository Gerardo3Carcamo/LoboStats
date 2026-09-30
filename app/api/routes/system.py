from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": get_settings().app_name}


@router.get("/auth/config")
async def auth_config() -> dict[str, str]:
    settings = get_settings()
    return {
        "provider": "google-workspace",
        "allowed_domain": settings.allowed_email_domain,
        "status": "development" if settings.is_development else "configuration-required",
    }
