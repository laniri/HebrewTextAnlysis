"""Admin configuration endpoints — view and update application settings.

GET  /admin/config  — return current admin configuration
POST /admin/config  — update configuration settings
GET  /admin/models  — list available Bedrock foundation models

All routes require the X-Admin-Password header matching the configured
ADMIN_PASSWORD environment variable.

Requirements implemented: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.config import settings
from app.models.schemas import AdminConfig, AdminConfigUpdate, ModelInfo
from app.api.rewrite import bedrock_service
from app.services.bedrock_service import BedrockUnavailableError

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------


def verify_admin(x_admin_password: str = Header(...)) -> None:
    """Verify the admin password from the X-Admin-Password header.

    Raises 401 if the password does not match the configured value.
    """
    if x_admin_password != settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/config", response_model=AdminConfig, dependencies=[Depends(verify_admin)])
async def get_config() -> AdminConfig:
    """Return the current admin configuration."""
    return AdminConfig(
        bedrock_model_id=settings.BEDROCK_MODEL_ID,
        severity_threshold=settings.SEVERITY_THRESHOLD,
        max_diagnoses_shown=settings.MAX_DIAGNOSES_SHOWN,
        max_interventions_shown=settings.MAX_INTERVENTIONS_SHOWN,
    )


@router.post("/config", response_model=AdminConfig, dependencies=[Depends(verify_admin)])
async def update_config(update: AdminConfigUpdate) -> AdminConfig:
    """Update admin configuration settings.

    Only provided (non-None) fields are updated. Returns the full
    configuration after applying changes.
    """
    if update.bedrock_model_id is not None:
        bedrock_service.update_model(update.bedrock_model_id)
        settings.BEDROCK_MODEL_ID = update.bedrock_model_id

    if update.severity_threshold is not None:
        settings.SEVERITY_THRESHOLD = update.severity_threshold

    if update.max_diagnoses_shown is not None:
        settings.MAX_DIAGNOSES_SHOWN = update.max_diagnoses_shown

    if update.max_interventions_shown is not None:
        settings.MAX_INTERVENTIONS_SHOWN = update.max_interventions_shown

    return AdminConfig(
        bedrock_model_id=settings.BEDROCK_MODEL_ID,
        severity_threshold=settings.SEVERITY_THRESHOLD,
        max_diagnoses_shown=settings.MAX_DIAGNOSES_SHOWN,
        max_interventions_shown=settings.MAX_INTERVENTIONS_SHOWN,
    )


@router.get("/models", response_model=list[ModelInfo], dependencies=[Depends(verify_admin)])
async def list_models() -> list[ModelInfo]:
    """List available Bedrock foundation models with text output.

    Returns 503 if the Bedrock service is unavailable.
    """
    try:
        return bedrock_service.list_models()
    except BedrockUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="שירות השכתוב אינו זמין כרגע, נסו שוב מאוחר יותר",
        )
