"""GET /api/health endpoint — application health check.

Returns the model loaded status and overall health of the service.
Returns 200 with healthy status when the model is loaded, or 503
with unhealthy status when the model is not loaded.

Requirements implemented: 15.4.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.analyze import model_service
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> JSONResponse:
    """Return application health status.

    Returns 200 with ``{"status": "healthy", "model_loaded": true}`` if
    the DictaBERT model is loaded and ready, or 503 with
    ``{"status": "unhealthy", "model_loaded": false}`` otherwise.
    """
    if model_service.is_loaded:
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "model_loaded": True},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "model_loaded": False},
    )
