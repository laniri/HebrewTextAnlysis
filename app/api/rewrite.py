"""POST /api/rewrite endpoint — AI-powered rewrite suggestions.

Accepts a text segment and diagnosis type, validates the diagnosis type
against the 8 recognized types, calls BedrockService for an AI rewrite
suggestion, and returns the suggestion with the model identifier.

Requirements implemented: 3.1, 3.3, 3.4, 3.5.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import RewriteRequest, RewriteResponse
from app.services.bedrock_service import BedrockService, BedrockUnavailableError
from app.services.localization import get_diagnosis_types

router = APIRouter(prefix="/api", tags=["rewrite"])

# Module-level instance — used by the endpoint.
bedrock_service = BedrockService()


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest) -> RewriteResponse:
    """Generate an AI rewrite suggestion for a diagnosed issue.

    Steps:
    1. Validate that diagnosis_type is one of the 8 recognized types.
    2. Call BedrockService.rewrite() with text, diagnosis_type, and context.
    3. Return the suggestion and model identifier.
    4. Handle BedrockUnavailableError → 503 with Hebrew message.
    """
    # --- Validate diagnosis type ---
    valid_types = get_diagnosis_types()
    if request.diagnosis_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"סוג האבחנה אינו מוכר: {request.diagnosis_type}",
        )

    # --- Call Bedrock for rewrite suggestion ---
    try:
        result = bedrock_service.rewrite(
            text=request.text,
            diagnosis_type=request.diagnosis_type,
            context=request.context,
        )
    except BedrockUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="שירות השכתוב אינו זמין כרגע, נסו שוב מאוחר יותר",
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="שירות השכתוב אינו זמין כרגע — בדקו הגדרות AWS",
        )

    return RewriteResponse(
        suggestion=result["suggestion"],
        model_used=result["model_used"],
    )
