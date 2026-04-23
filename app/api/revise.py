"""POST /api/revise endpoint — revision comparison.

Accepts original and edited Hebrew text, runs analysis on both, computes
delta scores, and identifies resolved and new diagnoses.

Requirements implemented: 2.1, 2.2, 2.3, 2.4.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.api.analyze import model_service
from app.models.schemas import (
    ReviseRequest,
    ReviseResponse,
    ScoresResponse,
)

router = APIRouter(prefix="/api", tags=["revision"])


@router.post("/revise", response_model=ReviseResponse)
async def revise(request: ReviseRequest) -> ReviseResponse:
    """Compare original and edited text, returning delta scores and diagnosis transitions.

    Steps:
    1. Validate non-empty text inputs (whitespace-only counts as empty).
    2. Run analysis on both original and edited text.
    3. Compute delta scores (revised - original) for all 5 score keys.
    4. Identify resolved diagnoses (active in original, not in revised).
    5. Identify new diagnoses (active in revised, not in original).
    6. Return the full ReviseResponse.
    """
    original_text = request.original_text.strip()
    edited_text = request.edited_text.strip()

    if not original_text or not edited_text:
        raise HTTPException(
            status_code=400,
            detail="הטקסט המקורי והטקסט המתוקן לא יכולים להיות ריקים",
        )

    # --- Run analysis on both texts ---
    try:
        original_raw = model_service.analyze(original_text)
        revised_raw = model_service.analyze(edited_text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="המודל אינו זמין כרגע — ודאו שקובצי המודל נמצאים בנתיב המוגדר",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"שגיאה בניתוח הטקסט: {exc}",
        )

    # --- Build score responses ---
    original_scores = ScoresResponse(**original_raw["scores"])
    revised_scores = ScoresResponse(**revised_raw["scores"])

    # --- Compute delta scores ---
    score_keys = ["difficulty", "style", "fluency", "cohesion", "complexity"]
    deltas: dict[str, float] = {}
    for key in score_keys:
        deltas[key] = getattr(revised_scores, key) - getattr(original_scores, key)

    # --- Determine active diagnoses using severity threshold ---
    threshold = settings.SEVERITY_THRESHOLD

    original_active: set[str] = {
        dtype
        for dtype, severity in original_raw["diagnoses"].items()
        if severity > threshold
    }
    revised_active: set[str] = {
        dtype
        for dtype, severity in revised_raw["diagnoses"].items()
        if severity > threshold
    }

    # Resolved: active in original but NOT in revised
    resolved_diagnoses = sorted(original_active - revised_active)

    # New: active in revised but NOT in original
    new_diagnoses = sorted(revised_active - original_active)

    return ReviseResponse(
        original_scores=original_scores,
        revised_scores=revised_scores,
        deltas=deltas,
        resolved_diagnoses=resolved_diagnoses,
        new_diagnoses=new_diagnoses,
    )
