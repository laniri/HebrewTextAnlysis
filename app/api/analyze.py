"""POST /api/analyze endpoint — full text analysis.

Accepts Hebrew text, runs the DictaBERT inference pipeline, localizes
diagnoses and interventions to Hebrew, and returns a structured response
with scores, diagnoses, interventions, sentence annotations, and
cohesion gaps.

Requirements implemented: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 7.2, 7.4.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CohesionGap,
    ScoresResponse,
    SentenceAnnotation,
)
from app.services.localization import localize_diagnosis, localize_intervention
from app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["analysis"])

# Module-level instance — initialized in main.py lifespan.
model_service = ModelService()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze Hebrew text and return localized linguistic feedback.

    Steps:
    1. Validate non-empty text (Pydantic handles min_length=1, but we
       also guard against whitespace-only input).
    2. Call ModelService.analyze() for raw predictions.
    3. Filter diagnoses by severity > threshold, sort desc, cap at max.
    4. Filter interventions to those targeting displayed diagnoses, cap.
    5. Localize diagnoses and interventions to Hebrew.
    6. Return the full AnalyzeResponse.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="הטקסט לא יכול להיות ריק")

    # --- Raw predictions from the model ---
    try:
        raw = model_service.analyze(text)
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

    # --- Scores ---
    scores = ScoresResponse(**raw["scores"])

    # --- Diagnoses: filter → sort → cap → localize ---
    threshold = settings.SEVERITY_THRESHOLD
    max_diag = settings.MAX_DIAGNOSES_SHOWN

    # raw["diagnoses"] is a dict of diagnosis_type → severity float
    filtered_diagnoses = [
        (dtype, severity)
        for dtype, severity in raw["diagnoses"].items()
        if severity > threshold
    ]
    filtered_diagnoses.sort(key=lambda x: x[1], reverse=True)
    displayed_diagnoses = filtered_diagnoses[: max_diag]

    # Set of active (displayed) diagnosis types for intervention filtering
    active_diagnosis_types = {dtype for dtype, _ in displayed_diagnoses}

    localized_diagnoses = [
        localize_diagnosis(dtype, severity)
        for dtype, severity in displayed_diagnoses
    ]

    # --- Interventions: filter by active diagnoses → cap → localize ---
    max_interv = settings.MAX_INTERVENTIONS_SHOWN

    relevant_interventions = [
        iv
        for iv in raw["interventions"]
        if iv.get("target_diagnosis") in active_diagnosis_types
    ]
    capped_interventions = relevant_interventions[: max_interv]

    localized_interventions = [
        localize_intervention(iv) for iv in capped_interventions
    ]

    # --- Sentence annotations ---
    sentences = [SentenceAnnotation(**s) for s in raw["sentences"]]

    # --- Cohesion gaps ---
    cohesion_gaps = [CohesionGap(**g) for g in raw["cohesion_gaps"]]

    return AnalyzeResponse(
        scores=scores,
        diagnoses=localized_diagnoses,
        interventions=localized_interventions,
        sentences=sentences,
        cohesion_gaps=cohesion_gaps,
    )
