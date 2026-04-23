"""Pydantic request/response schemas for the Hebrew Writing Coach API.

Defines all data models used by the FastAPI endpoints for text analysis,
revision comparison, AI rewrite, example texts, and admin configuration.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/analyze."""

    text: str = Field(..., min_length=1, description="Hebrew text to analyze")


class ReviseRequest(BaseModel):
    """Request body for POST /api/revise."""

    original_text: str = Field(..., min_length=1)
    edited_text: str = Field(..., min_length=1)


class RewriteRequest(BaseModel):
    """Request body for POST /api/rewrite."""

    text: str = Field(..., min_length=1, description="Sentence(s) to rewrite")
    diagnosis_type: str = Field(..., description="One of 8 diagnosis types")
    context: str = Field("", description="Optional surrounding text for context")


class AdminConfigUpdate(BaseModel):
    """Request body for POST /admin/config."""

    bedrock_model_id: str | None = None
    severity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    max_diagnoses_shown: int | None = Field(None, ge=1, le=10)
    max_interventions_shown: int | None = Field(None, ge=1, le=10)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class ScoresResponse(BaseModel):
    """Five linguistic scores, each in [0.0, 1.0]."""

    difficulty: float = Field(..., ge=0.0, le=1.0)
    style: float = Field(..., ge=0.0, le=1.0)
    fluency: float = Field(..., ge=0.0, le=1.0)
    cohesion: float = Field(..., ge=0.0, le=1.0)
    complexity: float = Field(..., ge=0.0, le=1.0)


class SentenceAnnotation(BaseModel):
    """Per-sentence annotation with character offsets and highlight level."""

    index: int
    text: str
    char_start: int
    char_end: int
    complexity: float
    highlight: Literal["red", "yellow", "none"]


class CohesionGap(BaseModel):
    """Cohesion gap between two adjacent sentences."""

    pair: tuple[int, int]
    severity: float
    char_start: int
    char_end: int


class LocalizedDiagnosis(BaseModel):
    """A diagnosis with all fields localized to Hebrew."""

    type: str
    severity: float
    label_he: str
    explanation_he: str
    actions_he: list[str]
    tip_he: str


class LocalizedIntervention(BaseModel):
    """An intervention with all fields localized to Hebrew."""

    type: str
    priority: float
    target_diagnosis: str
    actions_he: list[str]
    exercises_he: list[str]


class AnalyzeResponse(BaseModel):
    """Full response from POST /api/analyze."""

    scores: ScoresResponse
    diagnoses: list[LocalizedDiagnosis]
    interventions: list[LocalizedIntervention]
    sentences: list[SentenceAnnotation]
    cohesion_gaps: list[CohesionGap]


class ReviseResponse(BaseModel):
    """Response from POST /api/revise comparing original and edited text."""

    original_scores: ScoresResponse
    revised_scores: ScoresResponse
    deltas: dict[str, float]
    resolved_diagnoses: list[str]
    new_diagnoses: list[str]


class RewriteResponse(BaseModel):
    """Response from POST /api/rewrite with AI-generated suggestion."""

    suggestion: str
    model_used: str


class ExampleSummary(BaseModel):
    """Summary of an example text for listing."""

    id: str
    label: str
    category: str
    preview: str


class ExampleFull(BaseModel):
    """Full example text content."""

    id: str
    label: str
    category: str
    text: str


class AdminConfig(BaseModel):
    """Current admin configuration."""

    bedrock_model_id: str
    severity_threshold: float
    max_diagnoses_shown: int
    max_interventions_shown: int


class ModelInfo(BaseModel):
    """Information about an available Bedrock model."""

    model_id: str
    model_name: str
    provider: str


class HealthResponse(BaseModel):
    """Response from GET /api/health."""

    status: str
    model_loaded: bool
