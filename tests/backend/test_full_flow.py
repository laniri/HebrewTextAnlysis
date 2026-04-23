"""Integration tests for full analyze and revise flows.

Tests the complete request/response cycle through the FastAPI app with
mocked model_service, verifying all response fields are present and
correctly structured.

Requirements validated: 1.1, 2.1, 15.4.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Helpers — comprehensive mock data
# ---------------------------------------------------------------------------


def _make_full_analyze_result(
    *,
    scores: dict | None = None,
    diagnoses: dict | None = None,
) -> dict:
    """Build a comprehensive mock return value for model_service.analyze()."""
    if scores is None:
        scores = {
            "difficulty": 0.65,
            "style": 0.55,
            "fluency": 0.72,
            "cohesion": 0.48,
            "complexity": 0.61,
        }
    if diagnoses is None:
        diagnoses = {
            "low_cohesion": 0.75,
            "sentence_over_complexity": 0.62,
            "low_lexical_diversity": 0.45,
            "pronoun_overuse": 0.15,
            "structural_inconsistency": 0.28,
            "low_morphological_richness": 0.12,
            "fragmented_writing": 0.08,
            "punctuation_deficiency": 0.05,
        }
    return {
        "scores": scores,
        "diagnoses": diagnoses,
        "interventions": [
            {
                "type": "cohesion_improvement",
                "priority": 0.9,
                "target_diagnosis": "low_cohesion",
            },
            {
                "type": "sentence_simplification",
                "priority": 0.8,
                "target_diagnosis": "sentence_over_complexity",
            },
            {
                "type": "vocabulary_expansion",
                "priority": 0.6,
                "target_diagnosis": "low_lexical_diversity",
            },
        ],
        "sentences": [
            {
                "index": 0,
                "text": "זהו משפט ראשון לבדיקה",
                "char_start": 0,
                "char_end": 21,
                "complexity": 0.75,
                "highlight": "red",
            },
            {
                "index": 1,
                "text": "זהו משפט שני",
                "char_start": 22,
                "char_end": 35,
                "complexity": 0.55,
                "highlight": "yellow",
            },
            {
                "index": 2,
                "text": "משפט שלישי פשוט",
                "char_start": 36,
                "char_end": 52,
                "complexity": 0.2,
                "highlight": "none",
            },
        ],
        "cohesion_gaps": [
            {
                "pair": (0, 1),
                "severity": 0.65,
                "char_start": 21,
                "char_end": 22,
            },
            {
                "pair": (1, 2),
                "severity": 0.45,
                "char_start": 35,
                "char_end": 36,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Integration test — full analyze flow
# ---------------------------------------------------------------------------


def test_full_analyze_flow() -> None:
    """Submit text → get complete response with scores, diagnoses,
    interventions, sentences, and cohesion_gaps."""
    mock_data = _make_full_analyze_result()

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/analyze",
            json={"text": "זהו משפט ראשון לבדיקה. זהו משפט שני. משפט שלישי פשוט."},
        )

    assert response.status_code == 200
    data = response.json()

    # --- Scores ---
    assert "scores" in data
    scores = data["scores"]
    assert len(scores) == 5
    for key in ["difficulty", "style", "fluency", "cohesion", "complexity"]:
        assert key in scores
        assert isinstance(scores[key], (int, float))
        assert 0.0 <= scores[key] <= 1.0

    # --- Diagnoses ---
    assert "diagnoses" in data
    diagnoses = data["diagnoses"]
    assert isinstance(diagnoses, list)
    # With default threshold 0.3 and max 3, we should get up to 3 diagnoses
    assert len(diagnoses) <= 3
    for diag in diagnoses:
        assert "type" in diag
        assert "severity" in diag
        assert "label_he" in diag
        assert "explanation_he" in diag
        assert "actions_he" in diag
        assert isinstance(diag["actions_he"], list)
        assert "tip_he" in diag
        # All displayed diagnoses should have severity > threshold
        assert diag["severity"] > 0.3

    # --- Interventions ---
    assert "interventions" in data
    interventions = data["interventions"]
    assert isinstance(interventions, list)
    diagnosis_types = {d["type"] for d in diagnoses}
    for interv in interventions:
        assert "type" in interv
        assert "priority" in interv
        assert "target_diagnosis" in interv
        assert "actions_he" in interv
        assert "exercises_he" in interv
        # Each intervention should target a displayed diagnosis
        assert interv["target_diagnosis"] in diagnosis_types

    # --- Sentences ---
    assert "sentences" in data
    sentences = data["sentences"]
    assert isinstance(sentences, list)
    assert len(sentences) == 3
    for i, sent in enumerate(sentences):
        assert sent["index"] == i
        assert "text" in sent
        assert "char_start" in sent
        assert "char_end" in sent
        assert sent["char_start"] < sent["char_end"]
        assert "complexity" in sent
        assert "highlight" in sent
        assert sent["highlight"] in ("red", "yellow", "none")

    # --- Cohesion gaps ---
    assert "cohesion_gaps" in data
    gaps = data["cohesion_gaps"]
    assert isinstance(gaps, list)
    for gap in gaps:
        assert "pair" in gap
        assert "severity" in gap
        assert "char_start" in gap
        assert "char_end" in gap
        pair = gap["pair"]
        assert len(pair) == 2
        assert pair[1] == pair[0] + 1


# ---------------------------------------------------------------------------
# Integration test — full revise flow
# ---------------------------------------------------------------------------


def test_full_revise_flow() -> None:
    """Submit text pair → get deltas, resolved, and new diagnoses."""
    original_data = _make_full_analyze_result(
        scores={
            "difficulty": 0.65,
            "style": 0.55,
            "fluency": 0.72,
            "cohesion": 0.48,
            "complexity": 0.61,
        },
        diagnoses={
            "low_cohesion": 0.75,
            "sentence_over_complexity": 0.62,
            "low_lexical_diversity": 0.45,
            "pronoun_overuse": 0.15,
            "structural_inconsistency": 0.28,
            "low_morphological_richness": 0.12,
            "fragmented_writing": 0.08,
            "punctuation_deficiency": 0.05,
        },
    )
    revised_data = _make_full_analyze_result(
        scores={
            "difficulty": 0.50,
            "style": 0.70,
            "fluency": 0.80,
            "cohesion": 0.75,
            "complexity": 0.40,
        },
        diagnoses={
            "low_cohesion": 0.20,  # resolved
            "sentence_over_complexity": 0.15,  # resolved
            "low_lexical_diversity": 0.45,  # unchanged
            "pronoun_overuse": 0.50,  # new
            "structural_inconsistency": 0.28,
            "low_morphological_richness": 0.12,
            "fragmented_writing": 0.08,
            "punctuation_deficiency": 0.05,
        },
    )

    call_count = 0

    def mock_analyze(text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_data
        return revised_data

    with patch("app.api.revise.model_service") as mock_svc:
        mock_svc.analyze.side_effect = mock_analyze
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={
                "original_text": "טקסט מקורי ארוך לבדיקה מלאה",
                "edited_text": "טקסט מתוקן ומשופר לבדיקה מלאה",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # --- Original and revised scores ---
    assert "original_scores" in data
    assert "revised_scores" in data
    for key in ["difficulty", "style", "fluency", "cohesion", "complexity"]:
        assert key in data["original_scores"]
        assert key in data["revised_scores"]

    # --- Deltas ---
    assert "deltas" in data
    deltas = data["deltas"]
    for key in ["difficulty", "style", "fluency", "cohesion", "complexity"]:
        assert key in deltas
        expected = data["revised_scores"][key] - data["original_scores"][key]
        assert abs(deltas[key] - expected) < 1e-6

    # --- Resolved diagnoses ---
    assert "resolved_diagnoses" in data
    resolved = data["resolved_diagnoses"]
    assert isinstance(resolved, list)
    assert "low_cohesion" in resolved
    assert "sentence_over_complexity" in resolved

    # --- New diagnoses ---
    assert "new_diagnoses" in data
    new_diag = data["new_diagnoses"]
    assert isinstance(new_diag, list)
    assert "pronoun_overuse" in new_diag


# ---------------------------------------------------------------------------
# Integration test — health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_status_healthy() -> None:
    """GET /api/health should return status when model is loaded."""
    with patch("app.api.health.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_endpoint_returns_unhealthy_when_model_not_loaded() -> None:
    """GET /api/health should return 503 when model is not loaded."""
    with patch("app.api.health.model_service") as mock_svc:
        mock_svc.is_loaded = False
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False
