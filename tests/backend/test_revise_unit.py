"""Unit tests for POST /api/revise endpoint.

Tests empty text validation and correct delta structure with mocked
model_service.

Requirements validated: 2.1, 2.2, 2.3, 2.4.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_analyze_result(
    *,
    difficulty: float = 0.5,
    style: float = 0.4,
    fluency: float = 0.6,
    cohesion: float = 0.7,
    complexity: float = 0.3,
    diagnoses: dict | None = None,
) -> dict:
    """Build a mock return value for model_service.analyze()."""
    if diagnoses is None:
        diagnoses = {
            "low_cohesion": 0.8,
            "sentence_over_complexity": 0.2,
            "low_lexical_diversity": 0.6,
            "pronoun_overuse": 0.1,
            "structural_inconsistency": 0.15,
            "low_morphological_richness": 0.05,
            "fragmented_writing": 0.1,
            "punctuation_deficiency": 0.05,
        }
    return {
        "scores": {
            "difficulty": difficulty,
            "style": style,
            "fluency": fluency,
            "cohesion": cohesion,
            "complexity": complexity,
        },
        "diagnoses": diagnoses,
        "interventions": [],
        "sentences": [
            {
                "index": 0,
                "text": "שלום",
                "char_start": 0,
                "char_end": 4,
                "complexity": 0.3,
                "highlight": "none",
            },
        ],
        "cohesion_gaps": [],
    }


# ---------------------------------------------------------------------------
# Tests — empty text returns 400
# ---------------------------------------------------------------------------


def test_empty_original_text_returns_400() -> None:
    """Empty original_text should return 400."""
    with patch("app.api.revise.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={"original_text": "   ", "edited_text": "שלום עולם"},
        )

    assert response.status_code == 400
    assert "ריקים" in response.json()["detail"]


def test_empty_edited_text_returns_400() -> None:
    """Empty edited_text should return 400."""
    with patch("app.api.revise.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={"original_text": "שלום עולם", "edited_text": "   "},
        )

    assert response.status_code == 400


def test_both_empty_returns_400() -> None:
    """Both texts empty should return 400."""
    with patch("app.api.revise.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={"original_text": "  ", "edited_text": "  "},
        )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tests — valid text pair returns correct delta structure
# ---------------------------------------------------------------------------


def test_valid_revise_returns_delta_structure() -> None:
    """Valid text pair should return original_scores, revised_scores, deltas,
    resolved_diagnoses, and new_diagnoses."""
    original_result = _make_mock_analyze_result(
        difficulty=0.5, style=0.4, fluency=0.6, cohesion=0.3, complexity=0.7,
        diagnoses={
            "low_cohesion": 0.8,
            "sentence_over_complexity": 0.5,
            "low_lexical_diversity": 0.2,
            "pronoun_overuse": 0.1,
            "structural_inconsistency": 0.15,
            "low_morphological_richness": 0.05,
            "fragmented_writing": 0.1,
            "punctuation_deficiency": 0.05,
        },
    )
    revised_result = _make_mock_analyze_result(
        difficulty=0.4, style=0.6, fluency=0.7, cohesion=0.6, complexity=0.4,
        diagnoses={
            "low_cohesion": 0.2,  # resolved (was 0.8, now below threshold)
            "sentence_over_complexity": 0.1,  # resolved
            "low_lexical_diversity": 0.2,
            "pronoun_overuse": 0.5,  # new (was 0.1, now above threshold)
            "structural_inconsistency": 0.15,
            "low_morphological_richness": 0.05,
            "fragmented_writing": 0.1,
            "punctuation_deficiency": 0.05,
        },
    )

    call_count = 0

    def mock_analyze(text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_result
        return revised_result

    with patch("app.api.revise.model_service") as mock_svc:
        mock_svc.analyze.side_effect = mock_analyze
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={
                "original_text": "טקסט מקורי לבדיקה",
                "edited_text": "טקסט מתוקן לבדיקה",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Check structure
    assert "original_scores" in data
    assert "revised_scores" in data
    assert "deltas" in data
    assert "resolved_diagnoses" in data
    assert "new_diagnoses" in data

    # Verify delta arithmetic: deltas[key] == revised - original
    score_keys = ["difficulty", "style", "fluency", "cohesion", "complexity"]
    for key in score_keys:
        expected_delta = data["revised_scores"][key] - data["original_scores"][key]
        assert abs(data["deltas"][key] - expected_delta) < 1e-6, (
            f"Delta for {key}: expected {expected_delta}, got {data['deltas'][key]}"
        )

    # Verify resolved diagnoses (active in original, not in revised)
    # low_cohesion was 0.8 > 0.3 in original, 0.2 ≤ 0.3 in revised → resolved
    # sentence_over_complexity was 0.5 > 0.3 in original, 0.1 ≤ 0.3 in revised → resolved
    assert "low_cohesion" in data["resolved_diagnoses"]
    assert "sentence_over_complexity" in data["resolved_diagnoses"]

    # Verify new diagnoses (active in revised, not in original)
    # pronoun_overuse was 0.1 ≤ 0.3 in original, 0.5 > 0.3 in revised → new
    assert "pronoun_overuse" in data["new_diagnoses"]
