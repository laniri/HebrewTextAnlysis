"""Unit tests for POST /api/analyze endpoint.

Tests empty text validation, complete AnalyzeResponse structure with mocked
model_service, and highlight classification for specific severity values.

Requirements validated: 1.7, 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Helpers — mock data factories
# ---------------------------------------------------------------------------


def _make_mock_analyze_result(
    *,
    sentence_complexity: float = 0.5,
    cohesion_severity: float = 0.8,
) -> dict:
    """Build a realistic mock return value for model_service.analyze().

    Contains all 8 diagnosis types, 5 scores, 2 sentences with offsets,
    1 cohesion gap, and interventions targeting active diagnoses.
    """
    return {
        "scores": {
            "difficulty": 0.5,
            "style": 0.4,
            "fluency": 0.6,
            "cohesion": 0.7,
            "complexity": 0.3,
        },
        "diagnoses": {
            "low_cohesion": 0.8,
            "sentence_over_complexity": 0.2,
            "low_lexical_diversity": 0.6,
            "pronoun_overuse": 0.1,
            "structural_inconsistency": 0.15,
            "low_morphological_richness": 0.05,
            "fragmented_writing": 0.1,
            "punctuation_deficiency": 0.05,
        },
        "interventions": [
            {
                "type": "cohesion_improvement",
                "priority": 0.9,
                "target_diagnosis": "low_cohesion",
            },
            {
                "type": "vocabulary_expansion",
                "priority": 0.7,
                "target_diagnosis": "low_lexical_diversity",
            },
        ],
        "sentences": [
            {
                "index": 0,
                "text": "שלום עולם",
                "char_start": 0,
                "char_end": 9,
                "complexity": sentence_complexity,
                "highlight": (
                    "red" if sentence_complexity > 0.7
                    else "yellow" if sentence_complexity >= 0.4
                    else "none"
                ),
            },
            {
                "index": 1,
                "text": "זהו טקסט",
                "char_start": 10,
                "char_end": 19,
                "complexity": 0.3,
                "highlight": "none",
            },
        ],
        "cohesion_gaps": [
            {
                "pair": (0, 1),
                "severity": cohesion_severity,
                "char_start": 9,
                "char_end": 10,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests — empty text returns 400
# ---------------------------------------------------------------------------


def test_empty_text_returns_400_with_hebrew_message() -> None:
    """Empty text should return 400 with Hebrew error message."""
    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "   "})

    assert response.status_code == 400
    assert "הטקסט לא יכול להיות ריק" in response.json()["detail"]


def test_whitespace_only_text_returns_400() -> None:
    """Whitespace-only text should also return 400."""
    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "\t\n  "})

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tests — valid text returns complete AnalyzeResponse
# ---------------------------------------------------------------------------


def test_valid_text_returns_complete_analyze_response() -> None:
    """Valid Hebrew text should return a full AnalyzeResponse with all fields."""
    mock_data = _make_mock_analyze_result()

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    data = response.json()

    # Scores present with all 5 keys
    assert "scores" in data
    score_keys = {"difficulty", "style", "fluency", "cohesion", "complexity"}
    assert set(data["scores"].keys()) == score_keys
    for key in score_keys:
        assert 0.0 <= data["scores"][key] <= 1.0

    # Diagnoses present as a list
    assert "diagnoses" in data
    assert isinstance(data["diagnoses"], list)
    # At least one diagnosis should be active (low_cohesion=0.8, low_lexical_diversity=0.6)
    assert len(data["diagnoses"]) > 0
    for diag in data["diagnoses"]:
        assert "type" in diag
        assert "severity" in diag
        assert "label_he" in diag
        assert "explanation_he" in diag
        assert "actions_he" in diag
        assert "tip_he" in diag

    # Interventions present as a list
    assert "interventions" in data
    assert isinstance(data["interventions"], list)

    # Sentences present as a list
    assert "sentences" in data
    assert isinstance(data["sentences"], list)
    assert len(data["sentences"]) == 2

    # Cohesion gaps present as a list
    assert "cohesion_gaps" in data
    assert isinstance(data["cohesion_gaps"], list)


# ---------------------------------------------------------------------------
# Tests — highlight classification for specific severity values
# ---------------------------------------------------------------------------


def test_highlight_red_for_high_severity() -> None:
    """Sentence with complexity > 0.7 should get 'red' highlight."""
    mock_data = _make_mock_analyze_result(sentence_complexity=0.85)

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    sentences = response.json()["sentences"]
    # First sentence has complexity 0.85 → red
    assert sentences[0]["highlight"] == "red"


def test_highlight_yellow_for_medium_severity() -> None:
    """Sentence with complexity between 0.4 and 0.7 should get 'yellow' highlight."""
    mock_data = _make_mock_analyze_result(sentence_complexity=0.55)

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    sentences = response.json()["sentences"]
    assert sentences[0]["highlight"] == "yellow"


def test_highlight_none_for_low_severity() -> None:
    """Sentence with complexity ≤ 0.4 should get 'none' highlight."""
    mock_data = _make_mock_analyze_result(sentence_complexity=0.2)

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    sentences = response.json()["sentences"]
    assert sentences[0]["highlight"] == "none"


def test_highlight_boundary_at_0_4_is_yellow() -> None:
    """Sentence with complexity exactly 0.4 should get 'yellow' highlight."""
    mock_data = _make_mock_analyze_result(sentence_complexity=0.4)

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    sentences = response.json()["sentences"]
    assert sentences[0]["highlight"] == "yellow"


def test_highlight_boundary_at_0_7_is_yellow() -> None:
    """Sentence with complexity exactly 0.7 should get 'yellow' highlight."""
    mock_data = _make_mock_analyze_result(sentence_complexity=0.7)

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": "שלום עולם זהו טקסט"})

    assert response.status_code == 200
    sentences = response.json()["sentences"]
    assert sentences[0]["highlight"] == "yellow"
