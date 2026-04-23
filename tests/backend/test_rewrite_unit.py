"""Unit tests for POST /api/rewrite endpoint with Bedrock mocking.

Tests valid rewrite request, Bedrock unavailable (503), and invalid
diagnosis type (400).

Requirements validated: 3.1, 3.3, 3.4, 3.5.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.bedrock_service import BedrockUnavailableError


# ---------------------------------------------------------------------------
# Tests — valid rewrite request
# ---------------------------------------------------------------------------


def test_valid_rewrite_returns_suggestion_and_model() -> None:
    """Valid rewrite request should return suggestion and model_used."""
    mock_result = {
        "suggestion": "הטקסט המשוכתב בעברית",
        "model_used": "anthropic.claude-3-haiku-20240307-v1:0",
    }

    with patch("app.api.rewrite.bedrock_service") as mock_bedrock:
        mock_bedrock.rewrite.return_value = mock_result
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/rewrite",
            json={
                "text": "משפט מורכב מדי לבדיקה",
                "diagnosis_type": "sentence_over_complexity",
                "context": "",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["suggestion"] == "הטקסט המשוכתב בעברית"
    assert data["model_used"] == "anthropic.claude-3-haiku-20240307-v1:0"


def test_valid_rewrite_with_context() -> None:
    """Rewrite request with context should pass context to bedrock_service."""
    mock_result = {
        "suggestion": "טקסט משופר",
        "model_used": "test-model",
    }

    with patch("app.api.rewrite.bedrock_service") as mock_bedrock:
        mock_bedrock.rewrite.return_value = mock_result
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/rewrite",
            json={
                "text": "משפט לשכתוב",
                "diagnosis_type": "low_cohesion",
                "context": "הקשר נוסף לטקסט",
            },
        )

    assert response.status_code == 200
    # Verify bedrock_service.rewrite was called with the context
    mock_bedrock.rewrite.assert_called_once_with(
        text="משפט לשכתוב",
        diagnosis_type="low_cohesion",
        context="הקשר נוסף לטקסט",
    )


# ---------------------------------------------------------------------------
# Tests — Bedrock unavailable returns 503
# ---------------------------------------------------------------------------


def test_bedrock_unavailable_returns_503() -> None:
    """When Bedrock is unavailable, should return 503 with Hebrew message."""
    with patch("app.api.rewrite.bedrock_service") as mock_bedrock:
        mock_bedrock.rewrite.side_effect = BedrockUnavailableError(
            "שירות השכתוב אינו זמין כרגע, נסו שוב מאוחר יותר"
        )
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/rewrite",
            json={
                "text": "משפט לשכתוב",
                "diagnosis_type": "low_cohesion",
                "context": "",
            },
        )

    assert response.status_code == 503
    assert "שירות השכתוב אינו זמין" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests — invalid diagnosis type returns 400
# ---------------------------------------------------------------------------


def test_invalid_diagnosis_type_returns_400() -> None:
    """Invalid diagnosis type should return 400 with Hebrew message."""
    with patch("app.api.rewrite.bedrock_service") as mock_bedrock:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/rewrite",
            json={
                "text": "משפט לשכתוב",
                "diagnosis_type": "nonexistent_type",
                "context": "",
            },
        )

    assert response.status_code == 400
    assert "סוג האבחנה אינו מוכר" in response.json()["detail"]
    # bedrock_service.rewrite should NOT have been called
    mock_bedrock.rewrite.assert_not_called()


def test_empty_diagnosis_type_returns_400() -> None:
    """Empty string diagnosis type should return 400."""
    with patch("app.api.rewrite.bedrock_service") as mock_bedrock:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/rewrite",
            json={
                "text": "משפט לשכתוב",
                "diagnosis_type": "",
                "context": "",
            },
        )

    assert response.status_code == 400
