"""Unit tests for GET /api/examples and GET /api/examples/{id} endpoints.

Tests that the examples list endpoint returns data and that an invalid
example id returns 404.

Requirements validated: 4.1, 4.4.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ExampleSummary, ExampleFull


# ---------------------------------------------------------------------------
# Tests — GET /api/examples returns list
# ---------------------------------------------------------------------------


def test_get_examples_returns_list() -> None:
    """GET /api/examples should return a list of example summaries."""
    mock_summaries = [
        ExampleSummary(
            id="tweet_investing",
            label="ציוץ על השקעות",
            category="ציוץ",
            preview="טקסט לדוגמה...",
        ),
        ExampleSummary(
            id="news_security",
            label="כתבה על ביטחון",
            category="כתבה חדשותית",
            preview="טקסט חדשותי...",
        ),
    ]

    with patch("app.api.examples.example_service") as mock_svc:
        mock_svc.list_examples.return_value = mock_summaries
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/examples")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "tweet_investing"
    assert data[1]["id"] == "news_security"
    # Each item should have the expected keys
    for item in data:
        assert "id" in item
        assert "label" in item
        assert "category" in item
        assert "preview" in item


# ---------------------------------------------------------------------------
# Tests — GET /api/examples/{id} with invalid id returns 404
# ---------------------------------------------------------------------------


def test_get_example_invalid_id_returns_404() -> None:
    """GET /api/examples/{id} with unknown id should return 404 with Hebrew message."""
    with patch("app.api.examples.example_service") as mock_svc:
        mock_svc.get_example.return_value = None
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/examples/nonexistent_id")

    assert response.status_code == 404
    assert "דוגמה לא נמצאה" in response.json()["detail"]


def test_get_example_valid_id_returns_full() -> None:
    """GET /api/examples/{id} with valid id should return full example."""
    mock_example = ExampleFull(
        id="tweet_investing",
        label="ציוץ על השקעות",
        category="ציוץ",
        text="טקסט מלא של הדוגמה לבדיקה",
    )

    with patch("app.api.examples.example_service") as mock_svc:
        mock_svc.get_example.return_value = mock_example
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/examples/tweet_investing")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "tweet_investing"
    assert data["text"] == "טקסט מלא של הדוגמה לבדיקה"
