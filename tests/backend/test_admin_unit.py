"""Unit tests for admin endpoints (/admin/config, /admin/models).

Tests correct/incorrect password authentication (200/401) and config
update behavior.

Requirements validated: 12.1, 12.2, 12.4, 12.5, 12.6.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


# ---------------------------------------------------------------------------
# Tests — admin endpoints with correct password (200)
# ---------------------------------------------------------------------------


def test_get_config_with_correct_password() -> None:
    """GET /admin/config with correct password should return 200."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(
        "/admin/config",
        headers={"X-Admin-Password": settings.ADMIN_PASSWORD},
    )

    assert response.status_code == 200
    data = response.json()
    assert "bedrock_model_id" in data
    assert "severity_threshold" in data
    assert "max_diagnoses_shown" in data
    assert "max_interventions_shown" in data


# ---------------------------------------------------------------------------
# Tests — admin endpoints with incorrect password (401)
# ---------------------------------------------------------------------------


def test_get_config_with_wrong_password() -> None:
    """GET /admin/config with wrong password should return 401."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(
        "/admin/config",
        headers={"X-Admin-Password": "wrong_password_12345"},
    )

    assert response.status_code == 401


def test_post_config_with_wrong_password() -> None:
    """POST /admin/config with wrong password should return 401."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/admin/config",
        headers={"X-Admin-Password": "wrong_password_12345"},
        json={"severity_threshold": 0.5},
    )

    assert response.status_code == 401


def test_get_config_without_password_header() -> None:
    """GET /admin/config without password header should return 422."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/admin/config")

    # FastAPI returns 422 for missing required header
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests — admin config update changes threshold
# ---------------------------------------------------------------------------


def test_admin_config_update_changes_threshold() -> None:
    """POST /admin/config should update the severity threshold."""
    original_threshold = settings.SEVERITY_THRESHOLD

    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/admin/config",
            headers={"X-Admin-Password": settings.ADMIN_PASSWORD},
            json={"severity_threshold": 0.6},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["severity_threshold"] == 0.6
        # Verify the settings object was actually updated
        assert settings.SEVERITY_THRESHOLD == 0.6
    finally:
        # Restore original threshold to avoid side effects on other tests
        settings.SEVERITY_THRESHOLD = original_threshold


def test_admin_config_update_changes_max_diagnoses() -> None:
    """POST /admin/config should update max_diagnoses_shown."""
    original_value = settings.MAX_DIAGNOSES_SHOWN

    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/admin/config",
            headers={"X-Admin-Password": settings.ADMIN_PASSWORD},
            json={"max_diagnoses_shown": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["max_diagnoses_shown"] == 5
        assert settings.MAX_DIAGNOSES_SHOWN == 5
    finally:
        settings.MAX_DIAGNOSES_SHOWN = original_value


def test_admin_config_partial_update() -> None:
    """POST /admin/config with partial fields should only update provided fields."""
    original_threshold = settings.SEVERITY_THRESHOLD
    original_max_diag = settings.MAX_DIAGNOSES_SHOWN

    try:
        client = TestClient(app, raise_server_exceptions=False)
        # Only update threshold, leave max_diagnoses_shown unchanged
        response = client.post(
            "/admin/config",
            headers={"X-Admin-Password": settings.ADMIN_PASSWORD},
            json={"severity_threshold": 0.8},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["severity_threshold"] == 0.8
        assert data["max_diagnoses_shown"] == original_max_diag
    finally:
        settings.SEVERITY_THRESHOLD = original_threshold
        settings.MAX_DIAGNOSES_SHOWN = original_max_diag
