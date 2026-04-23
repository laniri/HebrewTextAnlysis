# Feature: hebrew-writing-coach, Property 6: Delta score arithmetic
# Feature: hebrew-writing-coach, Property 7: Diagnosis set transitions
"""Property-based tests for the POST /api/revise endpoint (Properties 6–7).

All tests mock ``model_service.analyze()`` to return Hypothesis-generated
structured data, then verify properties against the API response.

Validates: Requirements 2.1, 2.2, 2.3
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.localization import DIAGNOSIS_MAP, INTERVENTION_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEBREW_ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת "
ALL_DIAGNOSIS_TYPES = list(DIAGNOSIS_MAP.keys())
SCORE_KEYS = ["difficulty", "style", "fluency", "cohesion", "complexity"]
SEVERITY_THRESHOLD = 0.3  # default from config


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def scores_strategy():
    """Generate a dict of 5 scores each in [0.0, 1.0]."""
    return st.fixed_dictionaries({
        key: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for key in SCORE_KEYS
    })


def diagnoses_dict_strategy():
    """Generate a dict mapping all 8 diagnosis types to severity floats."""
    return st.fixed_dictionaries({
        dtype: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for dtype in ALL_DIAGNOSIS_TYPES
    })


def _classify_highlight(severity: float) -> str:
    if severity > 0.7:
        return "red"
    if severity >= 0.4:
        return "yellow"
    return "none"


def minimal_analyze_result(scores: dict, diagnoses: dict) -> dict:
    """Build a minimal model_service.analyze() return value."""
    return {
        "scores": scores,
        "diagnoses": diagnoses,
        "interventions": [],
        "sentences": [
            {
                "index": 0,
                "text": "שלום",
                "char_start": 0,
                "char_end": 4,
                "complexity": 0.1,
                "highlight": "none",
            }
        ],
        "cohesion_gaps": [],
    }


# ---------------------------------------------------------------------------
# Helper: call the revise endpoint with mocked model_service
# ---------------------------------------------------------------------------

def call_revise(
    original_mock: dict,
    revised_mock: dict,
    original_text: str = "טקסט מקורי",
    edited_text: str = "טקסט מתוקן",
) -> "Response":
    """Call POST /api/revise with mocked model_service.analyze().

    The mock is configured to return ``original_mock`` on the first call
    and ``revised_mock`` on the second call.

    We patch model_service in both ``app.api.analyze`` (where it is
    defined) and ``app.api.revise`` (which imports it via
    ``from app.api.analyze import model_service``).
    """
    from unittest.mock import MagicMock
    from fastapi.testclient import TestClient
    from app.main import app

    mock_svc = MagicMock()
    mock_svc.analyze.side_effect = [original_mock, revised_mock]
    mock_svc.is_loaded = True

    with (
        patch("app.api.analyze.model_service", mock_svc),
        patch("app.api.revise.model_service", mock_svc),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/revise",
            json={"original_text": original_text, "edited_text": edited_text},
        )
    return response


# ---------------------------------------------------------------------------
# Property 6 – Delta score arithmetic
# ---------------------------------------------------------------------------

# **Validates: Requirements 2.1**
@given(
    original_text=st.text(alphabet=HEBREW_ALPHABET, min_size=1, max_size=200),
    edited_text=st.text(alphabet=HEBREW_ALPHABET, min_size=1, max_size=200),
    original_scores=scores_strategy(),
    revised_scores=scores_strategy(),
    original_diagnoses=diagnoses_dict_strategy(),
    revised_diagnoses=diagnoses_dict_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_delta_score_arithmetic(
    original_text: str,
    edited_text: str,
    original_scores: dict,
    revised_scores: dict,
    original_diagnoses: dict,
    revised_diagnoses: dict,
) -> None:
    """For any pair of non-empty Hebrew texts, deltas[key] ==
    revised_scores[key] - original_scores[key] for all 5 score keys."""
    assume(original_text.strip())
    assume(edited_text.strip())

    original_mock = minimal_analyze_result(original_scores, original_diagnoses)
    revised_mock = minimal_analyze_result(revised_scores, revised_diagnoses)

    response = call_revise(original_mock, revised_mock, original_text, edited_text)
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    data = response.json()
    deltas = data["deltas"]

    for key in SCORE_KEYS:
        expected_delta = revised_scores[key] - original_scores[key]
        actual_delta = deltas[key]
        assert abs(actual_delta - expected_delta) < 1e-6, (
            f"Delta mismatch for '{key}': expected {expected_delta}, "
            f"got {actual_delta}"
        )


# ---------------------------------------------------------------------------
# Property 7 – Diagnosis set transitions
# ---------------------------------------------------------------------------

# **Validates: Requirements 2.2, 2.3**
@given(
    original_scores=scores_strategy(),
    revised_scores=scores_strategy(),
    original_diagnoses=diagnoses_dict_strategy(),
    revised_diagnoses=diagnoses_dict_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_diagnosis_set_transitions(
    original_scores: dict,
    revised_scores: dict,
    original_diagnoses: dict,
    revised_diagnoses: dict,
) -> None:
    """resolved_diagnoses ⊆ original_active \\ revised_active and
    new_diagnoses ⊆ revised_active \\ original_active."""
    original_mock = minimal_analyze_result(original_scores, original_diagnoses)
    revised_mock = minimal_analyze_result(revised_scores, revised_diagnoses)

    response = call_revise(original_mock, revised_mock)
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    data = response.json()

    # Compute expected active sets using the same threshold as the endpoint
    original_active = {
        dtype for dtype, sev in original_diagnoses.items()
        if sev > SEVERITY_THRESHOLD
    }
    revised_active = {
        dtype for dtype, sev in revised_diagnoses.items()
        if sev > SEVERITY_THRESHOLD
    }

    resolved = set(data["resolved_diagnoses"])
    new = set(data["new_diagnoses"])

    # resolved ⊆ original_active \ revised_active
    expected_resolved = original_active - revised_active
    assert resolved <= expected_resolved, (
        f"resolved_diagnoses {resolved} is not a subset of "
        f"original \\ revised {expected_resolved}"
    )

    # new ⊆ revised_active \ original_active
    expected_new = revised_active - original_active
    assert new <= expected_new, (
        f"new_diagnoses {new} is not a subset of "
        f"revised \\ original {expected_new}"
    )

    # Additionally verify exact equality (the endpoint computes exactly these sets)
    assert resolved == expected_resolved, (
        f"resolved_diagnoses {resolved} != expected {expected_resolved}"
    )
    assert new == expected_new, (
        f"new_diagnoses {new} != expected {expected_new}"
    )
