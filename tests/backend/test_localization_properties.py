# Feature: hebrew-writing-coach, Property 11: Localization completeness
"""Property-based tests for localization completeness.

Validates: Requirements 11.2, 11.3, 11.4

Property 11: For any of the 8 diagnosis types, localization produces
non-empty Hebrew label, explanation, action list, and tip. For any of
the 5 score names, localization produces a non-empty Hebrew label. For
any of the 4 intervention types, localization produces a non-empty
Hebrew label.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.localization import (
    DIAGNOSIS_MAP,
    INTERVENTION_MAP,
    SCORE_NAME_MAP,
    localize_diagnosis,
    localize_intervention,
    localize_score_name,
)

# ---------------------------------------------------------------------------
# Collected type lists for sampled_from strategies
# ---------------------------------------------------------------------------

ALL_DIAGNOSIS_TYPES = list(DIAGNOSIS_MAP.keys())
ALL_SCORE_NAMES = list(SCORE_NAME_MAP.keys())
ALL_INTERVENTION_TYPES = list(INTERVENTION_MAP.keys())


# ---------------------------------------------------------------------------
# Property 11a – Diagnosis localization completeness
# ---------------------------------------------------------------------------

# **Validates: Requirements 11.2**
@given(
    diagnosis_type=st.sampled_from(ALL_DIAGNOSIS_TYPES),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_diagnosis_localization_produces_nonempty_hebrew_fields(
    diagnosis_type: str,
    severity: float,
) -> None:
    """For any of the 8 diagnosis types, localize_diagnosis returns
    non-empty Hebrew label, explanation, action list, and tip."""
    result = localize_diagnosis(diagnosis_type, severity)

    assert result.type == diagnosis_type
    assert result.label_he, f"label_he is empty for {diagnosis_type}"
    assert result.explanation_he, f"explanation_he is empty for {diagnosis_type}"
    assert len(result.actions_he) > 0, f"actions_he is empty for {diagnosis_type}"
    assert all(
        action for action in result.actions_he
    ), f"actions_he contains empty string for {diagnosis_type}"
    assert result.tip_he, f"tip_he is empty for {diagnosis_type}"


# ---------------------------------------------------------------------------
# Property 11b – Score name localization completeness
# ---------------------------------------------------------------------------

# **Validates: Requirements 11.3**
@given(score_name=st.sampled_from(ALL_SCORE_NAMES))
@settings(max_examples=100)
def test_score_name_localization_produces_nonempty_hebrew_label(
    score_name: str,
) -> None:
    """For any of the 5 score names, localize_score_name returns a
    non-empty Hebrew label."""
    label = localize_score_name(score_name)

    assert isinstance(label, str)
    assert label, f"Hebrew label is empty for score name '{score_name}'"


# ---------------------------------------------------------------------------
# Property 11c – Intervention localization completeness
# ---------------------------------------------------------------------------

# **Validates: Requirements 11.4**
@given(intervention_type=st.sampled_from(ALL_INTERVENTION_TYPES))
@settings(max_examples=100)
def test_intervention_localization_produces_nonempty_hebrew_label(
    intervention_type: str,
) -> None:
    """For any of the 4 intervention types, localize_intervention returns
    a non-empty Hebrew label (via actions_he which contains the localized
    content)."""
    intervention_dict = {
        "type": intervention_type,
        "priority": 0.5,
        "target_diagnosis": "low_cohesion",
    }
    result = localize_intervention(intervention_dict)

    assert result.type == intervention_type
    # The INTERVENTION_MAP stores label_he for each type — verify it exists
    assert INTERVENTION_MAP[intervention_type]["label_he"], (
        f"label_he is empty for intervention type '{intervention_type}'"
    )
    assert len(result.actions_he) > 0, (
        f"actions_he is empty for intervention type '{intervention_type}'"
    )
    assert all(
        action for action in result.actions_he
    ), f"actions_he contains empty string for intervention type '{intervention_type}'"
