# Feature: hebrew-writing-coach, Property 10: Diagnosis display filtering and ordering
"""Property-based test for diagnosis display filtering and ordering.

Tests the pure filtering/sorting/capping logic used by the analyze
endpoint to determine which diagnoses are displayed to the user.

Validates: Requirements 7.2, 7.4
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.localization import DIAGNOSIS_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DIAGNOSIS_TYPES = list(DIAGNOSIS_MAP.keys())


# ---------------------------------------------------------------------------
# The pure function under test — extracted from app/api/analyze.py logic
# ---------------------------------------------------------------------------

def filter_sort_cap_diagnoses(
    diagnoses: dict[str, float],
    threshold: float,
    max_count: int,
) -> list[tuple[str, float]]:
    """Filter diagnoses by severity > threshold, sort by severity
    descending, and cap at max_count.

    This mirrors the logic in ``app.api.analyze.analyze()``:
        filtered = [(dtype, sev) for dtype, sev in diagnoses.items() if sev > threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        displayed = filtered[:max_count]
    """
    filtered = [
        (dtype, severity)
        for dtype, severity in diagnoses.items()
        if severity > threshold
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:max_count]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def diagnosis_list_strategy(draw):
    """Generate a dict mapping a subset of diagnosis types to severities."""
    # Pick a random subset of diagnosis types (1 to 8)
    num_types = draw(st.integers(min_value=0, max_value=len(ALL_DIAGNOSIS_TYPES)))
    selected_types = draw(
        st.lists(
            st.sampled_from(ALL_DIAGNOSIS_TYPES),
            min_size=num_types,
            max_size=num_types,
            unique=True,
        )
    )
    diagnoses = {}
    for dtype in selected_types:
        severity = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        diagnoses[dtype] = severity
    return diagnoses


# ---------------------------------------------------------------------------
# Property 10 – Diagnosis display filtering and ordering
# ---------------------------------------------------------------------------

# **Validates: Requirements 7.2, 7.4**
@given(
    diagnoses=diagnosis_list_strategy(),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    max_count=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_diagnosis_display_filtering_and_ordering(
    diagnoses: dict[str, float],
    threshold: float,
    max_count: int,
) -> None:
    """Displayed diagnoses include only those with severity > threshold,
    are sorted by severity descending, and are capped at max_count."""
    displayed = filter_sort_cap_diagnoses(diagnoses, threshold, max_count)

    # 1. Only diagnoses with severity > threshold are included
    for dtype, severity in displayed:
        assert severity > threshold, (
            f"Diagnosis '{dtype}' with severity {severity} should not be "
            f"displayed (threshold={threshold})"
        )

    # 2. No diagnosis above threshold is missing (up to the cap)
    all_above_threshold = [
        (dtype, sev) for dtype, sev in diagnoses.items() if sev > threshold
    ]
    all_above_threshold.sort(key=lambda x: x[1], reverse=True)
    expected = all_above_threshold[:max_count]
    assert len(displayed) == len(expected), (
        f"Expected {len(expected)} displayed diagnoses, got {len(displayed)}"
    )

    # 3. Sorted by severity descending
    severities = [sev for _, sev in displayed]
    for i in range(1, len(severities)):
        assert severities[i - 1] >= severities[i], (
            f"Diagnoses not sorted descending: "
            f"severity[{i-1}]={severities[i-1]} < severity[{i}]={severities[i]}"
        )

    # 4. Capped at max_count
    assert len(displayed) <= max_count, (
        f"Displayed {len(displayed)} diagnoses, exceeds max {max_count}"
    )

    # 5. The displayed set matches the expected top-N
    displayed_types = {dtype for dtype, _ in displayed}
    expected_types = {dtype for dtype, _ in expected}
    assert displayed_types == expected_types, (
        f"Displayed types {displayed_types} != expected {expected_types}"
    )
