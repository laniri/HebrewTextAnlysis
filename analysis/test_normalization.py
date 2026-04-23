"""Property-based tests for analysis/normalization.py."""

# Feature: probabilistic-analysis-layer, Property 5: soft_score formula correctness

from math import exp

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from analysis.normalization import soft_score

_FINITE_FLOAT = st.floats(allow_nan=False, allow_infinity=False)
_POSITIVE_STD = st.floats(min_value=1e-10, allow_nan=False, allow_infinity=False)


# Validates: Requirements 2.1
@settings(max_examples=100)
@given(v=_FINITE_FLOAT, m=_FINITE_FLOAT, s=_POSITIVE_STD)
def test_soft_score_formula_correctness(v: float, m: float, s: float) -> None:
    """Property 5: soft_score formula correctness.

    For any finite float v, m, and positive s,
    soft_score(v, m, s) SHALL equal 1 / (1 + exp(-(v - m) / s)).
    """
    # Skip inputs where the z-score would overflow exp() — outside the domain
    # where the formula is numerically representable as a float.
    z = (v - m) / s
    assume(abs(z) < 709.0)

    expected = 1.0 / (1.0 + exp(-z))
    result = soft_score(v, m, s)
    assert abs(result - expected) < 1e-12, (
        f"soft_score({v}, {m}, {s}) = {result}, expected {expected}"
    )


# Feature: probabilistic-analysis-layer, Property 6: soft_score range invariant

_NON_NEGATIVE_STD = st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)


# Validates: Requirements 2.2, 2.3, 2.5
@settings(max_examples=100)
@given(v=_FINITE_FLOAT, m=_FINITE_FLOAT, s=_NON_NEGATIVE_STD)
def test_soft_score_range_invariant(v: float, m: float, s: float) -> None:
    """Property 6: soft_score range invariant.

    For any finite float v, m, and non-negative s,
    soft_score(v, m, s) SHALL return a value in [0.0, 1.0].
    When s == 0, the result SHALL be exactly 0.5.
    """
    result = soft_score(v, m, s)
    assert 0.0 <= result <= 1.0, (
        f"soft_score({v}, {m}, {s}) = {result} is outside [0.0, 1.0]"
    )
    if s == 0.0:
        assert result == 0.5, (
            f"soft_score({v}, {m}, 0) = {result}, expected exactly 0.5"
        )
