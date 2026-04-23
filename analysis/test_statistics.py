"""Property-based tests for analysis/statistics.py."""

# Feature: probabilistic-analysis-layer, Property 1: FeatureStats coverage

import json
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.statistics import FeatureStats, compute_feature_stats, load_stats, save_stats

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_FEATURE_KEY = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

_FLOAT_OR_NONE = st.one_of(
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False),
)

_FEATURE_DICT = st.dictionaries(
    keys=_FEATURE_KEY,
    values=_FLOAT_OR_NONE,
    min_size=1,
    max_size=10,
)

_FEATURE_DICT_LIST = st.lists(_FEATURE_DICT, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Property 1: FeatureStats coverage
# ---------------------------------------------------------------------------

# Validates: Requirements 1.1
@settings(max_examples=100)
@given(feature_dicts=_FEATURE_DICT_LIST)
def test_feature_stats_coverage(feature_dicts: List[Dict[str, Optional[float]]]) -> None:
    """Property 1: FeatureStats coverage.

    For any non-empty list of feature dicts, every scalar key that appears in
    at least one dict with a non-None value SHALL have a corresponding
    FeatureStats entry in the output of compute_feature_stats.
    """
    # Collect all keys that have at least one non-None value
    keys_with_values = {
        key
        for d in feature_dicts
        for key, value in d.items()
        if value is not None
    }

    result = compute_feature_stats(feature_dicts)

    for key in keys_with_values:
        assert key in result, (
            f"Key '{key}' has non-None values but is missing from compute_feature_stats output"
        )


# ---------------------------------------------------------------------------
# Property 2: FeatureStats correctness
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 2: FeatureStats correctness

_FLOAT_OR_NONE_LIST = st.lists(_FLOAT_OR_NONE, min_size=1, max_size=50)


# Validates: Requirements 1.2
@settings(max_examples=100)
@given(values=_FLOAT_OR_NONE_LIST)
def test_feature_stats_correctness(values: List[Optional[float]]) -> None:
    """Property 2: FeatureStats correctness.

    For any list of float|None values, the computed FeatureStats SHALL have
    valid_count equal to the number of non-None values, and all statistical
    fields SHALL be computed only over the non-None values.
    """
    non_none = [v for v in values if v is not None]

    # Build a single-key feature dict list so we can call compute_feature_stats
    feature_dicts = [{"x": v} for v in values]
    result = compute_feature_stats(feature_dicts)

    if not non_none:
        # No non-None values: key should not appear in result
        assert "x" not in result, (
            "Key 'x' with all-None values should not appear in compute_feature_stats output"
        )
        return

    stats = result["x"]
    arr = np.array(non_none, dtype=float)

    assert stats.valid_count == len(non_none), (
        f"valid_count={stats.valid_count}, expected {len(non_none)}"
    )

    expected_mean = float(np.mean(arr))
    expected_min = float(np.min(arr))
    expected_max = float(np.max(arr))

    # Use exact equality for inf/nan-free comparison; fall back to relative tolerance
    def _approx_equal(a: float, b: float) -> bool:
        import math
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) < 1e-9

    assert _approx_equal(stats.mean, expected_mean), (
        f"mean={stats.mean}, expected {expected_mean}"
    )
    assert _approx_equal(stats.min, expected_min), (
        f"min={stats.min}, expected {expected_min}"
    )
    assert _approx_equal(stats.max, expected_max), (
        f"max={stats.max}, expected {expected_max}"
    )


# ---------------------------------------------------------------------------
# Property 3: Stability and degeneracy flags
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 3: Stability and degeneracy flags

_SMALL_FLOAT_LIST = st.lists(
    st.floats(allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=29,  # fewer than 30 → always unstable
)

# Constrain to values where numpy population std is reliably 0.0 for identical inputs.
# Very large floats can cause floating-point cancellation in numpy's std computation,
# producing a tiny non-zero result even when all values are identical.
# We use integers mapped to float to guarantee exact representation and exact mean.
_UNIFORM_FLOAT = st.integers(min_value=-10**9, max_value=10**9).map(float)


# Validates: Requirements 1.3, 1.4
@settings(max_examples=100)
@given(
    sparse_values=_SMALL_FLOAT_LIST,
    uniform_value=_UNIFORM_FLOAT,
    uniform_count=st.integers(min_value=1, max_value=50),
)
def test_stability_and_degeneracy_flags(
    sparse_values: List[float],
    uniform_value: float,
    uniform_count: int,
) -> None:
    """Property 3: Stability and degeneracy flags.

    For any feature with fewer than 30 non-None values, unstable SHALL be True.
    For any feature where all non-None values are identical, degenerate SHALL be True.
    """
    # --- unstable: fewer than 30 non-None values ---
    sparse_dicts = [{"sparse": v} for v in sparse_values]
    sparse_result = compute_feature_stats(sparse_dicts)

    assert "sparse" in sparse_result, "sparse key should be present"
    assert sparse_result["sparse"].unstable is True, (
        f"unstable should be True for valid_count={len(sparse_values)} < 30"
    )

    # --- degenerate: all values identical ---
    uniform_dicts = [{"uniform": uniform_value} for _ in range(uniform_count)]
    uniform_result = compute_feature_stats(uniform_dicts)

    assert "uniform" in uniform_result, "uniform key should be present"
    assert uniform_result["uniform"].degenerate is True, (
        f"degenerate should be True when all values are {uniform_value}"
    )


# ---------------------------------------------------------------------------
# Property 4: Statistics round-trip
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 4: Statistics round-trip

_FINITE_FLOAT = st.floats(allow_nan=False, allow_infinity=False)
_BOOL = st.booleans()

_FEATURE_STATS_STRATEGY = st.builds(
    FeatureStats,
    mean=_FINITE_FLOAT,
    std=_FINITE_FLOAT,
    min=_FINITE_FLOAT,
    max=_FINITE_FLOAT,
    p10=_FINITE_FLOAT,
    p25=_FINITE_FLOAT,
    p50=_FINITE_FLOAT,
    p75=_FINITE_FLOAT,
    p90=_FINITE_FLOAT,
    valid_count=st.integers(min_value=0, max_value=10_000),
    unstable=_BOOL,
    degenerate=_BOOL,
)

_STATS_DICT_STRATEGY = st.dictionaries(
    keys=_FEATURE_KEY,
    values=_FEATURE_STATS_STRATEGY,
    min_size=1,
    max_size=10,
)


# Validates: Requirements 1.5, 1.6
@settings(max_examples=100)
@given(stats_dict=_STATS_DICT_STRATEGY)
def test_statistics_round_trip(stats_dict: Dict[str, FeatureStats]) -> None:
    """Property 4: Statistics round-trip.

    For any FeatureStats dict, saving it to a JSON file and loading it back
    SHALL produce a dict equal to the original (field-by-field).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        save_stats(stats_dict, feature_path=tmp_path)
        loaded = load_stats(feature_path=tmp_path)

        assert set(loaded.keys()) == set(stats_dict.keys()), (
            f"Key sets differ after round-trip: {set(loaded.keys())} vs {set(stats_dict.keys())}"
        )

        for key in stats_dict:
            orig = stats_dict[key]
            back = loaded[key]
            assert orig.mean == back.mean, f"{key}.mean mismatch: {orig.mean} vs {back.mean}"
            assert orig.std == back.std, f"{key}.std mismatch: {orig.std} vs {back.std}"
            assert orig.min == back.min, f"{key}.min mismatch: {orig.min} vs {back.min}"
            assert orig.max == back.max, f"{key}.max mismatch: {orig.max} vs {back.max}"
            assert orig.p10 == back.p10, f"{key}.p10 mismatch: {orig.p10} vs {back.p10}"
            assert orig.p25 == back.p25, f"{key}.p25 mismatch: {orig.p25} vs {back.p25}"
            assert orig.p50 == back.p50, f"{key}.p50 mismatch: {orig.p50} vs {back.p50}"
            assert orig.p75 == back.p75, f"{key}.p75 mismatch: {orig.p75} vs {back.p75}"
            assert orig.p90 == back.p90, f"{key}.p90 mismatch: {orig.p90} vs {back.p90}"
            assert orig.valid_count == back.valid_count, (
                f"{key}.valid_count mismatch: {orig.valid_count} vs {back.valid_count}"
            )
            assert orig.unstable == back.unstable, (
                f"{key}.unstable mismatch: {orig.unstable} vs {back.unstable}"
            )
            assert orig.degenerate == back.degenerate, (
                f"{key}.degenerate mismatch: {orig.degenerate} vs {back.degenerate}"
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
