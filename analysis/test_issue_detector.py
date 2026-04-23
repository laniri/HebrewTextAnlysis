"""Property-based tests for analysis/issue_detector.py.

Feature: probabilistic-analysis-layer
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from analysis.issue_detector import _jaccard, detect_issues
from analysis.issue_models import Issue
from analysis.sentence_metrics import SentenceMetrics
from analysis.statistics import FeatureStats

# ---------------------------------------------------------------------------
# Helpers / shared strategies
# ---------------------------------------------------------------------------

VALID_GROUPS = {"morphology", "syntax", "lexicon", "structure", "discourse", "style"}

_float_or_none = st.one_of(
    st.none(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

_raw_features_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=30),
    values=_float_or_none,
    min_size=0,
    max_size=20,
)


def _sentence_metrics_strategy(index: int):
    return st.builds(
        SentenceMetrics,
        index=st.just(index),
        token_count=st.integers(min_value=1, max_value=200),
        tree_depth=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        lemma_set=st.frozensets(st.text(min_size=1, max_size=20), min_size=0, max_size=30),
    )


def _sentence_metrics_list_strategy(min_len: int = 0, max_len: int = 20):
    return st.integers(min_value=min_len, max_value=max_len).flatmap(
        lambda n: st.tuples(*[_sentence_metrics_strategy(i) for i in range(n)])
        if n > 0
        else st.just(())
    ).map(list)


_FULL_RAW_FEATURES: Dict[str, float] = {
    "agreement_error_rate": 0.05,
    "morphological_ambiguity": 0.3,
    "binyan_entropy": 1.5,
    "dependency_distance_variance": 2.0,
    "right_branching_ratio": 0.6,
    "lemma_diversity": 0.7,
    "type_token_ratio": 0.5,
    "rare_word_ratio": 0.1,
    "content_word_ratio": 0.55,
    "sentence_length_variance": 10.0,
    "punctuation_ratio": 0.08,
    "missing_terminal_punctuation_ratio": 0.2,
    "short_sentence_ratio": 0.15,
    "connective_ratio": 0.04,
    "pronoun_to_noun_ratio": 0.25,
    "pos_distribution_variance": 0.03,
    "sentence_length_trend": 0.01,
}


def _make_feature_stats_from_raw(raw_features: Dict) -> Dict[str, FeatureStats]:
    result: Dict[str, FeatureStats] = {}
    for key, val in raw_features.items():
        if val is not None:
            result[key] = FeatureStats(
                mean=float(val),
                std=1.0,
                min=0.0,
                max=1.0,
                p10=0.0,
                p25=0.0,
                p50=float(val),
                p75=1.0,
                p90=1.0,
                valid_count=50,
                unstable=False,
                degenerate=False,
            )
    return result


# ---------------------------------------------------------------------------
# Property 7: Issue field invariants
# Validates: Requirements 3.2, 3.3, 3.4
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 7: Issue field invariants
@given(
    raw_features=_raw_features_strategy,
    sentence_metrics=_sentence_metrics_list_strategy(min_len=0, max_len=10),
)
@settings(max_examples=100)
def test_property7_issue_field_invariants(
    raw_features: Dict,
    sentence_metrics: List[SentenceMetrics],
):
    """Validates: Requirements 3.2, 3.3, 3.4"""
    feature_stats = _make_feature_stats_from_raw(raw_features)
    issues = detect_issues(raw_features, sentence_metrics, feature_stats)

    for issue in issues:
        assert 0.0 <= issue.severity <= 1.0, (
            f"severity {issue.severity} out of [0,1] for issue type={issue.type}"
        )
        assert 0.0 <= issue.confidence <= 1.0, (
            f"confidence {issue.confidence} out of [0,1] for issue type={issue.type}"
        )
        assert issue.group in VALID_GROUPS, (
            f"group '{issue.group}' not in valid groups for issue type={issue.type}"
        )


# ---------------------------------------------------------------------------
# Property 9: Span correctness
# Validates: Requirements 5.4, 6.2, 6.5, 7.4, 8.4, 9.2, 9.5, 10.3, 13.1, 13.2, 13.3
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 9: Span correctness
@given(
    sentence_metrics=_sentence_metrics_list_strategy(min_len=0, max_len=15),
)
@settings(max_examples=100)
def test_property9_span_correctness(sentence_metrics: List[SentenceMetrics]):
    """Validates: Requirements 5.4, 6.2, 6.5, 7.4, 8.4, 9.2, 9.5, 10.3, 13.1, 13.2, 13.3"""
    N = len(sentence_metrics)
    raw_features = dict(_FULL_RAW_FEATURES)
    feature_stats = _make_feature_stats_from_raw(raw_features)

    issues = detect_issues(raw_features, sentence_metrics, feature_stats)

    sentence_complexity_issues = [i for i in issues if i.type == "sentence_complexity"]
    weak_cohesion_issues = [i for i in issues if i.type == "weak_cohesion"]
    document_level_issues = [
        i for i in issues
        if i.type not in ("sentence_complexity", "weak_cohesion")
    ]

    for issue in document_level_issues:
        assert issue.span == (0, N), (
            f"Document-level issue '{issue.type}' has span {issue.span}, expected (0, {N})"
        )

    assert len(sentence_complexity_issues) == N
    for sm, issue in zip(sentence_metrics, sentence_complexity_issues):
        assert issue.span == (sm.index,)

    expected_cohesion_count = max(0, N - 1)
    assert len(weak_cohesion_issues) == expected_cohesion_count
    for j, issue in enumerate(weak_cohesion_issues):
        assert issue.span == (j, j + 1)


# ---------------------------------------------------------------------------
# Property 10: Confidence formula
# Validates: Requirements 11.1, 11.2, 11.3
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 10: Confidence formula
@given(
    sentence_metrics=_sentence_metrics_list_strategy(min_len=1, max_len=5),
)
@settings(max_examples=100)
def test_property10_confidence_formula(sentence_metrics: List[SentenceMetrics]):
    """Validates: Requirements 11.1, 11.2, 11.3"""
    all_none_features: Dict[str, Optional[float]] = {
        "agreement_error_rate": None,
        "morphological_ambiguity": None,
        "binyan_entropy": None,
        "dependency_distance_variance": None,
        "right_branching_ratio": None,
        "lemma_diversity": None,
        "type_token_ratio": None,
        "rare_word_ratio": None,
        "content_word_ratio": None,
        "sentence_length_variance": None,
        "punctuation_ratio": None,
        "missing_terminal_punctuation_ratio": None,
        "short_sentence_ratio": None,
        "connective_ratio": None,
        "pronoun_to_noun_ratio": None,
        "pos_distribution_variance": None,
        "sentence_length_trend": None,
    }
    issues_all_none = detect_issues(all_none_features, sentence_metrics, {})
    doc_level_all_none = [
        i for i in issues_all_none
        if i.type not in ("sentence_complexity", "weak_cohesion")
    ]
    assert len(doc_level_all_none) == 0

    raw_features = dict(_FULL_RAW_FEATURES)
    feature_stats = _make_feature_stats_from_raw(raw_features)
    issues_full = detect_issues(raw_features, sentence_metrics, feature_stats)

    for issue in issues_full:
        assert 0.0 <= issue.confidence <= 1.0
        assert issue.confidence <= issue.severity + 1e-9

    all_stat_keys = list(raw_features.keys()) + [
        "avg_sentence_length", "avg_tree_depth", "sentence_overlap"
    ]
    degenerate_stats: Dict[str, FeatureStats] = {}
    for key in all_stat_keys:
        degenerate_stats[key] = FeatureStats(
            mean=0.5, std=0.0, min=0.0, max=1.0,
            p10=0.0, p25=0.0, p50=0.5, p75=1.0, p90=1.0,
            valid_count=50, unstable=False, degenerate=True,
        )
    issues_degenerate = detect_issues(raw_features, sentence_metrics, degenerate_stats)
    for issue in issues_degenerate:
        assert issue.confidence == 0.0, (
            f"Expected confidence=0.0 with degenerate stats, got {issue.confidence} "
            f"for issue type={issue.type}"
        )


# ---------------------------------------------------------------------------
# Property 13: Jaccard similarity correctness
# Validates: Requirements 9.1
# ---------------------------------------------------------------------------

# Feature: probabilistic-analysis-layer, Property 13: Jaccard similarity correctness
@given(
    a=st.frozensets(st.integers(min_value=0, max_value=100), min_size=0, max_size=20),
    b=st.frozensets(st.integers(min_value=0, max_value=100), min_size=0, max_size=20),
)
@settings(max_examples=100)
def test_property13_jaccard_similarity_correctness(a: frozenset, b: frozenset):
    """Validates: Requirements 9.1"""
    result = _jaccard(a, b)

    union = a | b
    if not union:
        assert result == 0.0
    else:
        expected = len(a & b) / len(union)
        assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)

    assert 0.0 <= result <= 1.0
