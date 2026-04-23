# Feature: diagnosis-interventions, Property 1: Helper function correctness

"""Property-based tests for helper functions in analysis/diagnosis_engine.py.

Tests _get_issues, _max_severity, and _weighted_mean using Hypothesis.
**Validates: Requirements 3.1, 3.2, 3.3**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.diagnosis_engine import _get_issues, _max_severity, _weighted_mean
from analysis.issue_models import Issue

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

GROUPS = ["morphology", "syntax", "lexicon", "structure", "discourse", "style"]

ISSUE_TYPES = [
    "agreement_errors", "morphological_ambiguity", "low_morphological_diversity",
    "sentence_complexity", "dependency_spread", "excessive_branching",
    "low_lexical_diversity", "rare_word_overuse", "low_content_density",
    "sentence_length_variability", "punctuation_issues", "fragmentation",
    "weak_cohesion", "missing_connectives", "pronoun_ambiguity",
    "structural_inconsistency", "sentence_progression_drift",
]

issue_strategy = st.builds(
    Issue,
    type=st.sampled_from(ISSUE_TYPES),
    group=st.sampled_from(GROUPS),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    span=st.one_of(
        st.tuples(st.integers(min_value=0, max_value=100)),
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        ),
    ),
    evidence=st.dictionaries(
        keys=st.text(min_size=1, max_size=30),
        values=st.floats(allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
)

issues_strategy = st.lists(issue_strategy, max_size=15)

# ---------------------------------------------------------------------------
# Property 1a: _get_issues returns exact subset matching type
# Validates: Requirement 3.1
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, target_type=st.sampled_from(ISSUE_TYPES))
@settings(max_examples=100)
def test_get_issues_returns_exact_matching_subset(issues, target_type):
    """_get_issues returns exactly the issues whose type matches target_type."""
    result = _get_issues(issues, target_type)

    # Every returned issue must have the target type
    for issue in result:
        assert issue.type == target_type

    # Count must match manual filter
    expected_count = sum(1 for i in issues if i.type == target_type)
    assert len(result) == expected_count

    # The returned issues must be the same objects from the input
    result_ids = {id(i) for i in result}
    for i in issues:
        if i.type == target_type:
            assert id(i) in result_ids


# ---------------------------------------------------------------------------
# Property 1b: _max_severity returns max severity or 0.0
# Validates: Requirement 3.2
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, target_type=st.sampled_from(ISSUE_TYPES))
@settings(max_examples=100)
def test_max_severity_returns_max_or_zero(issues, target_type):
    """_max_severity returns the maximum severity among matching issues, or 0.0."""
    result = _max_severity(issues, target_type)

    matching = [i for i in issues if i.type == target_type]
    if not matching:
        assert result == 0.0
    else:
        expected = max(i.severity for i in matching)
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Property 1c: _weighted_mean returns sum(v*w)/sum(w) or 0.0 when empty
# Validates: Requirement 3.3
# ---------------------------------------------------------------------------


@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=10,
    ),
    weights=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_weighted_mean_formula_or_zero(values, weights):
    """_weighted_mean returns sum(v*w)/sum(w) or 0.0 when values is empty."""
    # Ensure values and weights have the same length (as required by zip)
    min_len = min(len(values), len(weights))
    values = values[:min_len]
    weights = weights[:min_len]

    result = _weighted_mean(values, weights)

    if not values:
        assert result == 0.0
    else:
        weight_sum = sum(weights)
        if weight_sum == 0.0:
            assert result == 0.0
        else:
            expected = sum(v * w for v, w in zip(values, weights)) / weight_sum
            assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Feature: diagnosis-interventions, Property 2: Diagnosis severity formula and threshold correctness
# ---------------------------------------------------------------------------

"""
Property-based tests for all 8 diagnosis rules in analysis/diagnosis_engine.py.

Tests severity formula correctness, activation threshold, confidence calculation,
supporting_issues/spans population, and None score substitution.

**Validates: Requirements 4.1–11.4, 21.1–21.6**
"""

from analysis.diagnosis_engine import (
    _diagnose_low_lexical_diversity,
    _diagnose_pronoun_overuse,
    _diagnose_low_cohesion,
    _diagnose_sentence_over_complexity,
    _diagnose_structural_inconsistency,
    _diagnose_low_morphological_richness,
    _diagnose_fragmented_writing,
    _diagnose_punctuation_deficiency,
    _safe_score,
)

# ---------------------------------------------------------------------------
# Strategies for Property 2
# ---------------------------------------------------------------------------

scores_strategy = st.fixed_dictionaries({
    "difficulty": st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    "style": st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    "fluency": st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    "cohesion": st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    "complexity": st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
})


# ---------------------------------------------------------------------------
# Helpers for expected computation
# ---------------------------------------------------------------------------

def _expected_max_severity(issues, issue_type):
    """Recompute max severity from scratch (no dependency on SUT helpers)."""
    matching = [i for i in issues if i.type == issue_type]
    return max((i.severity for i in matching), default=0.0)


def _expected_mean_severity(issues, issue_type):
    """Recompute mean severity from scratch (no dependency on SUT helpers)."""
    matching = [i for i in issues if i.type == issue_type]
    if not matching:
        return 0.0
    return sum(i.severity for i in matching) / len(matching)


def _expected_weighted_mean(values, weights):
    """Recompute weighted mean from scratch."""
    if not values:
        return 0.0
    total = sum(v * w for v, w in zip(values, weights))
    w_sum = sum(weights)
    if w_sum == 0.0:
        return 0.0
    return total / w_sum


def _expected_safe_score(scores, key):
    """Recompute safe score from scratch."""
    value = scores.get(key)
    if value is None:
        return 0.0, {f"{key}_missing": True}
    return value, {}


# ---------------------------------------------------------------------------
# Property 2a: _diagnose_low_lexical_diversity
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_low_lexical_diversity_severity_and_threshold(issues, scores):
    """Rule 1: weighted mean of low_lexical_diversity(0.7) + low_content_density(0.3), threshold 0.6."""
    result = _diagnose_low_lexical_diversity(issues, scores)

    sev_lex = _expected_max_severity(issues, "low_lexical_diversity")
    sev_den = _expected_max_severity(issues, "low_content_density")
    expected_severity = _expected_weighted_mean([sev_lex, sev_den], [0.7, 0.3])

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "low_lexical_diversity"

        # Confidence = min confidence of supporting issues
        supporting = [i for i in issues if i.type in ("low_lexical_diversity", "low_content_density")]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Supporting issues — implementation groups low_lexical_diversity first, then low_content_density
        supporting_ordered = (
            [i for i in issues if i.type == "low_lexical_diversity"]
            + [i for i in issues if i.type == "low_content_density"]
        )
        expected_types = list(dict.fromkeys(i.type for i in supporting_ordered))
        assert result.supporting_issues == expected_types

        # Supporting spans — same grouped order
        expected_spans = [i.span for i in supporting_ordered]
        assert result.supporting_spans == expected_spans

        # Evidence
        assert abs(result.evidence["max_low_lexical_diversity_severity"] - sev_lex) < 1e-9
        assert abs(result.evidence["max_low_content_density_severity"] - sev_den) < 1e-9


# ---------------------------------------------------------------------------
# Property 2b: _diagnose_pronoun_overuse
# Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 21.1
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_pronoun_overuse_severity_and_threshold(issues, scores):
    """Rule 2: weighted mean of pronoun_ambiguity(0.8) + cohesion(0.2), threshold 0.6."""
    result = _diagnose_pronoun_overuse(issues, scores)

    sev_pron = _expected_max_severity(issues, "pronoun_ambiguity")
    cohesion_val, cohesion_ev = _expected_safe_score(scores, "cohesion")
    expected_severity = _expected_weighted_mean([sev_pron, cohesion_val], [0.8, 0.2])

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "pronoun_overuse"

        # Confidence = min confidence of supporting pronoun_ambiguity issues
        supporting = [i for i in issues if i.type == "pronoun_ambiguity"]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Evidence includes cohesion_score
        assert abs(result.evidence["cohesion_score"] - cohesion_val) < 1e-9

        # None score substitution
        if scores.get("cohesion") is None:
            assert result.evidence.get("cohesion_missing") is True
        else:
            assert "cohesion_missing" not in result.evidence


# ---------------------------------------------------------------------------
# Property 2c: _diagnose_low_cohesion
# Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_low_cohesion_severity_and_threshold(issues, scores):
    """Rule 3: weighted mean of weak_cohesion(0.6) + missing_connectives(0.4), threshold 0.6."""
    result = _diagnose_low_cohesion(issues, scores)

    sev_coh = _expected_max_severity(issues, "weak_cohesion")
    sev_con = _expected_max_severity(issues, "missing_connectives")
    expected_severity = _expected_weighted_mean([sev_coh, sev_con], [0.6, 0.4])

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "low_cohesion"

        # Confidence = min confidence of supporting issues
        supporting = [i for i in issues if i.type in ("weak_cohesion", "missing_connectives")]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Supporting issues — implementation groups weak_cohesion first, then missing_connectives
        supporting_ordered = (
            [i for i in issues if i.type == "weak_cohesion"]
            + [i for i in issues if i.type == "missing_connectives"]
        )
        expected_types = list(dict.fromkeys(i.type for i in supporting_ordered))
        assert result.supporting_issues == expected_types

        # Supporting spans — same grouped order
        expected_spans = [i.span for i in supporting_ordered]
        assert result.supporting_spans == expected_spans

        # Evidence
        assert abs(result.evidence["max_weak_cohesion_severity"] - sev_coh) < 1e-9
        assert abs(result.evidence["max_missing_connectives_severity"] - sev_con) < 1e-9


# ---------------------------------------------------------------------------
# Property 2d: _diagnose_sentence_over_complexity
# Validates: Requirements 7.1, 7.2, 7.3, 7.4, 21.2
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_sentence_over_complexity_severity_and_threshold(issues, scores):
    """Rule 4: weighted mean of mean_sentence_complexity(0.7) + difficulty(0.3), threshold 0.65."""
    result = _diagnose_sentence_over_complexity(issues, scores)

    sev_comp = _expected_mean_severity(issues, "sentence_complexity")
    diff_val, diff_ev = _expected_safe_score(scores, "difficulty")
    expected_severity = _expected_weighted_mean([sev_comp, diff_val], [0.7, 0.3])

    if expected_severity <= 0.65:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "sentence_over_complexity"

        # Confidence = min confidence of supporting sentence_complexity issues
        supporting = [i for i in issues if i.type == "sentence_complexity"]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Evidence includes mean_sentence_complexity_severity
        assert abs(result.evidence["mean_sentence_complexity_severity"] - sev_comp) < 1e-9

        # Evidence includes difficulty_score
        assert abs(result.evidence["difficulty_score"] - diff_val) < 1e-9

        # None score substitution
        if scores.get("difficulty") is None:
            assert result.evidence.get("difficulty_missing") is True
        else:
            assert "difficulty_missing" not in result.evidence


# ---------------------------------------------------------------------------
# Property 2e: _diagnose_structural_inconsistency
# Validates: Requirements 8.1, 8.2, 8.3, 8.4, 21.3
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_structural_inconsistency_severity_and_threshold(issues, scores):
    """Rule 5: weighted mean of structural_inconsistency(0.6) + fluency(0.4), threshold 0.6."""
    result = _diagnose_structural_inconsistency(issues, scores)

    sev_struct = _expected_max_severity(issues, "structural_inconsistency")
    flu_val, flu_ev = _expected_safe_score(scores, "fluency")
    expected_severity = _expected_weighted_mean([sev_struct, flu_val], [0.6, 0.4])

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "structural_inconsistency"

        # Confidence = min confidence of supporting structural_inconsistency issues
        supporting = [i for i in issues if i.type == "structural_inconsistency"]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Evidence includes fluency_score
        assert abs(result.evidence["fluency_score"] - flu_val) < 1e-9

        # None score substitution
        if scores.get("fluency") is None:
            assert result.evidence.get("fluency_missing") is True
        else:
            assert "fluency_missing" not in result.evidence


# ---------------------------------------------------------------------------
# Property 2f: _diagnose_low_morphological_richness
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 21.4
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_low_morphological_richness_severity_and_threshold(issues, scores):
    """Rule 6: weighted mean of low_morphological_diversity(0.7) + complexity(0.3), threshold 0.6."""
    result = _diagnose_low_morphological_richness(issues, scores)

    sev_morph = _expected_max_severity(issues, "low_morphological_diversity")
    comp_val, comp_ev = _expected_safe_score(scores, "complexity")
    expected_severity = _expected_weighted_mean([sev_morph, comp_val], [0.7, 0.3])

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "low_morphological_richness"

        # Confidence = min confidence of supporting low_morphological_diversity issues
        supporting = [i for i in issues if i.type == "low_morphological_diversity"]
        expected_confidence = min((i.confidence for i in supporting), default=0.0)
        assert abs(result.confidence - expected_confidence) < 1e-9

        # Evidence includes complexity_score
        assert abs(result.evidence["complexity_score"] - comp_val) < 1e-9

        # None score substitution
        if scores.get("complexity") is None:
            assert result.evidence.get("complexity_missing") is True
        else:
            assert "complexity_missing" not in result.evidence


# ---------------------------------------------------------------------------
# Property 2g: _diagnose_fragmented_writing
# Validates: Requirements 10.1, 10.2, 10.3, 10.4
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_fragmented_writing_severity_and_threshold(issues, scores):
    """Rule 7: direct max severity of fragmentation, threshold 0.6."""
    result = _diagnose_fragmented_writing(issues, scores)

    matching = [i for i in issues if i.type == "fragmentation"]
    expected_severity = max((i.severity for i in matching), default=0.0)

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "fragmented_writing"

        # Confidence = confidence of the highest-severity fragmentation issue
        best = max(matching, key=lambda i: i.severity)
        assert abs(result.confidence - best.confidence) < 1e-9

        # Supporting issues
        expected_types = list(dict.fromkeys(i.type for i in matching))
        assert result.supporting_issues == expected_types

        # Supporting spans
        expected_spans = [i.span for i in matching]
        assert result.supporting_spans == expected_spans

        # Evidence
        assert abs(result.evidence["max_fragmentation_severity"] - expected_severity) < 1e-9


# ---------------------------------------------------------------------------
# Property 2h: _diagnose_punctuation_deficiency
# Validates: Requirements 11.1, 11.2, 11.3, 11.4
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnose_punctuation_deficiency_severity_and_threshold(issues, scores):
    """Rule 8: direct max severity of punctuation_issues, threshold 0.6."""
    result = _diagnose_punctuation_deficiency(issues, scores)

    matching = [i for i in issues if i.type == "punctuation_issues"]
    expected_severity = max((i.severity for i in matching), default=0.0)

    if expected_severity <= 0.6:
        assert result is None
    else:
        assert result is not None
        assert abs(result.severity - expected_severity) < 1e-9
        assert result.type == "punctuation_deficiency"

        # Confidence = confidence of the highest-severity punctuation_issues issue
        best = max(matching, key=lambda i: i.severity)
        assert abs(result.confidence - best.confidence) < 1e-9

        # Supporting issues
        expected_types = list(dict.fromkeys(i.type for i in matching))
        assert result.supporting_issues == expected_types

        # Supporting spans
        expected_spans = [i.span for i in matching]
        assert result.supporting_spans == expected_spans

        # Evidence
        assert abs(result.evidence["max_punctuation_issues_severity"] - expected_severity) < 1e-9


# ---------------------------------------------------------------------------
# Property 2i: None score substitution records "{score_name}_missing" in evidence
# Validates: Requirements 21.1, 21.2, 21.3, 21.4, 21.5, 21.6
# ---------------------------------------------------------------------------


@given(issues=issues_strategy)
@settings(max_examples=100)
def test_none_score_substitution_records_missing_in_evidence(issues):
    """When a score is None, the diagnosis rule substitutes 0.0 and records missing."""
    all_none_scores = {
        "difficulty": None,
        "style": None,
        "fluency": None,
        "cohesion": None,
        "complexity": None,
    }

    # Rules that use scores: pronoun_overuse(cohesion), sentence_over_complexity(difficulty),
    # structural_inconsistency(fluency), low_morphological_richness(complexity)
    score_rules = [
        (_diagnose_pronoun_overuse, "cohesion"),
        (_diagnose_sentence_over_complexity, "difficulty"),
        (_diagnose_structural_inconsistency, "fluency"),
        (_diagnose_low_morphological_richness, "complexity"),
    ]

    for rule_fn, score_key in score_rules:
        result = rule_fn(issues, all_none_scores)
        if result is not None:
            # When score was None, evidence must record "{score_key}_missing": True
            assert result.evidence.get(f"{score_key}_missing") is True
            # The substituted value should be 0.0 in the evidence
            score_evidence_key = f"{score_key}_score"
            assert abs(result.evidence[score_evidence_key] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Feature: diagnosis-interventions, Property 3: Diagnosis aggregation ordering
# ---------------------------------------------------------------------------

"""
Property-based tests for run_diagnoses aggregation in analysis/diagnosis_engine.py.

Tests that run_diagnoses returns diagnoses sorted by severity descending,
only includes diagnoses exceeding their activation threshold, and returns
an empty list for empty issues.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**
"""

from analysis.diagnosis_engine import run_diagnoses

# ---------------------------------------------------------------------------
# Thresholds per diagnosis type (from design doc)
# ---------------------------------------------------------------------------

_THRESHOLDS = {
    "low_lexical_diversity": 0.6,
    "pronoun_overuse": 0.6,
    "low_cohesion": 0.6,
    "sentence_over_complexity": 0.65,
    "structural_inconsistency": 0.6,
    "low_morphological_richness": 0.6,
    "fragmented_writing": 0.6,
    "punctuation_deficiency": 0.6,
}

# ---------------------------------------------------------------------------
# Property 3a: run_diagnoses returns diagnoses sorted by severity descending
# Validates: Requirements 12.3, 12.4
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_run_diagnoses_sorted_by_severity_descending(issues, scores):
    """run_diagnoses returns a list of Diagnosis objects sorted by severity descending."""
    result = run_diagnoses(issues, scores)

    severities = [d.severity for d in result]
    for i in range(len(severities) - 1):
        assert severities[i] >= severities[i + 1], (
            f"Diagnoses not sorted descending: severity[{i}]={severities[i]} "
            f"< severity[{i+1}]={severities[i+1]}"
        )


# ---------------------------------------------------------------------------
# Property 3b: all returned diagnoses have severity above their threshold
# Validates: Requirements 12.1, 12.2
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_run_diagnoses_all_above_threshold(issues, scores):
    """All returned diagnoses have severity strictly above their activation threshold."""
    result = run_diagnoses(issues, scores)

    for d in result:
        threshold = _THRESHOLDS[d.type]
        assert d.severity > threshold, (
            f"Diagnosis '{d.type}' has severity {d.severity} "
            f"which does not exceed threshold {threshold}"
        )


# ---------------------------------------------------------------------------
# Property 3c: empty issues list returns empty result
# Validates: Requirements 12.5
# ---------------------------------------------------------------------------


@given(scores=scores_strategy)
@settings(max_examples=100)
def test_run_diagnoses_empty_issues_returns_empty(scores):
    """run_diagnoses with an empty issues list returns an empty list."""
    result = run_diagnoses([], scores)
    assert result == [], f"Expected empty list for empty issues, got {result}"
