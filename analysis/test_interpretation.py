# Feature: diagnosis-interventions, Property 6: Integration output structure

"""Property-based tests for the integration entry point in analysis/interpretation.py.

Tests that run_interpretation returns a dict with the correct keys, diagnoses
sorted by severity descending, interventions sorted by priority descending,
every intervention's target_diagnosis references a diagnosis in the list, and
empty issues produce empty output.

**Validates: Requirements 19.1, 19.2, 19.3, 19.4**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.interpretation import run_interpretation
from analysis.issue_models import Issue

# ---------------------------------------------------------------------------
# Strategies (reused pattern from test_diagnosis_engine.py)
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
# Property 6a: Output has keys "diagnoses" and "interventions"
# **Validates: Requirements 19.1, 19.2, 19.3**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_output_has_diagnoses_and_interventions_keys(issues, scores):
    """run_interpretation returns a dict with exactly the keys 'diagnoses' and 'interventions'."""
    result = run_interpretation(issues, scores)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "diagnoses" in result, "Output missing 'diagnoses' key"
    assert "interventions" in result, "Output missing 'interventions' key"
    assert set(result.keys()) == {"diagnoses", "interventions"}, (
        f"Unexpected keys in output: {set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Property 6b: Diagnoses sorted by severity descending
# **Validates: Requirements 19.1, 19.3**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_diagnoses_sorted_by_severity_descending(issues, scores):
    """Diagnoses in the output are sorted by severity in descending order."""
    result = run_interpretation(issues, scores)
    diagnoses = result["diagnoses"]

    severities = [d.severity for d in diagnoses]
    for i in range(len(severities) - 1):
        assert severities[i] >= severities[i + 1], (
            f"Diagnoses not sorted descending: severity[{i}]={severities[i]} "
            f"< severity[{i+1}]={severities[i+1]}"
        )


# ---------------------------------------------------------------------------
# Property 6c: Interventions sorted by priority descending
# **Validates: Requirements 19.2, 19.3**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_interventions_sorted_by_priority_descending(issues, scores):
    """Interventions in the output are sorted by priority in descending order."""
    result = run_interpretation(issues, scores)
    interventions = result["interventions"]

    priorities = [iv.priority for iv in interventions]
    for i in range(len(priorities) - 1):
        assert priorities[i] >= priorities[i + 1], (
            f"Interventions not sorted descending: priority[{i}]={priorities[i]} "
            f"< priority[{i+1}]={priorities[i+1]}"
        )


# ---------------------------------------------------------------------------
# Property 6d: Every intervention's target_diagnosis references a diagnosis
# **Validates: Requirements 19.2, 19.3**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_every_intervention_references_a_diagnosis(issues, scores):
    """Every intervention's target_diagnosis must reference a diagnosis type
    present in the diagnoses list."""
    result = run_interpretation(issues, scores)
    diagnoses = result["diagnoses"]
    interventions = result["interventions"]

    diagnosis_types = {d.type for d in diagnoses}
    for iv in interventions:
        assert iv.target_diagnosis in diagnosis_types, (
            f"Intervention target_diagnosis '{iv.target_diagnosis}' not found "
            f"in diagnosis types {diagnosis_types}"
        )


# ---------------------------------------------------------------------------
# Property 6e: Empty issues produce empty diagnoses and interventions
# **Validates: Requirement 19.4**
# ---------------------------------------------------------------------------


@given(scores=scores_strategy)
@settings(max_examples=100)
def test_empty_issues_produces_empty_output(scores):
    """An empty issues list produces empty diagnoses and empty interventions."""
    result = run_interpretation([], scores)

    assert result["diagnoses"] == [], (
        f"Expected empty diagnoses for empty issues, got {result['diagnoses']}"
    )
    assert result["interventions"] == [], (
        f"Expected empty interventions for empty issues, got {result['interventions']}"
    )


# ---------------------------------------------------------------------------
# Feature: diagnosis-interventions, Property 7: Serialization round-trip
# ---------------------------------------------------------------------------

import json

from analysis.serialization import serialize_interpretation


# ---------------------------------------------------------------------------
# Property 7a: Serialize then deserialize produces equivalent structure
# **Validates: Requirements 20.1, 20.2, 20.3, 20.7**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_serialization_round_trip(issues, scores):
    """Serializing then deserializing produces an equivalent structure."""
    output = run_interpretation(issues, scores)
    json_str = serialize_interpretation(output)
    parsed = json.loads(json_str)

    # Top-level keys match
    assert set(parsed.keys()) == {"diagnoses", "interventions"}

    # Same number of diagnoses and interventions
    assert len(parsed["diagnoses"]) == len(output["diagnoses"])
    assert len(parsed["interventions"]) == len(output["interventions"])

    # Verify each diagnosis field round-trips correctly
    for orig, deser in zip(output["diagnoses"], parsed["diagnoses"]):
        assert deser["type"] == orig.type
        assert deser["confidence"] == float(orig.confidence)
        assert deser["severity"] == float(orig.severity)
        assert deser["supporting_issues"] == list(orig.supporting_issues)
        assert deser["supporting_spans"] == [list(s) for s in orig.supporting_spans]
        assert set(deser["evidence"].keys()) == set(orig.evidence.keys())
        for k in orig.evidence:
            assert deser["evidence"][k] == float(orig.evidence[k])

    # Verify each intervention field round-trips correctly
    for orig, deser in zip(output["interventions"], parsed["interventions"]):
        assert deser["type"] == orig.type
        assert deser["priority"] == float(orig.priority)
        assert deser["target_diagnosis"] == orig.target_diagnosis
        assert deser["actions"] == list(orig.actions)
        assert deser["exercises"] == list(orig.exercises)
        assert deser["focus_features"] == list(orig.focus_features)


# ---------------------------------------------------------------------------
# Property 7b: Tuple spans become lists of integers
# **Validates: Requirements 20.4**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_serialization_spans_become_lists(issues, scores):
    """Tuple spans in supporting_spans become JSON arrays of integers."""
    output = run_interpretation(issues, scores)
    json_str = serialize_interpretation(output)
    parsed = json.loads(json_str)

    for diag in parsed["diagnoses"]:
        assert isinstance(diag["supporting_spans"], list)
        for span in diag["supporting_spans"]:
            assert isinstance(span, list), f"Span should be a list, got {type(span)}"
            for elem in span:
                assert isinstance(elem, int), (
                    f"Span element should be int, got {type(elem)}: {elem}"
                )


# ---------------------------------------------------------------------------
# Property 7c: Evidence values are floats
# **Validates: Requirements 20.5**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_serialization_evidence_values_are_floats(issues, scores):
    """Evidence dict values in serialized diagnoses are floats."""
    output = run_interpretation(issues, scores)
    json_str = serialize_interpretation(output)
    parsed = json.loads(json_str)

    for diag in parsed["diagnoses"]:
        assert isinstance(diag["evidence"], dict)
        for key, value in diag["evidence"].items():
            assert isinstance(key, str), f"Evidence key should be str, got {type(key)}"
            assert isinstance(value, (int, float)), (
                f"Evidence value for '{key}' should be float, got {type(value)}: {value}"
            )


# ---------------------------------------------------------------------------
# Property 7d: Hebrew characters preserved
# **Validates: Requirements 20.6**
# ---------------------------------------------------------------------------


@given(issues=issues_strategy, scores=scores_strategy)
@settings(max_examples=100)
def test_serialization_preserves_hebrew(issues, scores):
    """Hebrew characters in evidence keys/actions are preserved via ensure_ascii=False."""
    # Build output with a Hebrew string injected into evidence
    output = run_interpretation(issues, scores)

    # Inject a Hebrew key into the first diagnosis's evidence (if any)
    hebrew_key = "ניתוח_מורפולוגי"
    hebrew_value_str = "תרגיל_כתיבה"
    if output["diagnoses"]:
        output["diagnoses"][0].evidence[hebrew_key] = 0.42

    # Inject a Hebrew action into the first intervention (if any)
    if output["interventions"]:
        output["interventions"][0].actions.append(hebrew_value_str)

    json_str = serialize_interpretation(output)

    # Hebrew characters must appear literally, not as \uXXXX escapes
    if output["diagnoses"]:
        assert hebrew_key in json_str, (
            f"Hebrew key '{hebrew_key}' not found literally in JSON output"
        )
    if output["interventions"]:
        assert hebrew_value_str in json_str, (
            f"Hebrew string '{hebrew_value_str}' not found literally in JSON output"
        )

    # Verify round-trip preserves the Hebrew
    parsed = json.loads(json_str)
    if output["diagnoses"]:
        assert hebrew_key in parsed["diagnoses"][0]["evidence"]
        assert parsed["diagnoses"][0]["evidence"][hebrew_key] == 0.42
    if output["interventions"]:
        assert hebrew_value_str in parsed["interventions"][0]["actions"]
