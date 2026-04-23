# Feature: diagnosis-interventions, Property 4: Intervention mapping correctness

"""Property-based tests for intervention mapping in analysis/intervention_mapper.py.

Tests that each mapped diagnosis produces exactly one Intervention with the
correct type, priority = severity, and target_diagnosis = diagnosis type.
Also tests that unmapped diagnosis types produce no Intervention.

**Validates: Requirements 13.1–13.3, 14.1–14.3, 15.1–15.3, 16.1–16.3, 17.1–17.3**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.diagnosis_models import Diagnosis, Intervention
from analysis.intervention_mapper import INTERVENTION_MAP, map_interventions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_DIAGNOSIS_TYPES = [
    "low_lexical_diversity",
    "pronoun_overuse",
    "low_cohesion",
    "sentence_over_complexity",
    "structural_inconsistency",
    "low_morphological_richness",
    "fragmented_writing",
    "punctuation_deficiency",
]

EXPECTED_MAPPING = {
    "low_lexical_diversity": "vocabulary_expansion",
    "low_morphological_richness": "vocabulary_expansion",
    "pronoun_overuse": "pronoun_clarification",
    "sentence_over_complexity": "sentence_simplification",
    "structural_inconsistency": "sentence_simplification",
    "fragmented_writing": "sentence_simplification",
    "low_cohesion": "cohesion_improvement",
    "punctuation_deficiency": "cohesion_improvement",
}

# Strings that are guaranteed NOT to be in the intervention map
UNKNOWN_DIAGNOSIS_TYPES = [
    "unknown_type",
    "nonexistent_diagnosis",
    "agreement_errors",
    "morphological_ambiguity",
    "sentence_complexity",
    "rare_word_overuse",
]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

diagnosis_strategy = st.builds(
    Diagnosis,
    type=st.sampled_from(KNOWN_DIAGNOSIS_TYPES),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    supporting_issues=st.lists(st.text(min_size=1, max_size=30), max_size=5),
    supporting_spans=st.lists(
        st.one_of(
            st.tuples(st.integers(min_value=0, max_value=100)),
            st.tuples(
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
            ),
        ),
        max_size=5,
    ),
    evidence=st.dictionaries(
        keys=st.text(min_size=1, max_size=30),
        values=st.floats(allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
)

unknown_diagnosis_strategy = st.builds(
    Diagnosis,
    type=st.sampled_from(UNKNOWN_DIAGNOSIS_TYPES),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    supporting_issues=st.lists(st.text(min_size=1, max_size=30), max_size=5),
    supporting_spans=st.lists(
        st.one_of(
            st.tuples(st.integers(min_value=0, max_value=100)),
            st.tuples(
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
            ),
        ),
        max_size=5,
    ),
    evidence=st.dictionaries(
        keys=st.text(min_size=1, max_size=30),
        values=st.floats(allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
)


# ---------------------------------------------------------------------------
# Property 4a: Each mapped diagnosis produces exactly one Intervention
#              with correct type, priority = severity, target_diagnosis = type
# Validates: Requirements 13.1–13.3, 14.1–14.3, 15.1–15.3, 16.1–16.3, 17.2
# ---------------------------------------------------------------------------


@given(diag=diagnosis_strategy)
@settings(max_examples=100)
def test_mapped_diagnosis_produces_one_correct_intervention(diag):
    """A single known-type diagnosis produces exactly one Intervention with
    the correct intervention type, priority equal to severity, and
    target_diagnosis equal to the diagnosis type."""
    result = map_interventions([diag])

    assert len(result) == 1, (
        f"Expected exactly 1 intervention for diagnosis type '{diag.type}', "
        f"got {len(result)}"
    )

    intervention = result[0]

    # Correct intervention type per the mapping table
    expected_type = EXPECTED_MAPPING[diag.type]
    assert intervention.type == expected_type, (
        f"Diagnosis '{diag.type}' should map to '{expected_type}', "
        f"got '{intervention.type}'"
    )

    # Priority equals the diagnosis severity
    assert intervention.priority == diag.severity, (
        f"Intervention priority ({intervention.priority}) should equal "
        f"diagnosis severity ({diag.severity})"
    )

    # target_diagnosis equals the diagnosis type
    assert intervention.target_diagnosis == diag.type, (
        f"Intervention target_diagnosis ('{intervention.target_diagnosis}') "
        f"should equal diagnosis type ('{diag.type}')"
    )


# ---------------------------------------------------------------------------
# Property 4b: Unmapped diagnosis types produce no Intervention
# Validates: Requirement 17.3
# ---------------------------------------------------------------------------


@given(diag=unknown_diagnosis_strategy)
@settings(max_examples=100)
def test_unmapped_diagnosis_produces_no_intervention(diag):
    """A diagnosis with a type not in INTERVENTION_MAP produces no intervention."""
    result = map_interventions([diag])

    assert len(result) == 0, (
        f"Expected no interventions for unmapped diagnosis type '{diag.type}', "
        f"got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Property 4c: INTERVENTION_MAP contains exactly the 8 known diagnosis types
# Validates: Requirement 17.1
# ---------------------------------------------------------------------------


def test_intervention_map_has_all_8_entries():
    """INTERVENTION_MAP contains an entry for each of the 8 diagnosis types."""
    assert set(INTERVENTION_MAP.keys()) == set(KNOWN_DIAGNOSIS_TYPES), (
        f"INTERVENTION_MAP keys {set(INTERVENTION_MAP.keys())} do not match "
        f"expected {set(KNOWN_DIAGNOSIS_TYPES)}"
    )
    assert len(INTERVENTION_MAP) == 8


# ---------------------------------------------------------------------------
# Feature: diagnosis-interventions, Property 5: Intervention aggregation ordering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Property 5a: map_interventions returns interventions sorted by priority
#              descending
# **Validates: Requirements 18.1, 18.2, 18.3**
# ---------------------------------------------------------------------------


@given(
    diagnoses=st.lists(
        st.one_of(diagnosis_strategy, unknown_diagnosis_strategy),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_map_interventions_sorted_by_priority_descending(diagnoses):
    """Given a list of diagnoses (mix of known and unknown types), the
    returned interventions are sorted by priority in descending order."""
    result = map_interventions(diagnoses)

    priorities = [iv.priority for iv in result]
    assert priorities == sorted(priorities, reverse=True), (
        f"Interventions not sorted by priority descending: {priorities}"
    )


# ---------------------------------------------------------------------------
# Property 5b: Exactly one intervention per mapped diagnosis, zero for
#              unmapped
# **Validates: Requirements 18.1, 18.2, 18.3**
# ---------------------------------------------------------------------------


@given(
    known=st.lists(diagnosis_strategy, max_size=8),
    unknown=st.lists(unknown_diagnosis_strategy, max_size=5),
)
@settings(max_examples=100)
def test_map_interventions_count_matches_mapped_diagnoses(known, unknown):
    """The number of interventions equals the number of diagnoses whose type
    is present in INTERVENTION_MAP.  Each mapped diagnosis produces exactly
    one intervention; unmapped diagnoses produce zero."""
    all_diagnoses = known + unknown
    result = map_interventions(all_diagnoses)

    expected_count = sum(
        1 for d in all_diagnoses if d.type in INTERVENTION_MAP
    )
    assert len(result) == expected_count, (
        f"Expected {expected_count} interventions, got {len(result)}"
    )

    # Every intervention's target_diagnosis must reference a mapped diagnosis
    mapped_types = {d.type for d in all_diagnoses if d.type in INTERVENTION_MAP}
    for iv in result:
        assert iv.target_diagnosis in mapped_types, (
            f"Intervention target_diagnosis '{iv.target_diagnosis}' not in "
            f"mapped diagnosis types {mapped_types}"
        )


# ---------------------------------------------------------------------------
# Property 5c: Empty diagnoses list returns empty result
# **Validates: Requirements 18.3**
# ---------------------------------------------------------------------------


def test_map_interventions_empty_diagnoses_returns_empty():
    """An empty diagnoses list produces an empty interventions list."""
    result = map_interventions([])
    assert result == [], f"Expected empty list, got {result}"
