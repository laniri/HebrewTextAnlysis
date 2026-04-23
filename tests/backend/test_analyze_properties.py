# Feature: hebrew-writing-coach, Property 1: Score range invariant
# Feature: hebrew-writing-coach, Property 2: Diagnosis structure completeness
# Feature: hebrew-writing-coach, Property 3: Intervention referential integrity
# Feature: hebrew-writing-coach, Property 4: Sentence annotation consistency
# Feature: hebrew-writing-coach, Property 5: Cohesion gap adjacency
"""Property-based tests for the POST /api/analyze endpoint (Properties 1–5).

All tests mock ``model_service.analyze()`` to return Hypothesis-generated
structured data, then verify properties against the API response.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2, 11.2
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from httpx import ASGITransport, AsyncClient
import pytest

from app.services.localization import DIAGNOSIS_MAP, INTERVENTION_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEBREW_ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת "
ALL_DIAGNOSIS_TYPES = list(DIAGNOSIS_MAP.keys())
ALL_INTERVENTION_TYPES = list(INTERVENTION_MAP.keys())
SCORE_KEYS = ["difficulty", "style", "fluency", "cohesion", "complexity"]

# Mapping from diagnosis type to the intervention types that target it
DIAGNOSIS_TO_INTERVENTIONS = {
    "low_cohesion": "cohesion_improvement",
    "sentence_over_complexity": "sentence_simplification",
    "low_lexical_diversity": "vocabulary_expansion",
    "pronoun_overuse": "pronoun_clarification",
}


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating mock model outputs
# ---------------------------------------------------------------------------

def _classify_highlight(severity: float) -> str:
    """Mirror the classification logic from model_service."""
    if severity > 0.7:
        return "red"
    if severity >= 0.4:
        return "yellow"
    return "none"


def scores_strategy():
    """Generate a dict of 5 scores each in [0.0, 1.0]."""
    return st.fixed_dictionaries({
        key: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for key in SCORE_KEYS
    })


def diagnoses_dict_strategy():
    """Generate a dict mapping diagnosis types to severity floats.

    Returns all 8 types with random severities so the endpoint can
    filter by threshold.
    """
    return st.fixed_dictionaries({
        dtype: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for dtype in ALL_DIAGNOSIS_TYPES
    })


def sentence_strategy(index: int, char_start: int, min_len: int = 5):
    """Generate a single sentence annotation dict."""
    return st.integers(min_value=min_len, max_value=50).flatmap(
        lambda length: st.just({
            "index": index,
            "text": "א" * length,
            "char_start": char_start,
            "char_end": char_start + length,
        })
    ).map(lambda s: {
        **s,
        "complexity": 0.0,  # will be overridden
        "highlight": "none",  # will be overridden
    })


@st.composite
def sentences_strategy(draw, min_count=1, max_count=5):
    """Generate a list of sequential, non-overlapping sentence annotations."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    sentences = []
    char_pos = 0
    for i in range(count):
        length = draw(st.integers(min_value=3, max_value=30))
        complexity = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        highlight = _classify_highlight(complexity)
        sentences.append({
            "index": i,
            "text": "א" * length,
            "char_start": char_pos,
            "char_end": char_pos + length,
            "complexity": complexity,
            "highlight": highlight,
        })
        # Add a small gap between sentences (space)
        char_pos = char_pos + length + 1
    return sentences


@st.composite
def cohesion_gaps_strategy(draw, sentences):
    """Generate cohesion gaps for adjacent sentence pairs."""
    if len(sentences) < 2:
        return []
    gaps = []
    num_gaps = draw(st.integers(min_value=0, max_value=len(sentences) - 1))
    used_pairs = set()
    for _ in range(num_gaps):
        i = draw(st.integers(min_value=0, max_value=len(sentences) - 2))
        if i in used_pairs:
            continue
        used_pairs.add(i)
        severity = draw(st.floats(min_value=0.31, max_value=1.0, allow_nan=False, allow_infinity=False))
        gaps.append({
            "pair": (i, i + 1),
            "severity": severity,
            "char_start": sentences[i]["char_end"],
            "char_end": sentences[i + 1]["char_start"],
        })
    return gaps


@st.composite
def interventions_strategy(draw, active_diagnosis_types):
    """Generate interventions that target active diagnoses."""
    interventions = []
    for dtype in active_diagnosis_types:
        if dtype in DIAGNOSIS_TO_INTERVENTIONS:
            itype = DIAGNOSIS_TO_INTERVENTIONS[dtype]
            interventions.append({
                "type": itype,
                "priority": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
                "target_diagnosis": dtype,
            })
    return interventions


@st.composite
def full_analyze_mock(draw):
    """Generate a complete mock return value for model_service.analyze()."""
    scores = draw(scores_strategy())
    diagnoses = draw(diagnoses_dict_strategy())
    sents = draw(sentences_strategy(min_count=1, max_count=5))
    gaps = draw(cohesion_gaps_strategy(sents))

    # Determine active diagnoses (severity > 0.3 threshold)
    active_types = [dtype for dtype, sev in diagnoses.items() if sev > 0.3]
    interventions = draw(interventions_strategy(active_types))

    return {
        "scores": scores,
        "diagnoses": diagnoses,
        "interventions": interventions,
        "sentences": sents,
        "cohesion_gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Helper: call the analyze endpoint with mocked model_service
# ---------------------------------------------------------------------------

def call_analyze(mock_data: dict, text: str = "שלום עולם") -> dict:
    """Call POST /api/analyze with mocked model_service.analyze()."""
    from fastapi.testclient import TestClient
    from app.main import app

    with patch("app.api.analyze.model_service") as mock_svc:
        mock_svc.analyze.return_value = mock_data
        mock_svc.is_loaded = True
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/analyze", json={"text": text})
    return response


# ---------------------------------------------------------------------------
# Property 1 – Score range invariant
# ---------------------------------------------------------------------------

# **Validates: Requirements 1.1**
@given(
    text=st.text(alphabet=HEBREW_ALPHABET, min_size=1, max_size=500),
    mock_data=full_analyze_mock(),
)
@settings(max_examples=100, deadline=None)
def test_score_range_invariant(text: str, mock_data: dict) -> None:
    """For any non-empty Hebrew text, the analyze response contains
    exactly 5 scores each in [0.0, 1.0]."""
    assume(text.strip())  # skip whitespace-only

    response = call_analyze(mock_data, text)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    scores = data["scores"]

    # Exactly 5 score keys
    assert set(scores.keys()) == set(SCORE_KEYS), (
        f"Expected keys {SCORE_KEYS}, got {list(scores.keys())}"
    )

    # Each score in [0.0, 1.0]
    for key in SCORE_KEYS:
        val = scores[key]
        assert isinstance(val, (int, float)), f"Score '{key}' is not numeric: {val}"
        assert 0.0 <= val <= 1.0, f"Score '{key}' = {val} is out of range [0, 1]"


# ---------------------------------------------------------------------------
# Property 2 – Diagnosis structure completeness
# ---------------------------------------------------------------------------

# **Validates: Requirements 1.2, 11.2**
@given(mock_data=full_analyze_mock())
@settings(max_examples=100, deadline=None)
def test_diagnosis_structure_completeness(mock_data: dict) -> None:
    """Every diagnosis in the response has a valid type (one of 8),
    severity in [0.0, 1.0], and non-empty Hebrew strings for label_he,
    explanation_he, actions_he, tip_he."""
    response = call_analyze(mock_data)
    assert response.status_code == 200

    data = response.json()
    for diag in data["diagnoses"]:
        # Valid type
        assert diag["type"] in ALL_DIAGNOSIS_TYPES, (
            f"Unknown diagnosis type: {diag['type']}"
        )
        # Severity in range
        assert 0.0 <= diag["severity"] <= 1.0, (
            f"Severity {diag['severity']} out of range for {diag['type']}"
        )
        # Non-empty Hebrew strings
        assert diag["label_he"], f"label_he is empty for {diag['type']}"
        assert diag["explanation_he"], f"explanation_he is empty for {diag['type']}"
        assert len(diag["actions_he"]) > 0, f"actions_he is empty for {diag['type']}"
        assert all(a for a in diag["actions_he"]), (
            f"actions_he contains empty string for {diag['type']}"
        )
        assert diag["tip_he"], f"tip_he is empty for {diag['type']}"


# ---------------------------------------------------------------------------
# Property 3 – Intervention referential integrity
# ---------------------------------------------------------------------------

# **Validates: Requirements 1.3**
@given(mock_data=full_analyze_mock())
@settings(max_examples=100, deadline=None)
def test_intervention_referential_integrity(mock_data: dict) -> None:
    """Every intervention's target_diagnosis matches a diagnosis type in
    the same response, with non-empty actions_he and exercises_he."""
    response = call_analyze(mock_data)
    assert response.status_code == 200

    data = response.json()
    diagnosis_types_in_response = {d["type"] for d in data["diagnoses"]}

    for interv in data["interventions"]:
        # target_diagnosis must reference an active diagnosis
        assert interv["target_diagnosis"] in diagnosis_types_in_response, (
            f"Intervention target '{interv['target_diagnosis']}' not in "
            f"diagnoses: {diagnosis_types_in_response}"
        )
        # Non-empty actions and exercises
        assert len(interv["actions_he"]) > 0, (
            f"actions_he is empty for intervention type '{interv['type']}'"
        )
        assert len(interv["exercises_he"]) > 0, (
            f"exercises_he is empty for intervention type '{interv['type']}'"
        )


# ---------------------------------------------------------------------------
# Property 4 – Sentence annotation consistency
# ---------------------------------------------------------------------------

# **Validates: Requirements 1.4, 6.1, 6.2**
@given(mock_data=full_analyze_mock())
@settings(max_examples=100, deadline=None)
def test_sentence_annotation_consistency(mock_data: dict) -> None:
    """Sentence annotations have sequential indices from 0,
    non-overlapping char ranges (char_start < char_end), and correct
    highlight classification (red > 0.7, yellow 0.4–0.7, none < 0.4)."""
    response = call_analyze(mock_data)
    assert response.status_code == 200

    data = response.json()
    sentences = data["sentences"]

    for i, sent in enumerate(sentences):
        # Sequential indices from 0
        assert sent["index"] == i, (
            f"Expected index {i}, got {sent['index']}"
        )
        # Non-overlapping: char_start < char_end
        assert sent["char_start"] < sent["char_end"], (
            f"Sentence {i}: char_start ({sent['char_start']}) >= "
            f"char_end ({sent['char_end']})"
        )
        # Correct highlight classification
        complexity = sent["complexity"]
        expected_highlight = _classify_highlight(complexity)
        assert sent["highlight"] == expected_highlight, (
            f"Sentence {i}: complexity={complexity}, expected "
            f"highlight='{expected_highlight}', got '{sent['highlight']}'"
        )

    # Verify non-overlapping between consecutive sentences
    for i in range(1, len(sentences)):
        assert sentences[i]["char_start"] >= sentences[i - 1]["char_end"], (
            f"Sentences {i-1} and {i} overlap: "
            f"prev_end={sentences[i-1]['char_end']}, "
            f"curr_start={sentences[i]['char_start']}"
        )


# ---------------------------------------------------------------------------
# Property 5 – Cohesion gap adjacency
# ---------------------------------------------------------------------------

# **Validates: Requirements 1.5**
@given(mock_data=full_analyze_mock())
@settings(max_examples=100, deadline=None)
def test_cohesion_gap_adjacency(mock_data: dict) -> None:
    """Every cohesion gap has pair [i, i+1] for valid sentence index i,
    and severity exceeds the configured threshold."""
    response = call_analyze(mock_data)
    assert response.status_code == 200

    data = response.json()
    num_sentences = len(data["sentences"])
    threshold = 0.3  # default SEVERITY_THRESHOLD from config

    for gap in data["cohesion_gaps"]:
        pair = gap["pair"]
        # Pair must be [i, i+1]
        assert len(pair) == 2, f"Pair must have exactly 2 elements: {pair}"
        assert pair[1] == pair[0] + 1, (
            f"Pair must be adjacent [i, i+1], got {pair}"
        )
        # Valid sentence index
        assert 0 <= pair[0] < num_sentences, (
            f"Pair[0]={pair[0]} out of range [0, {num_sentences})"
        )
        assert 0 <= pair[1] < num_sentences, (
            f"Pair[1]={pair[1]} out of range [0, {num_sentences})"
        )
        # Severity exceeds threshold
        assert gap["severity"] >= threshold, (
            f"Gap severity {gap['severity']} below threshold {threshold}"
        )
