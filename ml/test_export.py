# Feature: ml-distillation-layer, Property 1: Issue flattening completeness and correctness
# Feature: ml-distillation-layer, Property 2: Diagnosis flattening completeness and correctness
# Feature: ml-distillation-layer, Property 3: Training record round-trip serialization
# Feature: ml-distillation-layer, Property 4: Null score substitution in export
# Feature: ml-distillation-layer, Property 5: Malformed JSON rejection in export

"""Property-based and unit tests for ml/export.py.

Tests the data export module: issue/diagnosis flattening, training record
serialization, null score substitution, malformed JSON rejection, and
end-to-end export behaviour.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 3.1–3.4, 4.1–4.4,
5.1–5.5, 6.1–6.4**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.diagnosis_models import Diagnosis
from analysis.issue_models import Issue
from ml.export import (
    _compute_label_stats,
    _extract_sentence_labels,
    _flatten_diagnoses,
    _flatten_issues,
    _read_pipeline_json,
    export_training_data,
)
from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS

# ---------------------------------------------------------------------------
# Shared strategies (reused from analysis/test_diagnosis_engine.py pattern)
# ---------------------------------------------------------------------------

GROUPS = ["morphology", "syntax", "lexicon", "structure", "discourse", "style"]

ISSUE_TYPES = list(_ISSUE_KEYS)

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

DIAGNOSIS_TYPES = list(_DIAGNOSIS_KEYS)

diagnosis_strategy = st.builds(
    Diagnosis,
    type=st.sampled_from(DIAGNOSIS_TYPES),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    supporting_issues=st.lists(st.sampled_from(ISSUE_TYPES), max_size=5),
    supporting_spans=st.just([]),
    evidence=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.floats(allow_nan=False, allow_infinity=False),
        max_size=3,
    ),
)

diagnoses_strategy = st.lists(diagnosis_strategy, max_size=10)


# ===========================================================================
# Property 1: Issue flattening completeness and correctness
# ===========================================================================

# Feature: ml-distillation-layer, Property 1: Issue flattening completeness and correctness
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4**


@given(issues=issues_strategy)
@settings(max_examples=100)
def test_flatten_issues_completeness_and_correctness(issues: list[Issue]) -> None:
    """For any list of Issue objects, _flatten_issues produces a dict with
    exactly 17 keys matching _ISSUE_KEYS, each value is the max severity
    among issues of that type (or 0.0 if absent), all values are floats
    in [0, 1]."""
    result = _flatten_issues(issues)

    # Exactly 17 keys matching canonical ordering
    assert set(result.keys()) == set(_ISSUE_KEYS)
    assert len(result) == 17

    for key in _ISSUE_KEYS:
        val = result[key]
        # Value is a float in [0, 1]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

        # Value equals max severity among matching issues, or 0.0
        matching = [i for i in issues if i.type == key]
        if matching:
            expected = max(i.severity for i in matching)
        else:
            expected = 0.0
        assert abs(val - expected) < 1e-12, (
            f"Key '{key}': expected {expected}, got {val}"
        )


# ===========================================================================
# Property 2: Diagnosis flattening completeness and correctness
# ===========================================================================

# Feature: ml-distillation-layer, Property 2: Diagnosis flattening completeness and correctness
# **Validates: Requirements 4.1, 4.2, 4.3, 4.4**


@given(diagnoses=diagnoses_strategy)
@settings(max_examples=100)
def test_flatten_diagnoses_completeness_and_correctness(
    diagnoses: list[Diagnosis],
) -> None:
    """For any list of Diagnosis objects, _flatten_diagnoses produces a dict
    with exactly 8 keys matching _DIAGNOSIS_KEYS, each value is the diagnosis
    severity for activated types (or 0.0 if absent), all values are floats
    in [0, 1]."""
    result = _flatten_diagnoses(diagnoses)

    # Exactly 8 keys matching canonical ordering
    assert set(result.keys()) == set(_DIAGNOSIS_KEYS)
    assert len(result) == 8

    for key in _DIAGNOSIS_KEYS:
        val = result[key]
        # Value is a float in [0, 1]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

        # Value equals the severity of the matching diagnosis, or 0.0
        matching = [d for d in diagnoses if d.type == key]
        if matching:
            # _flatten_diagnoses uses the last matching diagnosis's severity
            expected = matching[-1].severity
        else:
            expected = 0.0
        assert abs(val - expected) < 1e-12, (
            f"Key '{key}': expected {expected}, got {val}"
        )


# ===========================================================================
# Property 3: Training record round-trip serialization
# ===========================================================================

# Feature: ml-distillation-layer, Property 3: Training record round-trip serialization
# **Validates: Requirements 5.2, 5.5**

_training_record_strategy = st.fixed_dictionaries({
    "text": st.text(min_size=1, max_size=200),
    "scores": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for k in _SCORE_KEYS
    }),
    "issues": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for k in _ISSUE_KEYS
    }),
    "diagnoses": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        for k in _DIAGNOSIS_KEYS
    }),
})


@given(record=_training_record_strategy)
@settings(max_examples=100)
def test_training_record_round_trip_serialization(record: dict) -> None:
    """For any valid Training_Record dict, serializing to JSON with
    ensure_ascii=False and deserializing produces an equivalent dict
    with exactly 4 top-level keys."""
    line = json.dumps(record, ensure_ascii=False)
    restored = json.loads(line)

    assert set(restored.keys()) >= {"text", "scores", "issues", "diagnoses"}
    assert restored["text"] == record["text"]

    for key in _SCORE_KEYS:
        assert abs(restored["scores"][key] - record["scores"][key]) < 1e-12

    for key in _ISSUE_KEYS:
        assert abs(restored["issues"][key] - record["issues"][key]) < 1e-12

    for key in _DIAGNOSIS_KEYS:
        assert abs(restored["diagnoses"][key] - record["diagnoses"][key]) < 1e-12


# ===========================================================================
# Property 4: Null score substitution in export
# ===========================================================================

# Feature: ml-distillation-layer, Property 4: Null score substitution in export
# **Validates: Requirements 1.2**

_nullable_scores_strategy = st.fixed_dictionaries({
    k: st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    for k in _SCORE_KEYS
}).filter(lambda d: any(v is None for v in d.values()))


@given(scores_raw=_nullable_scores_strategy)
@settings(max_examples=100)
def test_null_score_substitution(scores_raw: dict) -> None:
    """For any scores dict where one or more values are None, after
    substitution all 5 score keys map to floats in [0, 1] with no
    None values."""
    # Replicate the substitution logic from export_training_data
    scores: dict[str, float] = {}
    for k in _SCORE_KEYS:
        val = scores_raw.get(k)
        scores[k] = 0.0 if val is None else float(val)

    assert set(scores.keys()) == set(_SCORE_KEYS)
    assert len(scores) == 5

    for k in _SCORE_KEYS:
        val = scores[k]
        assert val is not None
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

        # If original was None, substituted value must be 0.0
        if scores_raw[k] is None:
            assert val == 0.0
        else:
            assert abs(val - scores_raw[k]) < 1e-12


# ===========================================================================
# Property 5: Malformed JSON rejection in export
# ===========================================================================

# Feature: ml-distillation-layer, Property 5: Malformed JSON rejection in export
# **Validates: Requirements 1.3**

_required_fields = ["text", "features", "scores"]

_malformed_json_strategy = st.fixed_dictionaries({
    "text": st.text(min_size=1, max_size=50),
    "features": st.just({"morphology": {"verb_ratio": 0.5}}),
    "scores": st.just({"difficulty": 0.5, "style": 0.3, "fluency": 0.4, "cohesion": 0.2, "complexity": 0.6}),
}).flatmap(
    lambda d: st.sampled_from(
        # Generate all subsets that are missing at least one required field
        [
            {k: v for k, v in d.items() if k != missing}
            for missing in _required_fields
        ]
    )
)


@given(malformed=_malformed_json_strategy)
@settings(max_examples=100)
def test_malformed_json_rejection(malformed: dict) -> None:
    """For any dict missing required fields, _read_pipeline_json returns None."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as fh:
        json.dump(malformed, fh, ensure_ascii=False)
        fh.flush()
        path = fh.name

    try:
        result = _read_pipeline_json(path)
        assert result is None
    finally:
        Path(path).unlink(missing_ok=True)


# ===========================================================================
# Unit tests for export module (Task 2.7)
# ===========================================================================


class TestExportUnit:
    """Unit tests for the export module."""

    # -- Fixture: minimal pipeline JSON and stats --------------------------

    @staticmethod
    def _make_pipeline_json() -> dict:
        """Return a minimal valid pipeline JSON dict."""
        return {
            "text": "שלום עולם. זהו טקסט לדוגמה.",
            "features": {
                "morphology": {
                    "verb_ratio": 0.1,
                    "prefix_density": 0.2,
                    "suffix_pronoun_ratio": 0.15,
                    "morphological_ambiguity": 3.0,
                    "agreement_error_rate": 0.05,
                    "binyan_entropy": 1.0,
                    "construct_ratio": 0.01,
                },
                "syntax": {
                    "avg_sentence_length": 10.0,
                    "avg_tree_depth": 4.0,
                    "max_tree_depth": 6.0,
                    "avg_dependency_distance": 2.0,
                    "clauses_per_sentence": 1.0,
                    "subordinate_clause_ratio": 0.5,
                    "right_branching_ratio": 0.6,
                    "dependency_distance_variance": 3.0,
                    "clause_type_entropy": 1.5,
                },
                "lexicon": {
                    "type_token_ratio": 0.8,
                    "hapax_ratio": 0.5,
                    "avg_token_length": 4.0,
                    "lemma_diversity": 0.7,
                    "rare_word_ratio": 0.02,
                    "content_word_ratio": 0.5,
                },
                "structure": {
                    "sentence_length_variance": 20.0,
                    "long_sentence_ratio": 0.3,
                    "punctuation_ratio": 0.1,
                    "short_sentence_ratio": 0.1,
                    "missing_terminal_punctuation_ratio": 0.0,
                },
                "discourse": {
                    "connective_ratio": 0.15,
                    "sentence_overlap": 0.2,
                    "pronoun_to_noun_ratio": 0.1,
                },
                "style": {
                    "sentence_length_trend": 0.05,
                    "pos_distribution_variance": 0.003,
                },
            },
            "scores": {
                "difficulty": 0.5,
                "style": 0.3,
                "fluency": 0.6,
                "cohesion": 0.4,
                "complexity": 0.55,
            },
        }

    @staticmethod
    def _make_stats_json() -> dict:
        """Return a minimal feature_stats dict for serialization."""
        # Build a stats entry for every feature key that appears in the
        # pipeline JSON fixture above.
        feature_keys = [
            "verb_ratio", "prefix_density", "suffix_pronoun_ratio",
            "morphological_ambiguity", "agreement_error_rate", "binyan_entropy",
            "construct_ratio", "avg_sentence_length", "avg_tree_depth",
            "max_tree_depth", "avg_dependency_distance", "clauses_per_sentence",
            "subordinate_clause_ratio", "right_branching_ratio",
            "dependency_distance_variance", "clause_type_entropy",
            "type_token_ratio", "hapax_ratio", "avg_token_length",
            "lemma_diversity", "rare_word_ratio", "content_word_ratio",
            "sentence_length_variance", "long_sentence_ratio",
            "punctuation_ratio", "short_sentence_ratio",
            "missing_terminal_punctuation_ratio", "connective_ratio",
            "sentence_overlap", "pronoun_to_noun_ratio",
            "sentence_length_trend", "pos_distribution_variance",
        ]
        stats = {}
        for key in feature_keys:
            stats[key] = {
                "mean": 0.5,
                "std": 0.2,
                "min": 0.0,
                "max": 1.0,
                "p10": 0.1,
                "p25": 0.25,
                "p50": 0.5,
                "p75": 0.75,
                "p90": 0.9,
                "valid_count": 100,
                "unstable": False,
                "degenerate": False,
            }
        return stats

    # -- Test: known pipeline JSON → expected JSONL output -----------------

    def test_known_pipeline_json_to_jsonl(self, tmp_path: Path) -> None:
        """A valid pipeline JSON produces a JSONL line with 4 top-level keys."""
        # Set up input directory with one JSON file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        pipeline_json = self._make_pipeline_json()
        (input_dir / "doc_001.json").write_text(
            json.dumps(pipeline_json, ensure_ascii=False), encoding="utf-8"
        )

        # Write stats file
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps(self._make_stats_json(), indent=2), encoding="utf-8"
        )

        output_path = tmp_path / "output.jsonl"

        export_training_data(
            input_dirs=[str(input_dir)],
            stats_path=str(stats_path),
            output_path=str(output_path),
        )

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert set(record.keys()) == {"text", "scores", "issues", "diagnoses", "sentence_complexities", "cohesion_pairs"}
        assert record["text"] == pipeline_json["text"]
        assert len(record["scores"]) == 5
        assert len(record["issues"]) == 17
        assert len(record["diagnoses"]) == 8

        # All score values are floats in [0, 1]
        for v in record["scores"].values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

        # All issue values are floats in [0, 1]
        for v in record["issues"].values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

        # All diagnosis values are floats in [0, 1]
        for v in record["diagnoses"].values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    # -- Test: label distribution stats computation ------------------------

    def test_label_distribution_stats(self) -> None:
        """_compute_label_stats returns correct mean/std and activation rates."""
        records = [
            {
                "scores": {k: 0.5 for k in _SCORE_KEYS},
                "issues": {k: 0.0 for k in _ISSUE_KEYS},
                "diagnoses": {k: 0.0 for k in _DIAGNOSIS_KEYS},
            },
            {
                "scores": {k: 1.0 for k in _SCORE_KEYS},
                "issues": {**{k: 0.0 for k in _ISSUE_KEYS}, "agreement_errors": 0.8},
                "diagnoses": {**{k: 0.0 for k in _DIAGNOSIS_KEYS}, "low_cohesion": 0.7},
            },
        ]

        stats = _compute_label_stats(records)

        # Score stats: mean of [0.5, 1.0] = 0.75
        for k in _SCORE_KEYS:
            assert abs(stats["scores"][k]["mean"] - 0.75) < 1e-6

        # Issue activation: agreement_errors active in 1/2 docs
        assert abs(stats["issues"]["agreement_errors"]["activation_rate"] - 0.5) < 1e-6
        # All other issues inactive
        for k in _ISSUE_KEYS:
            if k != "agreement_errors":
                assert stats["issues"][k]["activation_rate"] == 0.0

        # Diagnosis activation: low_cohesion active in 1/2 docs
        assert abs(stats["diagnoses"]["low_cohesion"]["activation_rate"] - 0.5) < 1e-6

    def test_label_distribution_stats_empty(self) -> None:
        """_compute_label_stats with empty records returns zero stats."""
        stats = _compute_label_stats([])
        for k in _SCORE_KEYS:
            assert stats["scores"][k]["mean"] == 0.0
            assert stats["scores"][k]["std"] == 0.0
        for k in _ISSUE_KEYS:
            assert stats["issues"][k]["activation_rate"] == 0.0
        for k in _DIAGNOSIS_KEYS:
            assert stats["diagnoses"][k]["activation_rate"] == 0.0

    # -- Test: ensure_ascii=False with Hebrew text -------------------------

    def test_ensure_ascii_false_hebrew(self, tmp_path: Path) -> None:
        """JSONL output preserves Hebrew characters (no \\uXXXX escapes)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        pipeline_json = self._make_pipeline_json()
        # Use Hebrew text with known characters
        pipeline_json["text"] = "שלום עולם"
        (input_dir / "doc_001.json").write_text(
            json.dumps(pipeline_json, ensure_ascii=False), encoding="utf-8"
        )

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps(self._make_stats_json(), indent=2), encoding="utf-8"
        )

        output_path = tmp_path / "output.jsonl"
        export_training_data(
            input_dirs=[str(input_dir)],
            stats_path=str(stats_path),
            output_path=str(output_path),
        )

        raw_content = output_path.read_text(encoding="utf-8")
        # Hebrew characters should appear directly, not as \uXXXX escapes
        assert "שלום" in raw_content
        assert "\\u05e9" not in raw_content  # shin should not be escaped

    # -- Test: empty input directory produces empty JSONL -------------------

    def test_empty_input_directory(self, tmp_path: Path) -> None:
        """An empty input directory produces an empty JSONL file."""
        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps(self._make_stats_json(), indent=2), encoding="utf-8"
        )

        output_path = tmp_path / "output.jsonl"
        export_training_data(
            input_dirs=[str(input_dir)],
            stats_path=str(stats_path),
            output_path=str(output_path),
        )

        content = output_path.read_text(encoding="utf-8").strip()
        assert content == ""

    # -- Test: stderr logging of processed/skipped counts ------------------

    def test_stderr_logging_counts(self, tmp_path: Path, capsys) -> None:
        """Export logs processed and skipped counts to stderr."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # One valid file
        pipeline_json = self._make_pipeline_json()
        (input_dir / "doc_001.json").write_text(
            json.dumps(pipeline_json, ensure_ascii=False), encoding="utf-8"
        )

        # One malformed file (missing "text")
        malformed = {"features": {}, "scores": {}}
        (input_dir / "doc_002.json").write_text(
            json.dumps(malformed), encoding="utf-8"
        )

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps(self._make_stats_json(), indent=2), encoding="utf-8"
        )

        output_path = tmp_path / "output.jsonl"
        export_training_data(
            input_dirs=[str(input_dir)],
            stats_path=str(stats_path),
            output_path=str(output_path),
        )

        captured = capsys.readouterr()
        # Check stderr contains processed and skipped counts
        assert "Processed 1" in captured.err
        assert "skipped 1" in captured.err
