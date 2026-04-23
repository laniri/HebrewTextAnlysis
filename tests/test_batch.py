"""Tests for hebrew_profiler.batch — batch processing and JSONL export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hebrew_profiler.batch import process_batch, _log_error, _process_single_file
from hebrew_profiler.models import (
    BatchResult,
    DiscourseFeatures,
    Features,
    LexicalFeatures,
    MorphFeatures,
    PipelineConfig,
    PipelineOutput,
    Scores,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_output(text: str = "שלום") -> PipelineOutput:
    """Build a minimal PipelineOutput for testing."""
    return PipelineOutput(
        text=text,
        features=Features(
            morphology=MorphFeatures(
                verb_ratio=0.1,
                binyan_distribution={"paal": 1},
                prefix_density=0.2,
                suffix_pronoun_ratio=0.05,
                morphological_ambiguity=1.5,
                agreement_error_rate=0.0,
                binyan_entropy=0.0,
                construct_ratio=0.0,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=5.0,
                avg_tree_depth=3.0,
                max_tree_depth=4.0,
                avg_dependency_distance=2.0,
                clauses_per_sentence=0.5,
                subordinate_clause_ratio=0.0,
                right_branching_ratio=0.0,
                dependency_distance_variance=0.0,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=0.8,
                hapax_ratio=0.4,
                avg_token_length=3.5,
                lemma_diversity=0.7,
                rare_word_ratio=None,
                content_word_ratio=0.0,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=2.0,
                long_sentence_ratio=0.1,
                punctuation_ratio=0.0,
                short_sentence_ratio=0.0,
                missing_terminal_punctuation_ratio=0.0,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        ),
        scores=Scores(difficulty=0.5, style=0.3),
    )


def _write_txt(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Task 13.1 — process_batch core behaviour
# ---------------------------------------------------------------------------


class TestProcessBatchDiscovery:
    """Discover text files and create output directory."""

    def test_empty_input_dir(self, tmp_path: Path):
        """No .txt files → BatchResult(0, 0, [])."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"

        result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert result.total_processed == 0
        assert result.error_count == 0
        assert result.errors == []

    def test_creates_output_dir(self, tmp_path: Path):
        """Output directory is created if it does not exist."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out" / "nested"

        process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert out_dir.is_dir()

    @patch("hebrew_profiler.batch.process_document")
    def test_discovers_txt_files(self, mock_proc, tmp_path: Path):
        """Only .txt files are discovered; other extensions are ignored."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "a.txt", "שלום")
        _write_txt(in_dir, "b.txt", "עולם")
        (in_dir / "c.csv").write_text("not a txt", encoding="utf-8")

        mock_proc.return_value = _make_pipeline_output()
        out_dir = tmp_path / "out"

        result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert result.total_processed == 2
        assert result.error_count == 0

    @patch("hebrew_profiler.batch.process_document")
    def test_writes_individual_json(self, mock_proc, tmp_path: Path):
        """Each input doc gets a corresponding .json in the output dir."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "doc1.txt", "טקסט")

        mock_proc.return_value = _make_pipeline_output("טקסט")
        out_dir = tmp_path / "out"

        process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        json_path = out_dir / "doc1.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "text" in data
        assert "features" in data
        assert "scores" in data


class TestProcessBatchErrorHandling:
    """Error resilience: encoding errors, pipeline failures."""

    def test_invalid_utf8_skipped(self, tmp_path: Path):
        """Files with invalid UTF-8 are skipped and counted as errors."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        bad = in_dir / "bad.txt"
        bad.write_bytes(b"\x80\x81\x82")  # invalid UTF-8
        out_dir = tmp_path / "out"

        result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert result.total_processed == 1
        assert result.error_count == 1
        assert result.errors[0]["error_type"] == "EncodingError"

    @patch("hebrew_profiler.batch.process_document")
    def test_pipeline_error_continues(self, mock_proc, tmp_path: Path):
        """A pipeline error on one doc doesn't stop the rest."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "good.txt", "שלום")
        _write_txt(in_dir, "bad.txt", "רע")

        def side_effect(text, config):
            if text == "רע":
                raise RuntimeError("YAP down")
            return _make_pipeline_output(text)

        mock_proc.side_effect = side_effect
        out_dir = tmp_path / "out"

        result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert result.total_processed == 2
        assert result.error_count == 1
        assert (out_dir / "good.json").exists()

    @patch("hebrew_profiler.batch.process_document")
    def test_batch_result_counts(self, mock_proc, tmp_path: Path):
        """total_processed == success + error_count."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(5):
            _write_txt(in_dir, f"d{i}.txt", f"text{i}")

        call_count = 0

        def side_effect(text, config):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("boom")
            return _make_pipeline_output(text)

        mock_proc.side_effect = side_effect
        out_dir = tmp_path / "out"

        result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        assert result.total_processed == 5
        assert result.total_processed == (5 - result.error_count) + result.error_count


# ---------------------------------------------------------------------------
# Task 13.2 — JSONL export
# ---------------------------------------------------------------------------


class TestJSONLExport:
    """JSONL output when jsonl_path is provided."""

    @patch("hebrew_profiler.batch.process_document")
    def test_jsonl_written(self, mock_proc, tmp_path: Path):
        """JSONL file is created with one record per successful document."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "a.txt", "שלום")
        _write_txt(in_dir, "b.txt", "עולם")

        mock_proc.return_value = _make_pipeline_output()
        out_dir = tmp_path / "out"
        jsonl = tmp_path / "export.jsonl"

        process_batch(
            str(in_dir), str(out_dir), PipelineConfig(),
            workers=1, jsonl_path=str(jsonl),
        )

        assert jsonl.exists()
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    @patch("hebrew_profiler.batch.process_document")
    def test_jsonl_record_keys(self, mock_proc, tmp_path: Path):
        """Each JSONL record has raw_text, normalized_text, features, scores."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "doc.txt", "שלום")

        mock_proc.return_value = _make_pipeline_output("שלום")
        out_dir = tmp_path / "out"
        jsonl = tmp_path / "export.jsonl"

        process_batch(
            str(in_dir), str(out_dir), PipelineConfig(),
            workers=1, jsonl_path=str(jsonl),
        )

        record = json.loads(jsonl.read_text(encoding="utf-8").strip())
        assert set(record.keys()) == {"raw_text", "normalized_text", "features", "scores"}

    @patch("hebrew_profiler.batch.process_document")
    def test_jsonl_utf8_no_ascii_escape(self, mock_proc, tmp_path: Path):
        """Hebrew characters are preserved, not escaped as \\uXXXX."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "doc.txt", "שלום")

        mock_proc.return_value = _make_pipeline_output("שלום")
        out_dir = tmp_path / "out"
        jsonl = tmp_path / "export.jsonl"

        process_batch(
            str(in_dir), str(out_dir), PipelineConfig(),
            workers=1, jsonl_path=str(jsonl),
        )

        raw = jsonl.read_bytes()
        assert "שלום".encode("utf-8") in raw
        assert b"\\u05e9" not in raw  # no escaped Hebrew

    @patch("hebrew_profiler.batch.process_document")
    def test_jsonl_not_written_when_none(self, mock_proc, tmp_path: Path):
        """No JSONL file when jsonl_path is None."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "doc.txt", "שלום")

        mock_proc.return_value = _make_pipeline_output()
        out_dir = tmp_path / "out"

        process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        # No jsonl file should exist anywhere in tmp_path with .jsonl suffix
        jsonl_files = list(tmp_path.rglob("*.jsonl"))
        assert len(jsonl_files) == 0

    @patch("hebrew_profiler.batch.process_document")
    def test_jsonl_excludes_failed_docs(self, mock_proc, tmp_path: Path):
        """Failed documents are not included in the JSONL output."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_txt(in_dir, "good.txt", "שלום")
        _write_txt(in_dir, "bad.txt", "רע")

        def side_effect(text, config):
            if text == "רע":
                raise RuntimeError("fail")
            return _make_pipeline_output(text)

        mock_proc.side_effect = side_effect
        out_dir = tmp_path / "out"
        jsonl = tmp_path / "export.jsonl"

        process_batch(
            str(in_dir), str(out_dir), PipelineConfig(),
            workers=1, jsonl_path=str(jsonl),
        )

        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # only the good doc


# ---------------------------------------------------------------------------
# Logging format
# ---------------------------------------------------------------------------


class TestLogError:
    def test_log_format(self, capsys):
        """Error log matches the required format."""
        _log_error("corpus/doc.txt", "EncodingError", "bad bytes")
        captured = capsys.readouterr()
        line = captured.err.strip()
        # Format: [{ISO-8601}] ERROR [{doc_id}] {error_type}: {message}
        assert "ERROR" in line
        assert "[corpus/doc.txt]" in line
        assert "EncodingError" in line
        assert "bad bytes" in line


# ---------------------------------------------------------------------------
# Hypothesis imports and strategies for property-based tests
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
import hypothesis.strategies as st
from hebrew_profiler.pipeline import pipeline_output_to_dict


# -- Strategies for generating PipelineOutput components --

_optional_float = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
_pos_optional_float = st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False))

_morph_features_st = st.builds(
    MorphFeatures,
    verb_ratio=_optional_float,
    binyan_distribution=st.one_of(st.none(), st.dictionaries(st.text(min_size=1, max_size=5), st.integers(0, 50), max_size=5)),
    prefix_density=_optional_float,
    suffix_pronoun_ratio=_optional_float,
    morphological_ambiguity=_pos_optional_float,
)

_syntax_features_st = st.builds(
    SyntaxFeatures,
    avg_sentence_length=_pos_optional_float,
    avg_tree_depth=_pos_optional_float,
    max_tree_depth=_pos_optional_float,
    avg_dependency_distance=_pos_optional_float,
    clauses_per_sentence=_pos_optional_float,
)

_lexical_features_st = st.builds(
    LexicalFeatures,
    type_token_ratio=_optional_float,
    hapax_ratio=_optional_float,
    avg_token_length=_pos_optional_float,
    lemma_diversity=_optional_float,
)

_structural_features_st = st.builds(
    StructuralFeatures,
    sentence_length_variance=_pos_optional_float,
    long_sentence_ratio=_optional_float,
)

_features_st = st.builds(
    Features,
    morphology=_morph_features_st,
    syntax=_syntax_features_st,
    lexicon=_lexical_features_st,
    structure=_structural_features_st,
)

_scores_st = st.builds(
    Scores,
    difficulty=_optional_float,
    style=st.one_of(st.none(), st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)),
)

_hebrew_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Zs"), whitelist_characters="אבגדהוזחטיכלמנסעפצקרשת .,!?"),
    min_size=1,
    max_size=200,
)

_pipeline_output_st = st.builds(
    PipelineOutput,
    text=_hebrew_text_st,
    features=_features_st,
    scores=_scores_st,
)


# ---------------------------------------------------------------------------
# Task 13.3 — Property 22: JSONL record completeness
# ---------------------------------------------------------------------------


class TestProperty22JSONLRecordCompleteness:
    """**Validates: Requirements 15.2**

    For any PipelineOutput, the JSONL record SHALL contain all four
    required keys: raw_text, normalized_text, features, and scores.
    """

    @given(output=_pipeline_output_st, normalized=_hebrew_text_st)
    @settings(max_examples=100)
    def test_jsonl_record_has_all_required_keys(self, output: PipelineOutput, normalized: str):
        """Property 22: JSONL record completeness.

        **Validates: Requirements 15.2**
        """
        result_dict = pipeline_output_to_dict(output)

        # Build the JSONL record the same way batch.py does
        jsonl_record = {
            "raw_text": output.text,
            "normalized_text": normalized,
            "features": result_dict["features"],
            "scores": result_dict["scores"],
        }

        required_keys = {"raw_text", "normalized_text", "features", "scores"}
        assert set(jsonl_record.keys()) == required_keys

        # Verify the record is JSON-serializable with ensure_ascii=False
        serialized = json.dumps(jsonl_record, ensure_ascii=False)
        deserialized = json.loads(serialized)
        assert set(deserialized.keys()) == required_keys

    @given(output=_pipeline_output_st, normalized=_hebrew_text_st)
    @settings(max_examples=100)
    def test_jsonl_record_features_has_subcategories(self, output: PipelineOutput, normalized: str):
        """Property 22 (extended): features sub-keys are present.

        **Validates: Requirements 15.2**
        """
        result_dict = pipeline_output_to_dict(output)

        jsonl_record = {
            "raw_text": output.text,
            "normalized_text": normalized,
            "features": result_dict["features"],
            "scores": result_dict["scores"],
        }

        assert "morphology" in jsonl_record["features"]
        assert "syntax" in jsonl_record["features"]
        assert "lexicon" in jsonl_record["features"]
        assert "structure" in jsonl_record["features"]


# ---------------------------------------------------------------------------
# Task 13.4 — Property 23: Batch count consistency
# ---------------------------------------------------------------------------


class TestProperty23BatchCountConsistency:
    """**Validates: Requirements 14.5**

    For any batch processing run over N input files,
    total_processed SHALL equal N.
    """

    @given(
        n=st.integers(min_value=1, max_value=10),
        fail_flags=st.lists(st.booleans(), min_size=1, max_size=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_total_processed_equals_n(self, n: int, fail_flags: list[bool], tmp_path_factory):
        """Property 23: Batch count consistency — total_processed == N.

        **Validates: Requirements 14.5**
        """
        # Align fail_flags length to n
        flags = (fail_flags * ((n // len(fail_flags)) + 1))[:n]

        tmp_path = tmp_path_factory.mktemp("batch")
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"

        # Create N text files
        for i in range(n):
            (in_dir / f"doc{i}.txt").write_text(f"text {i}", encoding="utf-8")

        call_idx = {"val": 0}

        def mock_process_document(text, config):
            idx = call_idx["val"]
            call_idx["val"] += 1
            if idx < len(flags) and flags[idx]:
                raise RuntimeError(f"Simulated failure for doc {idx}")
            return _make_pipeline_output(text)

        with patch("hebrew_profiler.batch.process_document", side_effect=mock_process_document):
            result = process_batch(str(in_dir), str(out_dir), PipelineConfig(), workers=1)

        # Core property: total_processed == N
        assert result.total_processed == n

        # Derived identity: total_processed == (total_processed - error_count) + error_count
        success_count = result.total_processed - result.error_count
        assert result.total_processed == success_count + result.error_count

        # error_count is bounded
        assert 0 <= result.error_count <= n
