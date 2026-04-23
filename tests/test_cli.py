"""Unit tests for the run_pipeline CLI entry point.

Tests argument parsing for single-document and batch modes,
and verifies non-zero exit codes on errors.

Requirements: 14.3, 16.5, 18.2, 18.3, 18.4
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from run_pipeline import build_parser, main


# ---------------------------------------------------------------------------
# Argument parsing — single mode
# ---------------------------------------------------------------------------


class TestSingleArgParsing:
    """Verify that the 'single' sub-command parses all expected flags."""

    def test_single_minimal(self):
        parser = build_parser()
        args = parser.parse_args(["single", "--input", "file.txt"])
        assert args.command == "single"
        assert args.input == "file.txt"
        assert args.output is None
        assert args.pretty is False
        assert args.yap_url == "http://localhost:8000/yap/heb/joint"

    def test_single_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "single",
            "--input", "in.txt",
            "--output", "out.json",
            "--pretty",
            "--yap-url", "http://custom:9000/yap",
        ])
        assert args.command == "single"
        assert args.input == "in.txt"
        assert args.output == "out.json"
        assert args.pretty is True
        assert args.yap_url == "http://custom:9000/yap"


# ---------------------------------------------------------------------------
# Argument parsing — batch mode
# ---------------------------------------------------------------------------


class TestBatchArgParsing:
    """Verify that the 'batch' sub-command parses all expected flags."""

    def test_batch_minimal(self):
        parser = build_parser()
        args = parser.parse_args(["batch", "--input", "corpus/", "--output", "results/"])
        assert args.command == "batch"
        assert args.input == "corpus/"
        assert args.output == "results/"
        assert args.workers == 4
        assert args.jsonl is None
        assert args.pretty is False

    def test_batch_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "batch",
            "--input", "corpus/",
            "--output", "results/",
            "--workers", "8",
            "--jsonl", "data.jsonl",
            "--pretty",
            "--yap-url", "http://custom:9000/yap",
        ])
        assert args.command == "batch"
        assert args.workers == 8
        assert args.jsonl == "data.jsonl"
        assert args.pretty is True
        assert args.yap_url == "http://custom:9000/yap"


# ---------------------------------------------------------------------------
# No sub-command → exit code 1
# ---------------------------------------------------------------------------


class TestNoCommand:
    """Calling main() with no sub-command should return exit code 1."""

    def test_no_subcommand_returns_1(self):
        assert main([]) == 1


# ---------------------------------------------------------------------------
# Single mode — error exit codes
# ---------------------------------------------------------------------------


class TestSingleErrorExitCodes:
    """Verify non-zero exit code when single-document processing fails."""

    def test_missing_input_file_returns_1(self):
        """Non-existent input file → exit 1."""
        code = main(["single", "--input", "/nonexistent/path/file.txt"])
        assert code == 1

    @patch("run_pipeline.process_document", side_effect=RuntimeError("boom"))
    def test_pipeline_error_returns_1(self, _mock_proc):
        """Pipeline exception → exit 1."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("שלום עולם")
            tmp = f.name
        try:
            code = main(["single", "--input", tmp])
            assert code == 1
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Single mode — success path (mocked pipeline)
# ---------------------------------------------------------------------------


class TestSingleSuccess:
    """Verify single mode writes correct output when pipeline succeeds."""

    @staticmethod
    def _fake_output():
        """Return a minimal PipelineOutput-like object."""
        from hebrew_profiler.models import (
            Features,
            LexicalFeatures,
            MorphFeatures,
            PipelineOutput,
            Scores,
            StructuralFeatures,
            SyntaxFeatures,
            DiscourseFeatures,
            StyleFeatures,
        )
        return PipelineOutput(
            text="שלום",
            features=Features(
                morphology=MorphFeatures(0.0, {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                syntax=SyntaxFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                lexicon=LexicalFeatures(0.0, 0.0, 0.0, 0.0, None, 0.0),
                structure=StructuralFeatures(0.0, 0.0, 0.0, 0.0, 0.0),
                discourse=DiscourseFeatures(0.0, 0.0, 0.0),
                style=StyleFeatures(0.0, 0.0),
            ),
            scores=Scores(difficulty=0.5, style=0.3),
        )

    @patch("run_pipeline.process_document")
    def test_single_stdout(self, mock_proc, capsys):
        mock_proc.return_value = self._fake_output()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("שלום")
            tmp = f.name
        try:
            code = main(["single", "--input", tmp])
            assert code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert "text" in data
            assert "features" in data
            assert "scores" in data
        finally:
            os.unlink(tmp)

    @patch("run_pipeline.process_document")
    def test_single_to_file(self, mock_proc):
        mock_proc.return_value = self._fake_output()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("שלום")
            tmp_in = f.name
        out_fd, tmp_out = tempfile.mkstemp(suffix=".json")
        os.close(out_fd)
        try:
            code = main(["single", "--input", tmp_in, "--output", tmp_out, "--pretty"])
            assert code == 0
            with open(tmp_out, "r", encoding="utf-8") as fh:
                data = json.loads(fh.read())
            assert data["text"] == "שלום"
        finally:
            os.unlink(tmp_in)
            os.unlink(tmp_out)


# ---------------------------------------------------------------------------
# Batch mode — success path (mocked)
# ---------------------------------------------------------------------------


class TestBatchSuccess:
    """Verify batch mode calls process_batch and prints summary."""

    @patch("run_pipeline.process_batch")
    def test_batch_prints_summary(self, mock_batch, capsys):
        from hebrew_profiler.models import BatchResult
        mock_batch.return_value = BatchResult(
            total_processed=5, error_count=1, errors=[{"document": "bad.txt", "error_type": "EncodingError", "message": "bad"}]
        )
        code = main([
            "batch", "--input", "corpus/", "--output", "results/",
            "--workers", "2", "--jsonl", "out.jsonl",
        ])
        assert code == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert summary["total_processed"] == 5
        assert summary["error_count"] == 1
