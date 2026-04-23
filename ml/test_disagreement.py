# Feature: ml-distillation-layer, Property 13: Disagreement detection correctness
# Feature: ml-distillation-layer, Property 14: Training data deduplication on merge

"""Property-based and unit tests for ml/disagreement.py.

Tests the disagreement mining module: disagreement detection correctness,
training data deduplication on merge, per-type disagreement rates, JSONL
output format, merge record counts, and mismatched record handling.

**Validates: Requirements 20.1, 20.2, 20.3, 20.4, 21.1, 21.2, 21.3**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from ml.disagreement import find_disagreements, merge_training_data, _compute_disagreement
from ml.model import _SCORE_KEYS, _ISSUE_KEYS, _DIAGNOSIS_KEYS

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_severity = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_scores_strategy = st.fixed_dictionaries({k: _severity for k in _SCORE_KEYS})
_issues_strategy = st.fixed_dictionaries({k: _severity for k in _ISSUE_KEYS})
_diagnoses_strategy = st.fixed_dictionaries({k: _severity for k in _DIAGNOSIS_KEYS})

_record_strategy = st.fixed_dictionaries({
    "scores": _scores_strategy,
    "issues": _issues_strategy,
    "diagnoses": _diagnoses_strategy,
})

_threshold_strategy = st.floats(
    min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False,
)

_training_record_strategy = st.fixed_dictionaries({
    "text": st.text(min_size=1, max_size=20),
    "scores": _scores_strategy,
    "issues": _issues_strategy,
    "diagnoses": _diagnoses_strategy,
})


# ===========================================================================
# Property 13: Disagreement detection correctness
# ===========================================================================

# Feature: ml-distillation-layer, Property 13: Disagreement detection correctness
# **Validates: Requirements 20.1, 20.2**


@given(pred=_record_strategy, label=_record_strategy, threshold=_threshold_strategy)
@settings(max_examples=100)
def test_disagreement_detection_correctness(
    pred: dict, label: dict, threshold: float,
) -> None:
    """For any pair of prediction and label dicts, _compute_disagreement
    computes the absolute severity difference for each score, issue type,
    and diagnosis type, and flags the document as a disagreement case if
    and only if at least one difference exceeds the configured threshold."""
    is_disagreement, diffs = _compute_disagreement(pred, label, threshold)

    # Verify diffs contain all expected keys
    expected_keys = (
        {f"score_{k}" for k in _SCORE_KEYS}
        | {f"issue_{k}" for k in _ISSUE_KEYS}
        | {f"diagnosis_{k}" for k in _DIAGNOSIS_KEYS}
    )
    assert set(diffs.keys()) == expected_keys

    # Verify each diff is the absolute difference of the corresponding values
    for k in _SCORE_KEYS:
        expected_diff = abs(
            float(pred.get("scores", {}).get(k, 0.0))
            - float(label.get("scores", {}).get(k, 0.0))
        )
        assert abs(diffs[f"score_{k}"] - expected_diff) < 1e-12

    for k in _ISSUE_KEYS:
        expected_diff = abs(
            float(pred.get("issues", {}).get(k, 0.0))
            - float(label.get("issues", {}).get(k, 0.0))
        )
        assert abs(diffs[f"issue_{k}"] - expected_diff) < 1e-12

    for k in _DIAGNOSIS_KEYS:
        expected_diff = abs(
            float(pred.get("diagnoses", {}).get(k, 0.0))
            - float(label.get("diagnoses", {}).get(k, 0.0))
        )
        assert abs(diffs[f"diagnosis_{k}"] - expected_diff) < 1e-12

    # Verify is_disagreement iff at least one diff exceeds threshold
    any_exceeds = any(d > threshold for d in diffs.values())
    assert is_disagreement == any_exceeds, (
        f"is_disagreement={is_disagreement} but any_exceeds={any_exceeds} "
        f"(threshold={threshold})"
    )


# ===========================================================================
# Property 14: Training data deduplication on merge
# ===========================================================================

# Feature: ml-distillation-layer, Property 14: Training data deduplication on merge
# **Validates: Requirements 21.2**


@given(
    base_records=st.lists(_training_record_strategy, min_size=1, max_size=5),
    disagreement_records=st.lists(_training_record_strategy, min_size=1, max_size=5),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_training_data_deduplication_on_merge(
    base_records: list[dict],
    disagreement_records: list[dict],
    tmp_path_factory,
) -> None:
    """For any base training JSONL and disagreement JSONL containing records
    with overlapping text content, the merge function produces an output
    where each unique text appears exactly once, with the labels from the
    disagreement file taking precedence."""
    tmp_path = tmp_path_factory.mktemp("merge")

    # Write base JSONL
    base_path = tmp_path / "base.jsonl"
    with open(base_path, "w", encoding="utf-8") as fh:
        for rec in base_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write disagreement JSONL
    dis_path = tmp_path / "disagreements.jsonl"
    with open(dis_path, "w", encoding="utf-8") as fh:
        for rec in disagreement_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Merge
    output_path = tmp_path / "merged.jsonl"
    result = merge_training_data(
        str(base_path), str(dis_path), str(output_path),
    )

    # Read merged output
    merged_records = []
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                merged_records.append(json.loads(line))

    # Collect all unique texts from both inputs
    all_texts = set()
    for rec in base_records:
        all_texts.add(rec["text"])
    for rec in disagreement_records:
        all_texts.add(rec["text"])

    # Each unique text appears exactly once in merged output
    merged_texts = [rec["text"] for rec in merged_records]
    assert len(merged_texts) == len(set(merged_texts)), (
        "Duplicate texts found in merged output"
    )
    assert set(merged_texts) == all_texts

    # Total count matches unique texts
    assert result["total"] == len(all_texts)

    # Disagreement labels take precedence for overlapping texts
    # Build lookup from disagreement records (last one wins for duplicates)
    dis_lookup: dict[str, dict] = {}
    for rec in disagreement_records:
        dis_lookup[rec["text"]] = rec

    merged_lookup: dict[str, dict] = {}
    for rec in merged_records:
        merged_lookup[rec["text"]] = rec

    for text, dis_rec in dis_lookup.items():
        merged_rec = merged_lookup[text]
        # The merged record should have the disagreement file's labels
        assert merged_rec["scores"] == dis_rec["scores"]
        assert merged_rec["issues"] == dis_rec["issues"]
        assert merged_rec["diagnoses"] == dis_rec["diagnoses"]


# ===========================================================================
# Unit tests for disagreement mining module (Task 8.4)
# ===========================================================================


class TestDisagreementUnit:
    """Unit tests for the disagreement mining module."""

    @staticmethod
    def _make_record(
        text: str = "טקסט לדוגמה",
        score_val: float = 0.5,
        issue_val: float = 0.0,
        diag_val: float = 0.0,
    ) -> dict:
        """Build a Training_Record with uniform values."""
        return {
            "text": text,
            "scores": {k: score_val for k in _SCORE_KEYS},
            "issues": {k: issue_val for k in _ISSUE_KEYS},
            "diagnoses": {k: diag_val for k in _DIAGNOSIS_KEYS},
        }

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- Test: per-type disagreement rates computation ---------------------

    def test_per_type_disagreement_rates(self, tmp_path: Path) -> None:
        """Per-type rates reflect the fraction of documents where each type
        exceeds the threshold."""
        # Two documents: first has a big score difference, second does not
        pred1 = self._make_record(score_val=0.0, issue_val=0.0, diag_val=0.0)
        pred2 = self._make_record(score_val=0.5, issue_val=0.0, diag_val=0.0)

        label1 = self._make_record(score_val=0.8, issue_val=0.0, diag_val=0.0)
        label2 = self._make_record(score_val=0.5, issue_val=0.0, diag_val=0.0)

        preds_path = tmp_path / "preds.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        output_path = tmp_path / "disagreements.jsonl"

        self._write_jsonl(preds_path, [pred1, pred2])
        self._write_jsonl(labels_path, [label1, label2])

        result = find_disagreements(
            str(preds_path), str(labels_path), str(output_path), threshold=0.3,
        )

        assert result["total_documents"] == 2
        assert result["total_disagreements"] == 1

        # Score types should have rate 0.5 (1 out of 2 docs) for doc1
        # where diff = 0.8 > 0.3
        for k in _SCORE_KEYS:
            assert result["per_type_rates"][f"score_{k}"] == 0.5

        # Issue and diagnosis types should have rate 0.0
        for k in _ISSUE_KEYS:
            assert result["per_type_rates"][f"issue_{k}"] == 0.0
        for k in _DIAGNOSIS_KEYS:
            assert result["per_type_rates"][f"diagnosis_{k}"] == 0.0

    # -- Test: JSONL output format matches Training_Record schema ----------

    def test_jsonl_output_format(self, tmp_path: Path) -> None:
        """Flagged disagreement records are written as valid JSONL with the
        pipeline labels (Training_Record schema)."""
        pred = self._make_record(score_val=0.0)
        label = self._make_record(score_val=1.0)  # big difference

        preds_path = tmp_path / "preds.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        output_path = tmp_path / "disagreements.jsonl"

        self._write_jsonl(preds_path, [pred])
        self._write_jsonl(labels_path, [label])

        find_disagreements(
            str(preds_path), str(labels_path), str(output_path), threshold=0.3,
        )

        # Read output and verify format
        with open(output_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        assert len(lines) == 1
        record = json.loads(lines[0])

        # Must have Training_Record keys
        assert set(record.keys()) == {"text", "scores", "issues", "diagnoses"}
        assert len(record["scores"]) == 5
        assert len(record["issues"]) == 17
        assert len(record["diagnoses"]) == 8

        # Values should come from the label (pipeline labels), not predictions
        for k in _SCORE_KEYS:
            assert record["scores"][k] == label["scores"][k]

    # -- Test: merge record counts (added + existing = total) --------------

    def test_merge_record_counts(self, tmp_path: Path) -> None:
        """Merge produces correct added and total counts."""
        base = [
            self._make_record(text="טקסט א"),
            self._make_record(text="טקסט ב"),
        ]
        disagreements = [
            self._make_record(text="טקסט ג"),  # new
            self._make_record(text="טקסט א", score_val=0.9),  # duplicate, updated
        ]

        base_path = tmp_path / "base.jsonl"
        dis_path = tmp_path / "disagreements.jsonl"
        output_path = tmp_path / "merged.jsonl"

        self._write_jsonl(base_path, base)
        self._write_jsonl(dis_path, disagreements)

        result = merge_training_data(
            str(base_path), str(dis_path), str(output_path),
        )

        # "טקסט א" is a duplicate (overwritten), "טקסט ג" is new
        # Base had 2 unique texts, after merge we have 3 unique texts
        # added = 3 - 2 = 1 (only "טקסט ג" is truly new)
        assert result["added"] == 1
        assert result["total"] == 3

        # Verify output file
        with open(output_path, "r", encoding="utf-8") as fh:
            merged = [json.loads(l) for l in fh if l.strip()]

        assert len(merged) == 3
        texts = [r["text"] for r in merged]
        assert set(texts) == {"טקסט א", "טקסט ב", "טקסט ג"}

        # "טקסט א" should have the disagreement labels (score_val=0.9)
        for rec in merged:
            if rec["text"] == "טקסט א":
                assert rec["scores"]["difficulty"] == 0.9

    # -- Test: handling of mismatched record counts ------------------------

    def test_mismatched_record_counts(self, tmp_path: Path, capsys) -> None:
        """When predictions and labels have different counts, a warning is
        logged and only the minimum number of records is processed."""
        preds = [
            self._make_record(text="doc1"),
            self._make_record(text="doc2"),
            self._make_record(text="doc3"),
        ]
        labels = [
            self._make_record(text="doc1"),
            self._make_record(text="doc2"),
        ]

        preds_path = tmp_path / "preds.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        output_path = tmp_path / "disagreements.jsonl"

        self._write_jsonl(preds_path, preds)
        self._write_jsonl(labels_path, labels)

        result = find_disagreements(
            str(preds_path), str(labels_path), str(output_path), threshold=0.3,
        )

        # Should process only min(3, 2) = 2 records
        assert result["total_documents"] == 2

        # Warning should be logged to stderr
        captured = capsys.readouterr()
        assert "record_count_mismatch" in captured.err
        assert "predictions=3" in captured.err
        assert "labels=2" in captured.err
