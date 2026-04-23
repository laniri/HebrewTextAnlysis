"""Disagreement mining module for the ML Distillation Layer (Layer 6).

Compares student model predictions against teacher pipeline labels to
identify divergence cases, and merges flagged documents back into the
training set for iterative improvement.

Requirements implemented: 20.1–20.4, 21.1–21.3, 27.1, 27.3.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping malformed lines with a warning."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING [{path}:{line_no}] malformed_jsonl: {exc}",
                    file=sys.stderr,
                )
    return records


def _compute_disagreement(
    pred: dict,
    label: dict,
    threshold: float,
) -> tuple[bool, dict[str, float]]:
    """Check whether a single document is a disagreement case.

    Computes the absolute severity difference for each score, issue type,
    and diagnosis type.  Returns ``(is_disagreement, per_key_diffs)`` where
    *is_disagreement* is ``True`` when ANY difference exceeds *threshold*.
    """
    diffs: dict[str, float] = {}
    is_disagreement = False

    # Score differences
    pred_scores = pred.get("scores", {})
    label_scores = label.get("scores", {})
    for key in _SCORE_KEYS:
        diff = abs(float(pred_scores.get(key, 0.0)) - float(label_scores.get(key, 0.0)))
        diffs[f"score_{key}"] = diff
        if diff > threshold:
            is_disagreement = True

    # Issue differences
    pred_issues = pred.get("issues", {})
    label_issues = label.get("issues", {})
    for key in _ISSUE_KEYS:
        diff = abs(float(pred_issues.get(key, 0.0)) - float(label_issues.get(key, 0.0)))
        diffs[f"issue_{key}"] = diff
        if diff > threshold:
            is_disagreement = True

    # Diagnosis differences
    pred_diags = pred.get("diagnoses", {})
    label_diags = label.get("diagnoses", {})
    for key in _DIAGNOSIS_KEYS:
        diff = abs(float(pred_diags.get(key, 0.0)) - float(label_diags.get(key, 0.0)))
        diffs[f"diagnosis_{key}"] = diff
        if diff > threshold:
            is_disagreement = True

    return is_disagreement, diffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_disagreements(
    predictions_path: str,
    labels_path: str,
    output_path: str,
    threshold: float = 0.3,
) -> dict:
    """Compare model predictions against pipeline labels.

    For each pair of records (matched by line position), computes the
    absolute severity difference for every score, issue type, and
    diagnosis type.  A document is flagged as a disagreement case if
    ANY difference exceeds *threshold*.

    Flagged documents are written to *output_path* as JSONL using the
    pipeline labels (from *labels_path*).

    Parameters
    ----------
    predictions_path:
        Path to JSONL file with model predictions (Training_Record format).
    labels_path:
        Path to JSONL file with pipeline labels (Training_Record format).
    output_path:
        Path where flagged disagreement records are written as JSONL.
    threshold:
        Severity difference threshold for flagging (default 0.3).

    Returns
    -------
    Summary dict::

        {
            "total_disagreements": int,
            "total_documents": int,
            "per_type_rates": {str: float},
        }
    """
    predictions = _read_jsonl(predictions_path)
    labels = _read_jsonl(labels_path)

    # Warn on mismatched record counts
    if len(predictions) != len(labels):
        print(
            f"WARNING [disagreement] record_count_mismatch: "
            f"predictions={len(predictions)}, labels={len(labels)}. "
            f"Processing only the first {min(len(predictions), len(labels))} records.",
            file=sys.stderr,
        )

    n = min(len(predictions), len(labels))
    total_disagreements = 0

    # Track per-type disagreement counts
    type_disagree_counts: dict[str, int] = {}
    for key in _SCORE_KEYS:
        type_disagree_counts[f"score_{key}"] = 0
    for key in _ISSUE_KEYS:
        type_disagree_counts[f"issue_{key}"] = 0
    for key in _DIAGNOSIS_KEYS:
        type_disagree_counts[f"diagnosis_{key}"] = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for i in range(n):
            is_disagreement, diffs = _compute_disagreement(
                predictions[i], labels[i], threshold,
            )
            if is_disagreement:
                total_disagreements += 1
                # Write the pipeline labels (ground truth) for retraining
                out_fh.write(json.dumps(labels[i], ensure_ascii=False) + "\n")

            # Accumulate per-type disagreement counts
            for key, diff in diffs.items():
                if diff > threshold:
                    type_disagree_counts[key] = type_disagree_counts.get(key, 0) + 1

    # Compute per-type rates
    per_type_rates: dict[str, float] = {}
    for key, count in type_disagree_counts.items():
        per_type_rates[key] = count / n if n > 0 else 0.0

    print(
        f"[disagreement] Found {total_disagreements}/{n} disagreement cases "
        f"(threshold={threshold}).",
        file=sys.stderr,
    )

    return {
        "total_disagreements": total_disagreements,
        "total_documents": n,
        "per_type_rates": per_type_rates,
    }


def merge_training_data(
    base_path: str,
    disagreements_path: str,
    output_path: str,
) -> dict:
    """Merge disagreement records into existing training JSONL.

    Reads both files, deduplicates by text content, and keeps the most
    recent labels (from *disagreements_path*) when duplicates exist.

    Parameters
    ----------
    base_path:
        Path to the base training JSONL file.
    disagreements_path:
        Path to the disagreement JSONL file produced by
        :func:`find_disagreements`.
    output_path:
        Path where the merged output is written as JSONL.

    Returns
    -------
    dict::

        {"added": int, "total": int}
    """
    base_records = _read_jsonl(base_path)
    disagreement_records = _read_jsonl(disagreements_path)

    # Build a dict keyed by text content.  Base records go in first,
    # then disagreement records overwrite duplicates (most recent wins).
    merged: dict[str, dict] = {}
    for rec in base_records:
        text = rec.get("text", "")
        merged[text] = rec

    base_count = len(merged)

    for rec in disagreement_records:
        text = rec.get("text", "")
        merged[text] = rec  # overwrites if duplicate

    added = len(merged) - base_count
    total = len(merged)

    # Write merged output
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for rec in merged.values():
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[merge] Added {added} new records. Total training set size: {total}.",
        file=sys.stderr,
    )

    return {"added": added, "total": total}
