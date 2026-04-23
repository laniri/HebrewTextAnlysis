"""Data export module for the ML Distillation Layer (Layer 6).

Converts existing pipeline output JSON files into training JSONL with
soft labels (continuous severity values).  Runs only the fast analysis
and diagnosis layers on pre-computed pipeline features — does NOT invoke
Stanza, YAP, or any external NLP service.

Requirements implemented: 1.1–1.4, 2.1–2.4, 3.1–3.4, 4.1–4.4,
5.1–5.5, 6.1–6.4, 27.1, 27.3, 27.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.diagnosis_engine import run_diagnoses
from analysis.diagnosis_models import Diagnosis
from analysis.issue_detector import detect_issues
from analysis.issue_models import Issue
from analysis.sentence_metrics import SentenceMetrics
from analysis.statistics import flatten_corpus_json, load_stats
from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS
from ml.sentence_utils import split_into_sentences


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_pipeline_json(path: str) -> dict | None:
    """Read and validate a single pipeline JSON file.

    Returns the parsed dict if it contains the required keys
    (``"text"``, ``"features"``, ``"scores"``), or ``None`` on any error.
    Logs a warning to stderr on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING [{path}] read_error: {exc}", file=sys.stderr)
        return None

    missing = [k for k in ("text", "features", "scores") if k not in data]
    if missing:
        print(
            f"WARNING [{path}] missing_fields: {', '.join(missing)}",
            file=sys.stderr,
        )
        return None

    return data


def _flatten_issues(issues: list[Issue]) -> dict[str, float]:
    """Map 17 issue types to max severity, 0.0 for absent types.

    For each canonical issue key, takes the maximum severity among all
    ``Issue`` objects of that type.  Types not present in *issues* get 0.0.
    Always returns exactly 17 key-value pairs.
    """
    result: dict[str, float] = {k: 0.0 for k in _ISSUE_KEYS}
    for issue in issues:
        if issue.type in result:
            result[issue.type] = max(result[issue.type], issue.severity)
    return result


def _flatten_diagnoses(diagnoses: list[Diagnosis]) -> dict[str, float]:
    """Map 8 diagnosis types to severity, 0.0 for inactive.

    For each canonical diagnosis key, uses the diagnosis severity if
    activated, or 0.0 if the type is absent.  Always returns exactly
    8 key-value pairs.
    """
    result: dict[str, float] = {k: 0.0 for k in _DIAGNOSIS_KEYS}
    for diag in diagnoses:
        if diag.type in result:
            result[diag.type] = diag.severity
    return result


def _build_sentence_metrics_from_json(
    data: dict,
) -> list[SentenceMetrics]:
    """Build SentenceMetrics from the ``sentence_metrics`` field in a pipeline JSON.

    When the field is present (pipeline was run with the updated code),
    constructs accurate SentenceMetrics with real token counts, tree depths,
    and lemma sets from the IR.

    When the field is absent (old pipeline JSONs), falls back to the
    synthetic approximation using text splitting and document-level features.
    """
    stored = data.get("sentence_metrics")
    if stored is not None and len(stored) > 0:
        # Real sentence metrics from the pipeline
        result: list[SentenceMetrics] = []
        for sm in stored:
            result.append(SentenceMetrics(
                index=sm["index"],
                token_count=sm["token_count"],
                tree_depth=float(sm["tree_depth"]),
                lemma_set=frozenset(sm.get("lemmas", [])),
            ))
        return result

    # Fallback: synthetic approximation for old pipeline JSONs
    return _build_synthetic_sentence_metrics(
        data.get("text", ""),
        data.get("features", {}),
    )


def _build_synthetic_sentence_metrics(
    text: str,
    features: dict,
) -> list[SentenceMetrics]:
    """Build approximate SentenceMetrics from the pipeline JSON text and features.

    Since we don't have the IR during export, we approximate:
    - token_count: word count per sentence (split on whitespace)
    - tree_depth: document-level avg_tree_depth for every sentence
      (imperfect but gives the issue detector something to work with)
    - lemma_set: frozenset of whitespace-split words (surface forms,
      not true lemmas — a reasonable proxy for Jaccard cohesion)

    Returns one SentenceMetrics per sentence.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Get document-level avg_tree_depth as a fallback per-sentence depth
    syntax = features.get("syntax", {}) if isinstance(features, dict) else {}
    avg_depth = syntax.get("avg_tree_depth")
    if avg_depth is None:
        avg_depth = 0.0

    result: list[SentenceMetrics] = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        result.append(SentenceMetrics(
            index=i,
            token_count=len(words),
            tree_depth=float(avg_depth),
            lemma_set=frozenset(words),
        ))
    return result


def _extract_sentence_labels(issues: list[Issue], sentence_count: int) -> dict:
    """Extract per-sentence and per-pair labels from the issue list.

    Parameters
    ----------
    issues:
        Flat list of ``Issue`` objects produced by ``detect_issues()``.
    sentence_count:
        Number of sentences in the document.

    Returns
    -------
    dict with keys:
        ``"sentence_complexities"`` — list of *sentence_count* floats.
        ``"cohesion_pairs"`` — list of ``max(0, sentence_count - 1)`` floats.
    """
    sentence_complexities = [0.0] * sentence_count
    cohesion_pairs = [0.0] * max(0, sentence_count - 1)

    for issue in issues:
        if issue.type == "sentence_complexity" and len(issue.span) == 1:
            idx = issue.span[0]
            if 0 <= idx < sentence_count:
                sentence_complexities[idx] = issue.severity
        elif issue.type == "weak_cohesion" and len(issue.span) == 2:
            i, j = issue.span
            if j == i + 1 and 0 <= i < len(cohesion_pairs):
                cohesion_pairs[i] = issue.severity

    return {
        "sentence_complexities": sentence_complexities,
        "cohesion_pairs": cohesion_pairs,
    }


def _derive_sentence_count(issues: list[Issue], text: str) -> int:
    """Derive the sentence count from issues or by splitting the text.

    Counts ``sentence_complexity`` issues (one per sentence) when available,
    otherwise falls back to splitting the text on sentence-ending punctuation.
    """
    sc_issues = [i for i in issues if i.type == "sentence_complexity"]
    if sc_issues:
        return len(sc_issues)
    # Fallback: split on sentence-ending punctuation
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p for p in parts if p]
    return max(len(parts), 1)


def _compute_label_stats(records: list[dict]) -> dict:
    """Compute label distribution statistics across all training records.

    Returns a dict with:
    - ``"scores"``: mean and std for each of the 5 score values.
    - ``"issues"``: activation rate (fraction where severity > 0.0) per type.
    - ``"diagnoses"``: activation rate per type.
    """
    n = len(records)
    if n == 0:
        return {
            "scores": {k: {"mean": 0.0, "std": 0.0} for k in _SCORE_KEYS},
            "issues": {k: {"activation_rate": 0.0} for k in _ISSUE_KEYS},
            "diagnoses": {k: {"activation_rate": 0.0} for k in _DIAGNOSIS_KEYS},
        }

    # Score statistics
    score_arrays: dict[str, list[float]] = {k: [] for k in _SCORE_KEYS}
    for rec in records:
        for k in _SCORE_KEYS:
            score_arrays[k].append(rec["scores"][k])

    score_stats: dict[str, dict[str, float]] = {}
    for k in _SCORE_KEYS:
        arr = np.array(score_arrays[k], dtype=float)
        score_stats[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    # Issue activation rates
    issue_stats: dict[str, dict[str, float]] = {}
    for k in _ISSUE_KEYS:
        active = sum(1 for rec in records if rec["issues"][k] > 0.0)
        issue_stats[k] = {"activation_rate": active / n}

    # Diagnosis activation rates
    diag_stats: dict[str, dict[str, float]] = {}
    for k in _DIAGNOSIS_KEYS:
        active = sum(1 for rec in records if rec["diagnoses"][k] > 0.0)
        diag_stats[k] = {"activation_rate": active / n}

    return {
        "scores": score_stats,
        "issues": issue_stats,
        "diagnoses": diag_stats,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_training_data(
    input_dirs: list[str],
    stats_path: str,
    output_path: str,
    stats_output_path: str | None = None,
    embed: bool = False,
    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    workers: int = 1,
) -> None:
    """Read pipeline JSONs from *input_dirs*, run analysis + diagnosis layers,
    write Training_Records to *output_path* as JSONL.

    When *embed* is True, loads a sentence-transformers model and computes
    per-sentence embeddings for embedding-based cohesion detection (cosine
    similarity instead of Jaccard fallback).

    When *workers* > 1, uses a thread pool for parallel JSON reading and
    feature flattening (the embedding model itself is single-threaded but
    the I/O and analysis computation can overlap).

    Writes label distribution statistics to *stats_output_path* if provided.
    Logs progress and skip count to stderr.
    """
    # Load corpus feature statistics for issue detection
    feature_stats = load_stats(stats_path)

    # Load sentence embedder if requested
    embedder = None
    if embed:
        from analysis.embedder import get_embedder
        embedder = get_embedder(embed_model)
        if embedder is None:
            print(
                "[export] WARNING: sentence-transformers not available — "
                "falling back to Jaccard cohesion.",
                file=sys.stderr,
            )
        else:
            print(
                f"[export] Using sentence embeddings ({embed_model}) for cohesion.",
                file=sys.stderr,
            )

    # Collect all JSON file paths from input directories
    json_paths: list[Path] = []
    for dir_path in input_dirs:
        p = Path(dir_path)
        if p.is_dir():
            json_paths.extend(sorted(p.glob("*.json")))

    total_files = len(json_paths)
    print(f"[export] Found {total_files} JSON files to process.", file=sys.stderr)

    processed = 0
    skipped = 0
    records: list[dict] = []
    import time as _time
    t0 = _time.time()

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for jp in json_paths:
            data = _read_pipeline_json(str(jp))
            if data is None:
                skipped += 1
                continue

            text = data["text"]
            features = data["features"]
            scores_raw = data["scores"]

            # Substitute 0.0 for None score values
            scores: dict[str, float] = {}
            for k in _SCORE_KEYS:
                val = scores_raw.get(k)
                scores[k] = 0.0 if val is None else float(val)

            # Flatten features using the existing utility
            raw_features = flatten_corpus_json({"features": features})

            # Detect issues (with real or synthetic sentence metrics)
            sentence_metrics = _build_sentence_metrics_from_json(data)

            # Populate sentence embeddings for embedding-based cohesion
            if embedder is not None and sentence_metrics:
                sentences_text = split_into_sentences(text)
                texts_to_embed = sentences_text[:len(sentence_metrics)]
                if texts_to_embed:
                    embeddings = embedder.embed(texts_to_embed)
                    for sm, vec in zip(sentence_metrics, embeddings):
                        sm.embedding = vec

            issues = detect_issues(raw_features, sentence_metrics, feature_stats)

            # Run diagnoses
            diagnoses = run_diagnoses(issues, scores)

            # Flatten to fixed-size dicts
            flat_issues = _flatten_issues(issues)
            flat_diagnoses = _flatten_diagnoses(diagnoses)

            # Extract sentence-level labels
            sentence_count = _derive_sentence_count(issues, text)
            sentence_labels = _extract_sentence_labels(issues, sentence_count)

            record = {
                "text": text,
                "scores": scores,
                "issues": flat_issues,
                "diagnoses": flat_diagnoses,
                "sentence_complexities": sentence_labels["sentence_complexities"],
                "cohesion_pairs": sentence_labels["cohesion_pairs"],
            }
            records.append(record)

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            # Progress logging every 100 documents
            if processed % 100 == 0:
                elapsed = _time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_files - processed - skipped) / rate if rate > 0 else 0
                print(
                    f"[export] {processed}/{total_files} "
                    f"({elapsed:.0f}s elapsed, {rate:.1f} docs/s, ~{eta:.0f}s remaining)",
                    file=sys.stderr,
                )

    elapsed = _time.time() - t0
    print(
        f"[export] Processed {processed} documents, skipped {skipped} "
        f"in {elapsed:.1f}s ({processed / elapsed:.1f} docs/s).",
        file=sys.stderr,
    )

    # Write label distribution stats if requested
    if stats_output_path is not None:
        label_stats = _compute_label_stats(records)
        with open(stats_output_path, "w", encoding="utf-8") as sf:
            json.dump(label_stats, sf, indent=2, ensure_ascii=False)
        print(f"[export] Label stats written to {stats_output_path}.", file=sys.stderr)
