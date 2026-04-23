"""Statistics module for the probabilistic analysis layer.

Computes and persists corpus statistics from raw feature values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class FeatureStats:
    mean: float
    std: float
    min: float
    max: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    valid_count: int
    unstable: bool   # True when valid_count < 30
    degenerate: bool  # True when std == 0


def compute_feature_stats(
    feature_dicts: List[Dict[str, float | None]]
) -> Dict[str, FeatureStats]:
    """Compute FeatureStats for every scalar key across all feature dicts.

    Collects all non-None values per key, clips outliers to the [p5, p95]
    range before computing mean and std (population), then computes
    percentiles from the original (unclipped) data.

    Outlier capping prevents extreme values in diverse corpora (e.g., web
    crawl) from skewing the mean/std that soft_score uses for z-scores.

    Sets unstable=True when valid_count < 30.
    Sets degenerate=True when std == 0.
    """
    values_per_key: Dict[str, List[float]] = {}
    for d in feature_dicts:
        for key, value in d.items():
            if value is not None:
                values_per_key.setdefault(key, []).append(value)

    result: Dict[str, FeatureStats] = {}
    for key, values in values_per_key.items():
        arr = np.array(values, dtype=float)
        valid_count = len(arr)

        # Compute percentiles from original data
        p5 = float(np.percentile(arr, 5))
        p10 = float(np.percentile(arr, 10))
        p25 = float(np.percentile(arr, 25))
        p50 = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))

        # Clip to [p5, p95] for mean/std computation to reduce outlier impact
        clipped = np.clip(arr, p5, p95)
        mean = float(np.mean(clipped))
        std = float(np.std(clipped))  # population std

        result[key] = FeatureStats(
            mean=mean,
            std=std,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            valid_count=valid_count,
            unstable=valid_count < 30,
            degenerate=std == 0.0,
        )
    return result


def flatten_corpus_json(corpus_json: dict) -> Dict[str, float | None]:
    """Flatten the nested 'features' block from a corpus JSON file into a flat dict.

    Uses field names without group prefixes:
      {"morphology": {"agreement_error_rate": 0.0625}} → {"agreement_error_rate": 0.0625}

    Non-scalar values (e.g., binyan_distribution) are skipped.
    """
    result: Dict[str, float | None] = {}
    features_block = corpus_json.get("features", {})
    for group_values in features_block.values():
        if not isinstance(group_values, dict):
            continue
        for key, value in group_values.items():
            if isinstance(value, (int, float)) or value is None:
                result[key] = float(value) if isinstance(value, int) else value
    return result


def compute_embedding_stats(
    corpus_sentence_lists: List[List[str]],
    embedder: "Any",
    feature_stats: Dict[str, "FeatureStats"],
) -> Dict[str, "FeatureStats"]:
    """Compute corpus baseline for sentence_cosine_similarity and merge into feature_stats.

    Encodes all sentences in the corpus, computes cosine similarity for every
    adjacent sentence pair within each document, and stores the resulting
    distribution as a FeatureStats entry under the key
    "sentence_cosine_similarity".

    Args:
        corpus_sentence_lists: One list of sentence strings per corpus document.
        embedder: A SentenceEmbedder instance.
        feature_stats: Existing feature stats dict to update in-place.

    Returns:
        The updated feature_stats dict (same object, mutated).
    """
    all_cosines: List[float] = []

    for doc_sentences in corpus_sentence_lists:
        if len(doc_sentences) < 2:
            continue
        embeddings = embedder.embed(doc_sentences)
        for j in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[j - 1], embeddings[j]))
            all_cosines.append(sim)

    if all_cosines:
        arr = np.array(all_cosines, dtype=float)
        valid_count = len(arr)

        # Clip to [p5, p95] for mean/std, same as compute_feature_stats
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        clipped = np.clip(arr, p5, p95)
        mean = float(np.mean(clipped))
        std = float(np.std(clipped))

        feature_stats["sentence_cosine_similarity"] = FeatureStats(
            mean=mean,
            std=std,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p10=float(np.percentile(arr, 10)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            valid_count=valid_count,
            unstable=valid_count < 30,
            degenerate=std == 0.0,
        )

    return feature_stats


def save_stats(
    feature_stats: Dict[str, FeatureStats],
    feature_path: str = "feature_stats.json",
) -> None:
    """Persist feature statistics to a JSON file.

    Each FeatureStats is serialized as a flat dict of its fields.
    """
    data = {
        key: {
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "p10": stats.p10,
            "p25": stats.p25,
            "p50": stats.p50,
            "p75": stats.p75,
            "p90": stats.p90,
            "valid_count": stats.valid_count,
            "unstable": stats.unstable,
            "degenerate": stats.degenerate,
        }
        for key, stats in feature_stats.items()
    }
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_stats(
    feature_path: str = "feature_stats.json",
) -> Dict[str, FeatureStats]:
    """Load feature statistics from a JSON file.

    Returns a dict mapping feature name to FeatureStats.
    Raises FileNotFoundError if the file does not exist.
    """
    with open(feature_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        key: FeatureStats(
            mean=entry["mean"],
            std=entry["std"],
            min=entry["min"],
            max=entry["max"],
            p10=entry["p10"],
            p25=entry["p25"],
            p50=entry["p50"],
            p75=entry["p75"],
            p90=entry["p90"],
            valid_count=entry["valid_count"],
            unstable=entry["unstable"],
            degenerate=entry["degenerate"],
        )
        for key, entry in data.items()
    }
