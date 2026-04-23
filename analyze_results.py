#!/usr/bin/env python3
"""Analyze batch pipeline results for outliers and normalization boundary fitness.

Reads all JSON files from a results directory, computes statistics for every
numeric feature and score, flags outliers, and checks whether the current
min-max normalization ranges are appropriate for the observed data.
"""

import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, median, stdev


def load_results(results_dir: str) -> list[tuple[str, dict]]:
    """Load all JSON result files. Returns list of (filename, data)."""
    results = []
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            results.append((f.name, json.load(fh)))
    return results


def extract_feature_values(results: list[tuple[str, dict]]) -> dict[str, list[tuple[str, float]]]:
    """Extract all numeric feature values, keyed by feature path.
    Returns {feature_path: [(filename, value), ...]}."""
    features = {}

    for fname, data in results:
        # Morphology
        morph = data.get("features", {}).get("morphology", {})
        for key in ["verb_ratio", "prefix_density", "suffix_pronoun_ratio", "morphological_ambiguity",
                     "agreement_error_rate", "binyan_entropy", "construct_ratio"]:
            val = morph.get(key)
            if val is not None:
                features.setdefault(f"morph.{key}", []).append((fname, val))

        # Syntax
        syn = data.get("features", {}).get("syntax", {})
        for key in ["avg_sentence_length", "avg_tree_depth", "max_tree_depth",
                     "avg_dependency_distance", "clauses_per_sentence",
                     "subordinate_clause_ratio", "right_branching_ratio",
                     "dependency_distance_variance", "clause_type_entropy"]:
            val = syn.get(key)
            if val is not None:
                features.setdefault(f"syntax.{key}", []).append((fname, val))

        # Lexicon
        lex = data.get("features", {}).get("lexicon", {})
        for key in ["type_token_ratio", "hapax_ratio", "avg_token_length", "lemma_diversity",
                     "rare_word_ratio", "content_word_ratio"]:
            val = lex.get(key)
            if val is not None:
                features.setdefault(f"lexicon.{key}", []).append((fname, val))

        # Structure
        struct = data.get("features", {}).get("structure", {})
        for key in ["sentence_length_variance", "long_sentence_ratio",
                     "punctuation_ratio", "short_sentence_ratio",
                     "missing_terminal_punctuation_ratio"]:
            val = struct.get(key)
            if val is not None:
                features.setdefault(f"struct.{key}", []).append((fname, val))

        # Discourse
        disc = data.get("features", {}).get("discourse", {})
        for key in ["connective_ratio", "sentence_overlap", "pronoun_to_noun_ratio"]:
            val = disc.get(key)
            if val is not None:
                features.setdefault(f"discourse.{key}", []).append((fname, val))

        # Style
        style = data.get("features", {}).get("style", {})
        for key in ["sentence_length_trend", "pos_distribution_variance"]:
            val = style.get(key)
            if val is not None:
                features.setdefault(f"style.{key}", []).append((fname, val))

        # Scores
        scores = data.get("scores", {})
        for key in ["difficulty", "style", "fluency", "cohesion", "complexity"]:
            val = scores.get(key)
            if val is not None:
                features.setdefault(f"scores.{key}", []).append((fname, val))

    return features


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4] if n >= 4 else sorted_vals[0]
    q3 = sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1]
    iqr = q3 - q1
    return {
        "count": n,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if n > 1 else 0.0,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": q1 - 1.5 * iqr,
        "upper_fence": q3 + 1.5 * iqr,
    }


def find_outliers(entries: list[tuple[str, float]], stats: dict) -> list[tuple[str, float, str]]:
    """Find values outside 1.5×IQR fences. Returns [(filename, value, direction), ...]."""
    outliers = []
    for fname, val in entries:
        if val < stats["lower_fence"]:
            outliers.append((fname, val, "LOW"))
        elif val > stats["upper_fence"]:
            outliers.append((fname, val, "HIGH"))
    return outliers


# Current normalization ranges from config.py
NORM_RANGES = {
    "syntax.avg_sentence_length": (10.0, 40.0),
    "syntax.avg_tree_depth": (4.0, 15.0),
    "lexicon.hapax_ratio": (0.15, 0.55),
    "morph.morphological_ambiguity": (4.0, 10.0),
    "morph.suffix_pronoun_ratio": (0.05, 0.50),
    "struct.sentence_length_variance": (0.0, 400.0),
    "style.sentence_length_trend": (-1.5, 1.5),
    "style.pos_distribution_variance": (0.0, 0.008),
    "discourse.pronoun_to_noun_ratio": (0.0, 0.45),
    "lexicon.rare_word_ratio": (0.0, 0.3),
    "lexicon.content_word_ratio": (0.1, 0.8),
    "discourse.connective_ratio": (0.0, 1.2),
    "discourse.sentence_overlap": (0.0, 0.4),
    "morph.agreement_error_rate": (0.0, 0.3),
    "syntax.dependency_distance_variance": (0.0, 27.0),
    "syntax.clause_type_entropy": (2.0, 3.0),
}


def check_normalization_ranges(features: dict[str, list[tuple[str, float]]]) -> None:
    """Check if current normalization ranges fit the observed data."""
    print("\n" + "=" * 70)
    print("NORMALIZATION RANGE ANALYSIS")
    print("=" * 70)

    for feat_path, (cfg_min, cfg_max) in NORM_RANGES.items():
        if feat_path not in features:
            print(f"\n  {feat_path}: NO DATA")
            continue

        values = [v for _, v in features[feat_path]]
        stats = compute_stats(values)

        # What percentage of values fall outside the configured range?
        below = sum(1 for v in values if v < cfg_min)
        above = sum(1 for v in values if v > cfg_max)
        in_range = len(values) - below - above
        pct_in = 100.0 * in_range / len(values)

        # What percentage gets clamped to 0 or 1?
        pct_clamped_low = 100.0 * below / len(values)
        pct_clamped_high = 100.0 * above / len(values)

        print(f"\n  {feat_path}")
        print(f"    Config range:    [{cfg_min}, {cfg_max}]")
        print(f"    Observed range:  [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Observed mean:   {stats['mean']:.3f}  median: {stats['median']:.3f}")
        print(f"    In range:        {in_range}/{len(values)} ({pct_in:.1f}%)")
        print(f"    Clamped to 0.0:  {below} ({pct_clamped_low:.1f}%)")
        print(f"    Clamped to 1.0:  {above} ({pct_clamped_high:.1f}%)")

        # Suggest new range if >20% of values are clamped
        if pct_clamped_low > 20 or pct_clamped_high > 20:
            suggested_min = max(0, stats["q1"] - 0.5 * stats["iqr"])
            suggested_max = stats["q3"] + 0.5 * stats["iqr"]
            print(f"    ⚠️  SUGGESTION: Consider [{suggested_min:.1f}, {suggested_max:.1f}]")
        elif pct_in == 100:
            # Range might be too wide — check if data clusters in a narrow band
            data_span = stats["max"] - stats["min"]
            config_span = cfg_max - cfg_min
            if data_span < 0.3 * config_span:
                print(f"    ℹ️  Data uses only {100*data_span/config_span:.0f}% of the range — consider tightening")
        else:
            print(f"    ✅ Range looks reasonable")


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two lists.

    Returns None if fewer than 3 paired values or zero variance.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _align_by_filename(
    a_entries: list[tuple[str, float]],
    b_entries: list[tuple[str, float]],
) -> tuple[list[float], list[float]]:
    """Align two feature entry lists by filename, returning paired value lists."""
    b_map = {fname: val for fname, val in b_entries}
    xs, ys = [], []
    for fname, val in a_entries:
        if fname in b_map:
            xs.append(val)
            ys.append(b_map[fname])
    return xs, ys


def score_independence_analysis(
    features: dict[str, list[tuple[str, float]]],
) -> None:
    """Analyze whether scores measure different aspects of the text.

    Prints:
    1. Pairwise correlation matrix between all scores
    2. Top feature correlations for each score (what drives each score)
    3. Redundancy warnings for highly correlated score pairs
    """
    score_keys = [k for k in ["scores.difficulty", "scores.style", "scores.fluency",
                               "scores.cohesion", "scores.complexity"] if k in features]
    feature_keys = [k for k in sorted(features.keys()) if not k.startswith("scores.")]

    if len(score_keys) < 2:
        return

    print("\n" + "=" * 70)
    print("SCORE INDEPENDENCE ANALYSIS")
    print("=" * 70)

    # --- 1. Pairwise score correlation matrix ---
    print("\n  Pairwise Pearson correlations between scores:")
    short_names = {k: k.replace("scores.", "") for k in score_keys}

    # Header
    header = "  " + " " * 14
    for k in score_keys:
        header += f"{short_names[k]:>12s}"
    print(header)

    redundant_pairs: list[tuple[str, str, float]] = []

    for ka in score_keys:
        row = f"  {short_names[ka]:>12s}"
        for kb in score_keys:
            if ka == kb:
                row += "        1.00"
                continue
            xs, ys = _align_by_filename(features[ka], features[kb])
            r = pearson_r(xs, ys)
            if r is not None:
                marker = " ⚠️" if abs(r) > 0.7 else ""
                row += f"  {r:>9.3f}{marker}"
                if abs(r) > 0.7 and ka < kb:
                    redundant_pairs.append((short_names[ka], short_names[kb], r))
            else:
                row += "         n/a"
        print(row)

    # Interpretation guide
    print("\n  Interpretation: |r| < 0.3 = independent, 0.3–0.7 = moderate, > 0.7 = redundant ⚠️")

    if redundant_pairs:
        print("\n  ⚠️  Redundant score pairs (|r| > 0.7):")
        for a, b, r in redundant_pairs:
            print(f"    {a} ↔ {b}: r={r:.3f} — these may be measuring the same thing")
    else:
        print("\n  ✅ No redundant score pairs detected — each score captures a different aspect")

    # --- 2. Top feature correlations per score ---
    print("\n  Top feature drivers per score (|r| > 0.3):")

    for sk in score_keys:
        correlations: list[tuple[str, float]] = []
        for fk in feature_keys:
            xs, ys = _align_by_filename(features[sk], features[fk])
            r = pearson_r(xs, ys)
            if r is not None:
                correlations.append((fk, r))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top = [(name, r) for name, r in correlations if abs(r) > 0.3]

        print(f"\n    {short_names[sk]}:")
        if top:
            for name, r in top[:8]:
                direction = "+" if r > 0 else "−"
                print(f"      {direction} {name:<45s} r={r:+.3f}")
        else:
            print(f"      (no features with |r| > 0.3)")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_sample"

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files.\n")

    features = extract_feature_values(results)

    # Check for null features (missing layers)
    null_counts = {}
    for fname, data in results:
        for category in ["morphology", "syntax", "lexicon", "structure", "discourse", "style"]:
            cat_data = data.get("features", {}).get(category, {})
            for key, val in cat_data.items():
                path = f"{category}.{key}"
                if val is None:
                    null_counts[path] = null_counts.get(path, 0) + 1

    if null_counts:
        print("NULL FEATURES (missing upstream data):")
        for path, count in sorted(null_counts.items()):
            print(f"  {path}: {count}/{len(results)} null")
        print()

    # Per-feature statistics and outliers
    print("=" * 70)
    print("FEATURE STATISTICS AND OUTLIERS")
    print("=" * 70)

    total_outlier_count = 0
    for feat_path in sorted(features.keys()):
        entries = features[feat_path]
        values = [v for _, v in entries]
        stats = compute_stats(values)
        outliers = find_outliers(entries, stats)
        total_outlier_count += len(outliers)

        flag = " ⚠️" if outliers else ""
        print(f"\n  {feat_path}{flag}")
        print(f"    n={stats['count']}  min={stats['min']:.4f}  max={stats['max']:.4f}  "
              f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  stdev={stats['stdev']:.4f}")
        print(f"    Q1={stats['q1']:.4f}  Q3={stats['q3']:.4f}  IQR={stats['iqr']:.4f}  "
              f"fences=[{stats['lower_fence']:.4f}, {stats['upper_fence']:.4f}]")

        if outliers:
            print(f"    Outliers ({len(outliers)}):")
            for fname, val, direction in sorted(outliers, key=lambda x: x[1]):
                print(f"      {direction:4s}  {val:10.4f}  {fname}")

    print(f"\n  Total outliers across all features: {total_outlier_count}")

    # Normalization range check
    check_normalization_ranges(features)

    # Score distribution summary
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION SUMMARY")
    print("=" * 70)
    for score_key in ["scores.difficulty", "scores.style", "scores.fluency", "scores.cohesion", "scores.complexity"]:
        if score_key in features:
            values = [v for _, v in features[score_key]]
            stats = compute_stats(values)
            print(f"\n  {score_key}")
            print(f"    min={stats['min']:.4f}  max={stats['max']:.4f}  "
                  f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  stdev={stats['stdev']:.4f}")

            # Check if difficulty is well-distributed across [0,1]
            if score_key == "scores.difficulty":
                buckets = [0] * 10
                for v in values:
                    idx = min(int(v * 10), 9)
                    buckets[idx] += 1
                print(f"    Distribution across [0,1] in 10 buckets:")
                for i, count in enumerate(buckets):
                    bar = "█" * count
                    print(f"      [{i*0.1:.1f}-{(i+1)*0.1:.1f}): {count:3d} {bar}")

    # Score independence analysis
    score_independence_analysis(features)

    print()


if __name__ == "__main__":
    main()
