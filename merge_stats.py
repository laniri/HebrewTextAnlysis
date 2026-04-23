#!/usr/bin/env python3
"""Merge corpus statistics from multiple pipeline result directories.

Loads all result JSON files from one or more directories, computes unified
feature statistics, and optionally computes sentence embedding statistics.
Produces a single feature_stats.json that reflects the combined corpus.

This is useful for enriching the baseline beyond a single corpus source
(e.g., combining Wikipedia results with HeDC4 web crawl results).

Usage:
    # Merge two result directories
    python merge_stats.py --results-dirs results_sample/ results_hedc4/ \
        --output feature_stats_merged.json

    # Include sentence embedding statistics
    python merge_stats.py --results-dirs results_sample/ results_hedc4/ \
        --output feature_stats_merged.json --embed

    # Custom embedding model
    python merge_stats.py --results-dirs results_sample/ results_hedc4/ \
        --output feature_stats_merged.json --embed \
        --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.statistics import (
    compute_embedding_stats,
    compute_feature_stats,
    flatten_corpus_json,
    save_stats,
)
from hebrew_profiler.yap_adapter import _SENTENCE_SPLIT_RE


def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def _load_results_dir(results_dir: Path) -> list[dict]:
    """Load all *.json files from a results directory."""
    files = sorted(results_dir.glob("*.json"))
    # Exclude feature_stats.json itself
    files = [f for f in files if f.name != "feature_stats.json"]
    docs = []
    for f in files:
        try:
            docs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[warn] Skipping {f}: {exc}", file=sys.stderr)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge corpus statistics from multiple pipeline result directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dirs", "-r",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more directories containing pipeline result JSON files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="FILE",
        help="Output path for the merged feature_stats.json.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also compute sentence embedding statistics (requires sentence-transformers).",
    )
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        metavar="MODEL",
        help="Sentence-transformers model name (default: paraphrase-multilingual-mpnet-base-v2).",
    )

    args = parser.parse_args()

    # Load all documents from all directories
    all_docs: list[dict] = []
    for dir_path in args.results_dirs:
        d = Path(dir_path)
        if not d.is_dir():
            _die(f"Not a directory: {d}")
        docs = _load_results_dir(d)
        print(f"[info] Loaded {len(docs)} documents from {d}", file=sys.stderr)
        all_docs.extend(docs)

    if not all_docs:
        _die("No documents loaded from any directory.")

    print(f"[info] Total documents: {len(all_docs)}", file=sys.stderr)

    # Compute feature stats
    feature_dicts = [flatten_corpus_json(doc) for doc in all_docs]
    feature_stats = compute_feature_stats(feature_dicts)
    print(f"[info] Computed stats for {len(feature_stats)} features.", file=sys.stderr)

    # Optionally compute embedding stats
    if args.embed:
        from analysis.embedder import get_embedder

        print(f"[info] Loading embedding model {args.embed_model} ...", file=sys.stderr)
        embedder = get_embedder(args.embed_model)
        if embedder is None:
            print(
                "[warn] sentence-transformers not available — skipping embedding stats.",
                file=sys.stderr,
            )
        else:
            print(
                f"[info] Computing sentence embedding statistics over {len(all_docs)} documents ...",
                file=sys.stderr,
            )
            corpus_sentence_lists: list[list[str]] = []
            for doc in all_docs:
                text_raw = doc.get("text", "")
                sents = [
                    s.strip()
                    for s in _SENTENCE_SPLIT_RE.split(text_raw.strip())
                    if s.strip()
                ]
                if sents:
                    corpus_sentence_lists.append(sents)

            feature_stats = compute_embedding_stats(
                corpus_sentence_lists, embedder, feature_stats
            )
            print("[info] Embedding statistics computed.", file=sys.stderr)

    # Save
    save_stats(feature_stats, feature_path=args.output)
    print(f"[info] Merged statistics saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
