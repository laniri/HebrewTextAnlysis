#!/usr/bin/env python3
"""Run the probabilistic analysis layer on a Hebrew text file.

Usage:
    python run_analysis.py --results-dir results_sample/ --text input.txt
    python run_analysis.py --results-dir results_sample/ --text input.txt --freq-dict freq_dict.json
    python run_analysis.py --results-dir results_sample/ --text input.txt --output analysis.json --pretty
    python run_analysis.py --results-dir results_sample/ --text input.txt --stats-cache feature_stats.json
    python run_analysis.py --results-dir results_sample/ --text input.txt --top-k 10 --yap-url http://localhost:9000/yap/heb/joint

The script:
  1. Loads all profiler result JSON files from --results-dir to compute (or load cached)
     corpus statistics.
  2. Runs the full pipeline once on --text to extract features and per-sentence metrics.
  3. Detects linguistic issues across all 6 groups and ranks the top-K.
  4. Writes the result JSON to --output (or stdout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.analysis_pipeline import run_analysis_pipeline
from analysis.issue_detector import detect_issues
from analysis.issue_ranker import rank_issues
from analysis.serialization import serialize_issues
from analysis.statistics import (
    compute_embedding_stats,
    compute_feature_stats,
    flatten_corpus_json,
    load_stats,
    save_stats,
)
from hebrew_profiler.models import PipelineConfig


# ---------------------------------------------------------------------------
# Corpus statistics helpers
# ---------------------------------------------------------------------------

def _load_results_dir(results_dir: Path) -> list[dict]:
    """Load all *.json files from results_dir."""
    files = sorted(results_dir.glob("*.json"))
    if not files:
        _die(f"No JSON files found in results directory: {results_dir}")
    docs = []
    for f in files:
        try:
            docs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[warn] Skipping {f.name}: {exc}", file=sys.stderr)
    return docs


def _get_feature_stats(results_dir: Path, stats_cache: Path | None) -> dict:
    """Return feature stats, loading from cache or computing from corpus."""
    # Try loading from cache first
    cache_path = stats_cache or results_dir / "feature_stats.json"
    if cache_path.exists():
        print(f"[info] Loading corpus statistics from {cache_path}", file=sys.stderr)
        return load_stats(feature_path=str(cache_path))

    # Compute from corpus
    print(f"[info] Computing corpus statistics from {results_dir} ...", file=sys.stderr)
    docs = _load_results_dir(results_dir)
    feature_dicts = [flatten_corpus_json(doc) for doc in docs]
    feature_stats = compute_feature_stats(feature_dicts)

    # Persist for future runs
    save_stats(feature_stats, feature_path=str(cache_path))
    print(f"[info] Saved corpus statistics to {cache_path}", file=sys.stderr)
    return feature_stats


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the probabilistic analysis layer on a Hebrew text file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-dir", "-r",
        required=True,
        metavar="DIR",
        help="Directory containing profiler result JSON files (used to compute corpus statistics).",
    )
    p.add_argument(
        "--text", "-t",
        required=True,
        metavar="FILE",
        help="Input Hebrew text file to analyse.",
    )
    p.add_argument(
        "--freq-dict", "-f",
        metavar="FILE",
        default=None,
        help="Path to a JSON word-frequency dictionary (enables rare_word_ratio).",
    )
    p.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Write JSON output to this file instead of stdout.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    p.add_argument(
        "--stats-cache",
        metavar="FILE",
        default=None,
        help=(
            "Path to a feature_stats.json cache file. "
            "If it exists, statistics are loaded from it; otherwise they are computed "
            "from --results-dir and saved here. "
            "Defaults to <results-dir>/feature_stats.json."
        ),
    )
    p.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top issues to return (default: 5).",
    )
    p.add_argument(
        "--yap-url",
        default="http://localhost:8000/yap/heb/joint",
        metavar="URL",
        help="YAP API endpoint (default: http://localhost:8000/yap/heb/joint).",
    )
    p.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Use sentence embeddings for cohesion detection instead of Jaccard similarity. "
            "Requires sentence-transformers to be installed. "
            "On first run with a new stats cache, corpus embeddings are computed (~1-2 min). "
            "Subsequent runs load the cached stats and only embed the input document."
        ),
    )
    p.add_argument(
        "--embed-model",
        metavar="MODEL",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformers model name (default: paraphrase-multilingual-mpnet-base-v2).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    text_path = Path(args.text)
    stats_cache = Path(args.stats_cache) if args.stats_cache else None

    # Validate inputs
    if not results_dir.is_dir():
        _die(f"--results-dir is not a directory: {results_dir}")
    if not text_path.is_file():
        _die(f"--text file not found: {text_path}")
    if args.freq_dict and not Path(args.freq_dict).is_file():
        _die(f"--freq-dict file not found: {args.freq_dict}")

    # Step 1: Corpus statistics
    feature_stats = _get_feature_stats(results_dir, stats_cache)

    # Step 1b: Load embedder if requested
    embedder = None
    if args.embed:
        from analysis.embedder import get_embedder
        print(f"[info] Loading embedding model {args.embed_model} ...", file=sys.stderr)
        embedder = get_embedder(args.embed_model)
        if embedder is None:
            print(
                "[warn] sentence-transformers is not installed. "
                "Falling back to Jaccard cohesion. "
                "Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
        else:
            print(f"[info] Embedding model loaded.", file=sys.stderr)
            # Compute corpus embedding stats if not already cached
            if "sentence_cosine_similarity" not in feature_stats:
                print(
                    "[info] Computing corpus sentence embedding statistics "
                    "(one-time, may take 1-2 min) ...",
                    file=sys.stderr,
                )
                docs = _load_results_dir(results_dir)
                from hebrew_profiler.yap_adapter import _SENTENCE_SPLIT_RE
                corpus_sentence_lists = []
                for doc in docs:
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
                cache_path = stats_cache or results_dir / "feature_stats.json"
                save_stats(feature_stats, feature_path=str(cache_path))
                print(
                    f"[info] Corpus embedding stats saved to {cache_path}",
                    file=sys.stderr,
                )

    # Step 2: Run pipeline on input text
    text = text_path.read_text(encoding="utf-8")
    config = PipelineConfig(
        yap_url=args.yap_url,
        freq_dict_path=args.freq_dict,
    )

    print(f"[info] Running pipeline on {text_path} ...", file=sys.stderr)
    try:
        analysis_input = run_analysis_pipeline(text, config, embedder=embedder)
    except Exception as exc:
        _die(f"Pipeline failed: {exc}")

    print(
        f"[info] Pipeline complete — {analysis_input.sentence_count} sentences, "
        f"{sum(1 for v in analysis_input.raw_features.values() if v is not None)} features extracted.",
        file=sys.stderr,
    )

    # Step 3: Detect issues
    issues = detect_issues(
        analysis_input.raw_features,
        analysis_input.sentence_metrics,
        feature_stats,
    )
    print(f"[info] Detected {len(issues)} issues total.", file=sys.stderr)

    # Step 4: Rank and select top-K
    ranked = rank_issues(issues, k=args.top_k)
    print(f"[info] Returning top {len(ranked)} issues.", file=sys.stderr)

    # Step 5: Serialize and attach sentence text to each issue
    raw_json = json.loads(serialize_issues(ranked))
    sentences = analysis_input.sentences

    for issue in raw_json["issues"]:
        span = issue["span"]
        if len(span) == 1:
            # sentence-level: single index
            idx = span[0]
            issue["sentence"] = sentences[idx] if idx < len(sentences) else None
        elif len(span) == 2 and span[1] - span[0] == 1:
            # discourse pair: two adjacent sentences
            i, j = span[0], span[1]
            parts = [sentences[k] for k in (i, j) if k < len(sentences)]
            issue["sentence"] = " / ".join(parts) if parts else None
        else:
            # document-level: no single sentence to attach
            issue["sentence"] = None

    # Add top-level scores and metadata
    raw_json["scores"] = analysis_input.scores
    raw_json["cohesion_method"] = (
        "sentence_embeddings" if embedder is not None else "jaccard"
    )

    # Always pretty-print when writing to a file; honour --pretty for stdout too
    indent = 2 if (args.output or args.pretty) else None
    output_str = json.dumps(raw_json, ensure_ascii=False, indent=indent)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"[info] Output written to {out_path}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
