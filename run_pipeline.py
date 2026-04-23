#!/usr/bin/env python3
"""CLI entry point for the Hebrew Linguistic Profiling Engine.

Single document:
    python run_pipeline.py single --input file.txt --output result.json --pretty --yap-url URL

Batch mode:
    python run_pipeline.py batch --input corpus/ --output results/ --workers 4 --jsonl dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from hebrew_profiler.batch import process_batch
from hebrew_profiler.models import PipelineConfig
from hebrew_profiler.pipeline import pipeline_output_to_json, process_document


DEFAULT_YAP_URL = "http://localhost:8000/yap/heb/joint"


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Hebrew Linguistic Profiling Engine — analyse Hebrew text "
        "and produce structured JSON with linguistic features and scores.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- single-document sub-command ---
    single = subparsers.add_parser(
        "single",
        help="Process a single Hebrew text file.",
    )
    single.add_argument(
        "--input", required=True, help="Path to input Hebrew text file (UTF-8)."
    )
    single.add_argument(
        "--output",
        default=None,
        help="Path to output JSON file. Defaults to stdout.",
    )
    single.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output (indent=2)."
    )
    single.add_argument(
        "--yap-url",
        default=DEFAULT_YAP_URL,
        help=f"YAP API endpoint URL (default: {DEFAULT_YAP_URL}).",
    )
    single.add_argument(
        "--freq-dict",
        default=None,
        metavar="PATH",
        help="Path to JSON frequency dictionary for rare_word_ratio computation.",
    )
    single.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Run the probabilistic analysis layer after the pipeline and include "
            "ranked issue diagnostics in the output JSON. Requires --stats-cache or "
            "a results directory with a pre-built feature_stats.json."
        ),
    )
    single.add_argument(
        "--stats-cache",
        default=None,
        metavar="PATH",
        help=(
            "Path to feature_stats.json for the analysis layer (required when --analyze is set). "
            "Typically the feature_stats.json produced by: "
            "run_pipeline.py batch ... --build-stats"
        ),
    )
    single.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of top issues to return when --analyze is set (default: 5).",
    )
    single.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Use sentence embeddings for cohesion detection when --analyze is set. "
            "Requires sentence-transformers and a stats cache that includes "
            "sentence_cosine_similarity (built with --build-stats --embed)."
        ),
    )
    single.add_argument(
        "--embed-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        metavar="MODEL",
        help="Sentence-transformers model name for --embed (default: paraphrase-multilingual-mpnet-base-v2).",
    )

    # --- batch sub-command ---
    batch = subparsers.add_parser(
        "batch",
        help="Process a directory of Hebrew text files in batch mode.",
    )
    batch.add_argument(
        "--input", required=True, help="Path to input directory containing .txt files."
    )
    batch.add_argument(
        "--output", required=True, help="Path to output directory for JSON results."
    )
    batch.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    batch.add_argument(
        "--jsonl",
        default=None,
        help="Path to JSONL export file (optional).",
    )
    batch.add_argument(
        "--pretty", action="store_true", help="Pretty-print individual JSON results."
    )
    batch.add_argument(
        "--yap-url",
        default=DEFAULT_YAP_URL,
        help=f"YAP API endpoint URL (default: {DEFAULT_YAP_URL}).",
    )
    batch.add_argument(
        "--freq-dict",
        default=None,
        metavar="PATH",
        help="Path to JSON frequency dictionary for rare_word_ratio computation.",
    )
    batch.add_argument(
        "--build-stats",
        action="store_true",
        help=(
            "After batch processing, compute analysis layer corpus statistics "
            "from the output results and save to feature_stats.json in the output directory. "
            "Pass --embed to also compute sentence embedding statistics (requires sentence-transformers)."
        ),
    )
    batch.add_argument(
        "--embed",
        action="store_true",
        help=(
            "When used with --build-stats, also compute sentence embedding statistics "
            "for semantic cohesion detection. Requires sentence-transformers."
        ),
    )
    batch.add_argument(
        "--embed-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        metavar="MODEL",
        help="Sentence-transformers model name for --embed (default: paraphrase-multilingual-mpnet-base-v2).",
    )
    batch.add_argument(
        "--manage-yap",
        action="store_true",
        help=(
            "Let the pipeline manage the YAP server process. "
            "If YAP crashes during batch processing, it will be automatically restarted. "
            "Requires the 'yap' binary to be in PATH, or set YAP_BIN env var."
        ),
    )
    batch.add_argument(
        "--yap-bin",
        default=None,
        metavar="PATH",
        help="Path to the YAP binary (default: 'yap' from PATH, or YAP_BIN env var).",
    )
    batch.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort batch processing if YAP is unresponsive (instead of producing "
            "results with null syntax features). Logs detailed error info and exits."
        ),
    )

    return parser


def run_single(args: argparse.Namespace) -> int:
    """Execute single-document mode. Returns exit code."""
    try:
        with open(args.input, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Error reading input file: {exc}", file=sys.stderr)
        return 1

    config = PipelineConfig(
        yap_url=args.yap_url,
        pretty_output=args.pretty,
        freq_dict_path=args.freq_dict,
    )

    # --- Analysis layer path ---
    if getattr(args, "analyze", False):
        return _run_single_with_analysis(text, args, config)

    # --- Pipeline-only path ---
    try:
        output = process_document(text, config)
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    json_str = pipeline_output_to_json(output, pretty=args.pretty)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(json_str + "\n")
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            return 1
    else:
        print(json_str)

    return 0


def _run_single_with_analysis(
    text: str, args: argparse.Namespace, config: PipelineConfig
) -> int:
    """Run pipeline + analysis layer for a single document."""
    from pathlib import Path

    from analysis.analysis_pipeline import run_analysis_pipeline
    from analysis.issue_detector import detect_issues
    from analysis.issue_ranker import rank_issues
    from analysis.serialization import serialize_issues
    from analysis.statistics import load_stats

    # Resolve stats cache
    stats_path = getattr(args, "stats_cache", None)
    if not stats_path:
        print(
            "Error: --analyze requires --stats-cache pointing to a feature_stats.json file.\n"
            "Build one first with: python run_pipeline.py batch --input corpus/ --output results/ --build-stats",
            file=sys.stderr,
        )
        return 1

    if not Path(stats_path).is_file():
        print(f"Error: stats cache not found: {stats_path}", file=sys.stderr)
        return 1

    try:
        feature_stats = load_stats(feature_path=stats_path)
    except Exception as exc:
        print(f"Error loading stats cache: {exc}", file=sys.stderr)
        return 1

    # Load embedder if requested
    embedder = None
    if getattr(args, "embed", False):
        from analysis.embedder import get_embedder
        embedder = get_embedder(getattr(args, "embed_model",
                                        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
        if embedder is None:
            print(
                "[warn] sentence-transformers not available — falling back to Jaccard cohesion.",
                file=sys.stderr,
            )

    # Run pipeline + analysis
    try:
        analysis_input = run_analysis_pipeline(text, config, embedder=embedder)
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    issues = detect_issues(
        analysis_input.raw_features,
        analysis_input.sentence_metrics,
        feature_stats,
    )
    top_k = getattr(args, "top_k", 5)
    ranked = rank_issues(issues, k=top_k)

    # Build output: pipeline JSON merged with analysis results
    raw_analysis = json.loads(serialize_issues(ranked))
    sentences = analysis_input.sentences

    for issue in raw_analysis["issues"]:
        span = issue["span"]
        if len(span) == 1:
            idx = span[0]
            issue["sentence"] = sentences[idx] if idx < len(sentences) else None
        elif len(span) == 2 and span[1] - span[0] == 1:
            parts = [sentences[k] for k in span if k < len(sentences)]
            issue["sentence"] = " / ".join(parts) if parts else None
        else:
            issue["sentence"] = None

    raw_analysis["scores"] = analysis_input.scores
    raw_analysis["cohesion_method"] = (
        "sentence_embeddings" if embedder is not None else "jaccard"
    )

    # Layer 4–5: Diagnosis + Intervention mapping
    from analysis.interpretation import run_interpretation
    from analysis.serialization import serialize_interpretation

    interpretation = run_interpretation(issues, analysis_input.scores)
    interp_json = json.loads(serialize_interpretation(interpretation))
    raw_analysis["diagnoses"] = interp_json["diagnoses"]
    raw_analysis["interventions"] = interp_json["interventions"]

    indent = 2 if (args.output or args.pretty) else None
    output_str = json.dumps(raw_analysis, ensure_ascii=False, indent=indent)

    if args.output:
        try:
            Path(args.output).write_text(output_str, encoding="utf-8")
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            return 1
    else:
        print(output_str)

    return 0


def run_batch(args: argparse.Namespace) -> int:
    """Execute batch mode. Returns exit code."""
    # Set up YAP server manager if requested
    if getattr(args, "manage_yap", False):
        from hebrew_profiler.yap_adapter import YAPServerManager, set_yap_manager
        port = int(args.yap_url.split(":")[2].split("/")[0]) if ":" in args.yap_url else 8000
        manager = YAPServerManager(
            yap_bin=getattr(args, "yap_bin", None),
            port=port,
        )
        set_yap_manager(manager)
        # Ensure YAP is running before starting the batch
        if not manager.is_alive():
            if not manager.start():
                print("[error] Could not start YAP server.", file=sys.stderr)
                return 1

    config = PipelineConfig(
        yap_url=args.yap_url,
        pretty_output=args.pretty,
        workers=args.workers,
        freq_dict_path=args.freq_dict,
    )

    result = process_batch(
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        workers=args.workers,
        jsonl_path=args.jsonl,
        strict=getattr(args, "strict", False),
    )

    print(
        json.dumps(
            {
                "total_processed": result.total_processed,
                "error_count": result.error_count,
                "errors": result.errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if getattr(args, "build_stats", False):
        _build_analysis_stats(args)

    return 0


def _build_analysis_stats(args: argparse.Namespace) -> None:
    """Compute and save analysis layer corpus statistics from batch output."""
    import glob
    from pathlib import Path

    from analysis.statistics import (
        compute_embedding_stats,
        compute_feature_stats,
        flatten_corpus_json,
        save_stats,
    )
    from hebrew_profiler.yap_adapter import _SENTENCE_SPLIT_RE

    output_dir = Path(args.output)
    stats_path = output_dir / "feature_stats.json"

    # Load all result JSON files from the output directory
    result_files = sorted(output_dir.glob("*.json"))
    if not result_files:
        print("[stats] No JSON result files found in output directory.", file=sys.stderr)
        return

    print(f"[stats] Loading {len(result_files)} result files ...", file=sys.stderr)
    docs = []
    for f in result_files:
        try:
            docs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[stats] Skipping {f.name}: {exc}", file=sys.stderr)

    if not docs:
        print("[stats] No valid documents loaded.", file=sys.stderr)
        return

    # Compute feature stats from raw feature values
    feature_dicts = [flatten_corpus_json(doc) for doc in docs]
    feature_stats = compute_feature_stats(feature_dicts)
    print(f"[stats] Computed stats for {len(feature_stats)} features.", file=sys.stderr)

    # Optionally compute sentence embedding stats
    if getattr(args, "embed", False):
        from analysis.embedder import get_embedder
        print(
            f"[stats] Loading embedding model {args.embed_model} ...",
            file=sys.stderr,
        )
        embedder = get_embedder(args.embed_model)
        if embedder is None:
            print(
                "[stats] sentence-transformers not available — skipping embedding stats. "
                "Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
        else:
            print(
                "[stats] Computing sentence embedding statistics (may take 1-2 min) ...",
                file=sys.stderr,
            )
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
            print("[stats] Embedding statistics computed.", file=sys.stderr)

    save_stats(feature_stats, feature_path=str(stats_path))
    print(f"[stats] Corpus statistics saved to {stats_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate sub-command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "single":
        return run_single(args)
    elif args.command == "batch":
        return run_batch(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
