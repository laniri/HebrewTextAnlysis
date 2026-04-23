"""CLI for data export, inference, and disagreement mining (Layer 6).

Provides four subcommands:

- ``export``  — Convert pipeline JSONs to training JSONL.
- ``infer``   — Run inference on text using a trained model.
- ``disagree`` — Compare model predictions against pipeline labels.
- ``merge``   — Merge disagreement records into training data.

Requirements implemented: 22.1, 22.3, 22.4.
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="ML Distillation Layer — data export, inference, and disagreement mining.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- export subcommand ---
    export_parser = subparsers.add_parser(
        "export",
        help="Export pipeline JSONs to training JSONL.",
    )
    export_parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing pipeline output JSON files.",
    )
    export_parser.add_argument(
        "--stats-path",
        required=True,
        help="Path to corpus feature statistics JSON (e.g. feature_stats_merged.json).",
    )
    export_parser.add_argument(
        "--output",
        required=True,
        help="Output path for training JSONL file.",
    )
    export_parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional path to write label distribution statistics JSON.",
    )
    export_parser.add_argument(
        "--embed",
        action="store_true",
        help="Use sentence embeddings for cohesion detection (requires sentence-transformers).",
    )
    export_parser.add_argument(
        "--embed-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformers model name for --embed.",
    )
    export_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of threads for parallel JSON reading and analysis (default: 1).",
    )

    # --- infer subcommand ---
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on text using a trained model.",
    )
    infer_group = infer_parser.add_mutually_exclusive_group(required=True)
    infer_group.add_argument(
        "--text",
        help="Raw Hebrew text to analyse.",
    )
    infer_group.add_argument(
        "--input",
        help="Path to a text file to analyse.",
    )
    infer_parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model checkpoint directory.",
    )
    infer_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid mode with pipeline fallback.",
    )
    infer_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for hybrid mode (default: 0.7).",
    )
    infer_parser.add_argument(
        "--device",
        default=None,
        help="Device string (e.g. cuda, cpu). Auto-detected if not provided.",
    )
    infer_parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. Prints to stdout if not provided.",
    )
    infer_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )

    # --- disagree subcommand ---
    disagree_parser = subparsers.add_parser(
        "disagree",
        help="Compare model predictions against pipeline labels.",
    )
    disagree_parser.add_argument(
        "--predictions",
        required=True,
        help="Path to JSONL file with model predictions.",
    )
    disagree_parser.add_argument(
        "--labels",
        required=True,
        help="Path to JSONL file with pipeline labels.",
    )
    disagree_parser.add_argument(
        "--output",
        required=True,
        help="Output path for disagreement JSONL.",
    )
    disagree_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Severity difference threshold for flagging (default: 0.3).",
    )

    # --- merge subcommand ---
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge disagreement records into training data.",
    )
    merge_parser.add_argument(
        "--base",
        required=True,
        help="Path to base training JSONL file.",
    )
    merge_parser.add_argument(
        "--disagreements",
        required=True,
        help="Path to disagreement JSONL file.",
    )
    merge_parser.add_argument(
        "--output",
        required=True,
        help="Output path for merged training JSONL.",
    )

    return parser


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand."""
    from ml.export import export_training_data

    export_training_data(
        input_dirs=args.input_dirs,
        stats_path=args.stats_path,
        output_path=args.output,
        stats_output_path=args.stats_output,
        embed=getattr(args, "embed", False),
        embed_model=getattr(args, "embed_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        workers=getattr(args, "workers", 1),
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """Handle the 'infer' subcommand."""
    from ml.inference import predict, predict_hybrid, serialize_prediction

    # Get text from --text or --input
    if args.text is not None:
        text = args.text
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()

    if args.hybrid:
        result = predict_hybrid(
            text=text,
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
    else:
        result = predict(
            text=text,
            model_path=args.model_path,
            device=args.device,
        )

    indent = 2 if args.pretty else None
    output_str = json.dumps(result, ensure_ascii=False, indent=indent)

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
    else:
        print(output_str)


def cmd_disagree(args: argparse.Namespace) -> None:
    """Handle the 'disagree' subcommand."""
    from ml.disagreement import find_disagreements

    summary = find_disagreements(
        predictions_path=args.predictions,
        labels_path=args.labels,
        output_path=args.output,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_merge(args: argparse.Namespace) -> None:
    """Handle the 'merge' subcommand."""
    from ml.disagreement import merge_training_data

    summary = merge_training_data(
        base_path=args.base,
        disagreements_path=args.disagreements,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "export": cmd_export,
        "infer": cmd_infer,
        "disagree": cmd_disagree,
        "merge": cmd_merge,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
