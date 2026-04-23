"""SageMaker training entry point for the ML Distillation Layer (Layer 6).

Reads SageMaker environment variables (``SM_CHANNEL_TRAINING``,
``SM_MODEL_DIR``, ``SM_OUTPUT_DATA_DIR``, ``SM_NUM_GPUS``), maps them
to :class:`ml.trainer.TrainConfig`, and delegates to
:func:`ml.trainer.train`.

Hyperparameters are passed by SageMaker as ``--key value`` CLI arguments.

Requirements implemented: 24.1, 24.2, 24.3, 24.7, 25.4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for SageMaker hyperparameters."""
    parser = argparse.ArgumentParser(
        description="SageMaker entry point for ML distillation training.",
    )
    parser.add_argument(
        "--encoder",
        default="dicta-il/dictabert",
        help="HuggingFace encoder model name (default: dicta-il/dictabert).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=2e-5,
        help="Learning rate for encoder parameters (default: 2e-5).",
    )
    parser.add_argument(
        "--heads-lr",
        type=float,
        default=1e-3,
        help="Learning rate for head parameters (default: 1e-3).",
    )
    parser.add_argument(
        "--sentence-heads-lr",
        type=float,
        default=5e-3,
        help="Learning rate for sentence-level head parameters (default: 5e-3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps for linear warmup (default: 0.1).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum token sequence length (default: 512).",
    )
    parser.add_argument(
        "--loss-weights",
        default="1.0,1.5,2.0",
        help="Comma-separated loss weights for scores,issues,diagnoses (default: 1.0,1.5,2.0).",
    )
    parser.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        help="Enable learned uncertainty weighting (Kendall et al. 2018).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser


def parse_loss_weights(raw: str) -> tuple[float, float, float]:
    """Parse a comma-separated string into a 3-tuple of floats."""
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--loss-weights requires exactly 3 comma-separated values, got {len(parts)}"
        )
    return (parts[0], parts[1], parts[2])


def main(argv: list[str] | None = None) -> None:
    """Entry point for SageMaker training."""
    # Read SageMaker environment variables
    data_path = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    # Determine device from GPU count
    device: str | None = None
    if num_gpus > 0:
        device = "cuda"

    parser = build_parser()
    args = parser.parse_args(argv)

    from ml.trainer import TrainConfig, train

    config = TrainConfig(
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        heads_lr=args.heads_lr,
        sentence_heads_lr=getattr(args, "sentence_heads_lr", 5e-3),
        epochs=args.epochs,
        warmup_fraction=args.warmup_fraction,
        max_seq_length=args.max_seq_length,
        loss_weights=parse_loss_weights(args.loss_weights),
        use_uncertainty_weighting=args.uncertainty_weighting,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Find the JSONL file in the training channel directory
    data_dir = data_path
    jsonl_file = data_dir
    if os.path.isdir(data_dir):
        jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
        if jsonl_files:
            jsonl_file = os.path.join(data_dir, jsonl_files[0])
        else:
            print(
                f"ERROR: No .jsonl files found in {data_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"[sagemaker_train] data_path={jsonl_file}", file=sys.stderr)
    print(f"[sagemaker_train] output_dir={output_dir}", file=sys.stderr)
    print(f"[sagemaker_train] device={device}", file=sys.stderr)

    metrics = train(
        data_path=jsonl_file,
        output_dir=output_dir,
        config=config,
        device=device,
    )

    # Write metrics to output data dir for SageMaker
    os.makedirs(output_data_dir, exist_ok=True)
    metrics_path = os.path.join(output_data_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"[sagemaker_train] Training complete. Metrics written to {metrics_path}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
