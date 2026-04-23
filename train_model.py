"""Local training CLI for the ML Distillation Layer (Layer 6).

Parses command-line arguments, builds a :class:`ml.trainer.TrainConfig`,
and delegates to :func:`ml.trainer.train`.  Auto-detects hardware
(CPU / MPS / CUDA) when ``--device`` is not provided.

Requirements implemented: 22.2, 23.1, 23.2, 23.3, 23.4, 23.5, 25.3.
"""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for local training."""
    parser = argparse.ArgumentParser(
        description="Train the ML distillation student model locally.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training JSONL file produced by export_training_data.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model checkpoints.",
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
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string (e.g. cuda, cpu, mps). Auto-detected if not provided.",
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
    """Entry point for local training."""
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

    metrics = train(
        data_path=args.data,
        output_dir=args.output,
        config=config,
        device=args.device,
        resume_from=args.resume,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
