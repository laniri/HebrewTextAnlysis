"""SageMaker job launcher for the ML Distillation Layer (Layer 6).

Configures and submits a SageMaker training job using the PyTorch
framework estimator.  Uploads local training data to S3 if a local
path is provided.

Defaults to eu-west-1 region.  Supports AWS SSO authentication via
``--profile`` (uses ``boto3.Session(profile_name=...)``).

Requirements implemented: 24.4, 24.5, 24.6.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

_DEFAULT_REGION = "eu-west-1"
_DEFAULT_BUCKET = "hebrew-profiler-ml-training"


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the SageMaker launcher."""
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker training job for ML distillation.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Local path or S3 URI to training JSONL file.",
    )
    parser.add_argument(
        "--output-s3",
        default=f"s3://{_DEFAULT_BUCKET}/output/",
        help=f"S3 URI for model output (default: s3://{_DEFAULT_BUCKET}/output/).",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g4dn.xlarge",
        help="SageMaker instance type (default: ml.g4dn.xlarge).",
    )
    parser.add_argument(
        "--role",
        default="arn:aws:iam::921400262514:role/SageMakerTrainingRole",
        help="SageMaker execution role ARN (default: SageMakerTrainingRole in account 921400262514).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS CLI profile name for SSO authentication (e.g. d-9067931f77-921400262514-admin+Q).",
    )
    parser.add_argument(
        "--region",
        default=_DEFAULT_REGION,
        help=f"AWS region (default: {_DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Training job name (auto-generated if not provided).",
    )
    # Training hyperparameters (passed through to sagemaker_train.py)
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
        help="Enable learned uncertainty weighting.",
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


def _package_source_code(s3_client, bucket: str, prefix: str) -> str:
    """Package the training source code into a tar.gz and upload to S3.

    Includes ``sagemaker_train.py`` (entry point) and the ``ml/`` package.
    Returns the S3 URI of the uploaded archive.
    """
    import tarfile
    import tempfile

    source_files = [
        "sagemaker_train.py",
        "ml/__init__.py",
        "ml/model.py",
        "ml/dataset.py",
        "ml/trainer.py",
        "ml/sentence_utils.py",
        "ml/requirements.txt",
    ]

    # SageMaker training toolkit installs requirements.txt from the
    # source directory root automatically.  Copy ml/requirements.txt
    # to the archive root so the toolkit finds it.
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    with tarfile.open(tmp_path, "w:gz") as tar:
        for filepath in source_files:
            if os.path.exists(filepath):
                tar.add(filepath)
            else:
                print(
                    f"WARNING: source file {filepath} not found, skipping",
                    file=sys.stderr,
                )
        # Add requirements.txt at the archive root for the toolkit
        if os.path.exists("ml/requirements.txt"):
            tar.add("ml/requirements.txt", arcname="requirements.txt")

    s3_key = f"{prefix}/source/sourcedir.tar.gz"
    print(f"[launcher] Uploading source code to s3://{bucket}/{s3_key}", file=sys.stderr)
    s3_client.upload_file(tmp_path, bucket, s3_key)
    os.unlink(tmp_path)

    return f"s3://{bucket}/{s3_key}"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the SageMaker launcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    import boto3

    # Create a boto3 session with optional SSO profile and region
    boto_session = boto3.Session(
        profile_name=args.profile,
        region_name=args.region,
    )
    sm_client = boto_session.client("sagemaker")
    s3_client = boto_session.client("s3")

    # Parse output S3 URI
    output_parts = args.output_s3.replace("s3://", "").split("/", 1)
    bucket = output_parts[0]
    prefix = output_parts[1].rstrip("/") if len(output_parts) > 1 else ""

    # Upload training data if local
    data_input = args.data
    if not data_input.startswith("s3://"):
        s3_key = f"{prefix}/training-data/{os.path.basename(data_input)}"
        print(
            f"[launcher] Uploading {data_input} to s3://{bucket}/{s3_key}",
            file=sys.stderr,
        )
        s3_client.upload_file(data_input, bucket, s3_key)
        data_input = f"s3://{bucket}/{s3_key}"

    # Package and upload source code
    source_s3_uri = _package_source_code(s3_client, bucket, prefix)

    # Generate job name if not provided
    job_name = args.job_name
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"ml-distillation-{timestamp}"

    # Build hyperparameters — must include sagemaker_program and
    # sagemaker_submit_directory so the training toolkit knows which
    # script to run and where the source code is.
    hyperparameters = {
        "sagemaker_program": "sagemaker_train.py",
        "sagemaker_submit_directory": source_s3_uri,
        "encoder": str(args.encoder),
        "batch-size": str(args.batch_size),
        "encoder-lr": str(args.encoder_lr),
        "heads-lr": str(args.heads_lr),
        "sentence-heads-lr": str(getattr(args, "sentence_heads_lr", 5e-3)),
        "epochs": str(args.epochs),
        "warmup-fraction": str(args.warmup_fraction),
        "max-seq-length": str(args.max_seq_length),
        "loss-weights": str(args.loss_weights),
        "val-split": str(args.val_split),
        "seed": str(args.seed),
    }
    if args.uncertainty_weighting:
        hyperparameters["uncertainty-weighting"] = ""

    # Use the official PyTorch 2.1 training image
    ecr_account = "763104351884"
    training_image = (
        f"{ecr_account}.dkr.ecr.{args.region}.amazonaws.com/"
        f"pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
    )

    print(f"[launcher] Region: {args.region}", file=sys.stderr)
    print(f"[launcher] Job name: {job_name}", file=sys.stderr)
    print(f"[launcher] Instance: {args.instance_type}", file=sys.stderr)
    print(f"[launcher] Image: {training_image}", file=sys.stderr)
    print(f"[launcher] Data: {data_input}", file=sys.stderr)
    print(f"[launcher] Source: {source_s3_uri}", file=sys.stderr)

    sm_client.create_training_job(
        TrainingJobName=job_name,
        RoleArn=args.role,
        AlgorithmSpecification={
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
        },
        HyperParameters=hyperparameters,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": data_input,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": args.output_s3,
        },
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400,
        },
    )

    print(
        f"[launcher] Training job {job_name} submitted successfully.",
        file=sys.stderr,
    )
    print(
        f"[launcher] Monitor at: https://{args.region}.console.aws.amazon.com/"
        f"sagemaker/home?region={args.region}#/jobs/{job_name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
