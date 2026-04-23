"""Unit tests for CLI argument parsing (Layer 6).

Tests argument parsing for all four CLI entry points without invoking
the actual training, inference, or export logic.

Requirements tested: 22.1, 22.2, 22.3, 22.4, 23.1, 24.1, 24.4.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# train_model.py tests
# ---------------------------------------------------------------------------


class TestTrainModelParser:
    """Test argument parsing for train_model.py."""

    def test_required_args_only(self) -> None:
        from train_model import build_parser

        parser = build_parser()
        args = parser.parse_args(["--data", "train.jsonl", "--output", "out/"])
        assert args.data == "train.jsonl"
        assert args.output == "out/"

    def test_defaults(self) -> None:
        from train_model import build_parser

        parser = build_parser()
        args = parser.parse_args(["--data", "d.jsonl", "--output", "o/"])
        assert args.encoder == "dicta-il/dictabert"
        assert args.batch_size == 16
        assert args.encoder_lr == 2e-5
        assert args.heads_lr == 1e-3
        assert args.epochs == 3
        assert args.warmup_fraction == 0.1
        assert args.max_seq_length == 512
        assert args.loss_weights == "1.0,1.5,2.0"
        assert args.uncertainty_weighting is False
        assert args.val_split == 0.1
        assert args.seed == 42
        assert args.resume is None
        assert args.device is None

    def test_all_args(self) -> None:
        from train_model import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--data", "train.jsonl",
            "--output", "checkpoints/",
            "--encoder", "onlplab/alephbert-base",
            "--batch-size", "32",
            "--encoder-lr", "3e-5",
            "--heads-lr", "5e-4",
            "--epochs", "10",
            "--warmup-fraction", "0.2",
            "--max-seq-length", "256",
            "--loss-weights", "2.0,3.0,4.0",
            "--uncertainty-weighting",
            "--val-split", "0.15",
            "--seed", "123",
            "--resume", "ckpt/",
            "--device", "cuda",
        ])
        assert args.encoder == "onlplab/alephbert-base"
        assert args.batch_size == 32
        assert args.encoder_lr == 3e-5
        assert args.heads_lr == 5e-4
        assert args.epochs == 10
        assert args.warmup_fraction == 0.2
        assert args.max_seq_length == 256
        assert args.loss_weights == "2.0,3.0,4.0"
        assert args.uncertainty_weighting is True
        assert args.val_split == 0.15
        assert args.seed == 123
        assert args.resume == "ckpt/"
        assert args.device == "cuda"

    def test_missing_required_args(self) -> None:
        from train_model import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parse_loss_weights(self) -> None:
        from train_model import parse_loss_weights

        result = parse_loss_weights("1.0,1.5,2.0")
        assert result == (1.0, 1.5, 2.0)

    def test_parse_loss_weights_custom(self) -> None:
        from train_model import parse_loss_weights

        result = parse_loss_weights("0.5, 2.5, 3.5")
        assert result == (0.5, 2.5, 3.5)

    def test_parse_loss_weights_wrong_count(self) -> None:
        from train_model import parse_loss_weights

        with pytest.raises(argparse.ArgumentTypeError):
            parse_loss_weights("1.0,2.0")

    def test_main_builds_config_and_calls_train(self) -> None:
        """Verify main() builds TrainConfig correctly and calls train()."""
        from train_model import main

        with patch("ml.trainer.train", return_value={"val_loss": 0.1}) as mock_t:
            main([
                "--data", "data.jsonl",
                "--output", "out/",
                "--batch-size", "8",
                "--epochs", "2",
            ])
            mock_t.assert_called_once()
            call_kwargs = mock_t.call_args
            assert call_kwargs[1]["data_path"] == "data.jsonl"
            assert call_kwargs[1]["output_dir"] == "out/"
            config = call_kwargs[1]["config"]
            assert config.batch_size == 8
            assert config.epochs == 2


# ---------------------------------------------------------------------------
# sagemaker_train.py tests
# ---------------------------------------------------------------------------


class TestSagemakerTrainParser:
    """Test argument parsing and env var mapping for sagemaker_train.py."""

    def test_defaults(self) -> None:
        from sagemaker_train import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.encoder == "dicta-il/dictabert"
        assert args.batch_size == 16
        assert args.epochs == 3

    def test_custom_hyperparameters(self) -> None:
        from sagemaker_train import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--encoder", "onlplab/alephbert-base",
            "--batch-size", "64",
            "--epochs", "5",
            "--encoder-lr", "1e-5",
        ])
        assert args.encoder == "onlplab/alephbert-base"
        assert args.batch_size == 64
        assert args.epochs == 5
        assert args.encoder_lr == 1e-5

    def test_env_var_mapping(self, tmp_path: "Path") -> None:
        """Verify SageMaker env vars are read correctly."""
        from sagemaker_train import main

        # Create a dummy JSONL file in the training channel
        train_dir = tmp_path / "training"
        train_dir.mkdir()
        jsonl_file = train_dir / "data.jsonl"
        jsonl_file.write_text('{"text": "test"}\n')

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        env = {
            "SM_CHANNEL_TRAINING": str(train_dir),
            "SM_MODEL_DIR": str(model_dir),
            "SM_OUTPUT_DATA_DIR": str(output_dir),
            "SM_NUM_GPUS": "1",
        }

        with patch.dict("os.environ", env), \
             patch("ml.trainer.train", return_value={"val_loss": 0.05}) as mock_train:
            main(["--epochs", "1"])
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args
            assert call_kwargs[1]["data_path"] == str(jsonl_file)
            assert call_kwargs[1]["output_dir"] == str(model_dir)
            assert call_kwargs[1]["device"] == "cuda"

    def test_env_var_no_gpus(self, tmp_path: "Path") -> None:
        """When SM_NUM_GPUS=0, device should be None (auto-detect)."""
        from sagemaker_train import main

        train_dir = tmp_path / "training"
        train_dir.mkdir()
        jsonl_file = train_dir / "data.jsonl"
        jsonl_file.write_text('{"text": "test"}\n')

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        env = {
            "SM_CHANNEL_TRAINING": str(train_dir),
            "SM_MODEL_DIR": str(model_dir),
            "SM_OUTPUT_DATA_DIR": str(output_dir),
            "SM_NUM_GPUS": "0",
        }

        with patch.dict("os.environ", env), \
             patch("ml.trainer.train", return_value={"val_loss": 0.05}) as mock_train:
            main(["--epochs", "1"])
            call_kwargs = mock_train.call_args
            assert call_kwargs[1]["device"] is None

    def test_parse_loss_weights(self) -> None:
        from sagemaker_train import parse_loss_weights

        result = parse_loss_weights("1.0,1.5,2.0")
        assert result == (1.0, 1.5, 2.0)


# ---------------------------------------------------------------------------
# launch_sagemaker_training.py tests
# ---------------------------------------------------------------------------


class TestLaunchSagemakerParser:
    """Test argument parsing for launch_sagemaker_training.py."""

    def test_required_args(self) -> None:
        from launch_sagemaker_training import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--data", "s3://bucket/data.jsonl",
            "--output-s3", "s3://bucket/output/",
            "--role", "arn:aws:iam::123456789:role/SageMakerRole",
        ])
        assert args.data == "s3://bucket/data.jsonl"
        assert args.output_s3 == "s3://bucket/output/"
        assert args.role == "arn:aws:iam::123456789:role/SageMakerRole"

    def test_defaults(self) -> None:
        from launch_sagemaker_training import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--data", "data.jsonl",
            "--output-s3", "s3://b/o/",
            "--role", "arn:role",
        ])
        assert args.instance_type == "ml.g4dn.xlarge"
        assert args.job_name is None
        assert args.encoder == "dicta-il/dictabert"
        assert args.batch_size == 16
        assert args.epochs == 3

    def test_all_args(self) -> None:
        from launch_sagemaker_training import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--data", "local/train.jsonl",
            "--output-s3", "s3://bucket/output/",
            "--instance-type", "ml.p3.2xlarge",
            "--role", "arn:role",
            "--job-name", "my-job",
            "--encoder", "onlplab/alephbert-base",
            "--batch-size", "32",
            "--epochs", "5",
            "--uncertainty-weighting",
        ])
        assert args.instance_type == "ml.p3.2xlarge"
        assert args.job_name == "my-job"
        assert args.encoder == "onlplab/alephbert-base"
        assert args.batch_size == 32
        assert args.epochs == 5
        assert args.uncertainty_weighting is True

    def test_missing_required_args(self) -> None:
        from launch_sagemaker_training import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_main_with_s3_data(self) -> None:
        """Verify main() uses S3 data directly and submits a training job."""
        mock_s3_client = MagicMock()
        mock_sm_client = MagicMock()
        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value.client.side_effect = lambda svc: (
            mock_sm_client if svc == "sagemaker" else mock_s3_client
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from launch_sagemaker_training import main

            main([
                "--data", "s3://bucket/data.jsonl",
                "--output-s3", "s3://bucket/output/",
                "--role", "arn:role",
                "--job-name", "test-job",
            ])
            # S3 data should NOT be uploaded (already on S3)
            # But source code IS uploaded
            mock_sm_client.create_training_job.assert_called_once()
            call_kwargs = mock_sm_client.create_training_job.call_args[1]
            assert call_kwargs["TrainingJobName"] == "test-job"

    def test_main_uploads_local_data(self) -> None:
        """Verify main() uploads local data to S3."""
        mock_s3_client = MagicMock()
        mock_sm_client = MagicMock()
        mock_boto3 = MagicMock()
        mock_boto3.Session.return_value.client.side_effect = lambda svc: (
            mock_sm_client if svc == "sagemaker" else mock_s3_client
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from launch_sagemaker_training import main

            main([
                "--data", "/local/path/train.jsonl",
                "--output-s3", "s3://mybucket/prefix/",
                "--role", "arn:role",
                "--job-name", "test-upload",
            ])
            # upload_file called at least once (data + source code)
            assert mock_s3_client.upload_file.call_count >= 1
            # Verify the training job was submitted
            mock_sm_client.create_training_job.assert_called_once()


# ---------------------------------------------------------------------------
# export_training_data.py tests
# ---------------------------------------------------------------------------


class TestExportTrainingDataParser:
    """Test subcommand parsing for export_training_data.py."""

    def test_export_subcommand(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "export",
            "--input-dirs", "dir1/", "dir2/",
            "--stats-path", "stats.json",
            "--output", "out.jsonl",
        ])
        assert args.command == "export"
        assert args.input_dirs == ["dir1/", "dir2/"]
        assert args.stats_path == "stats.json"
        assert args.output == "out.jsonl"
        assert args.stats_output is None

    def test_export_with_stats_output(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "export",
            "--input-dirs", "dir1/",
            "--stats-path", "stats.json",
            "--output", "out.jsonl",
            "--stats-output", "label_stats.json",
        ])
        assert args.stats_output == "label_stats.json"

    def test_infer_subcommand_with_text(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "infer",
            "--text", "שלום עולם",
            "--model-path", "model/",
        ])
        assert args.command == "infer"
        assert args.text == "שלום עולם"
        assert args.model_path == "model/"
        assert args.hybrid is False
        assert args.confidence_threshold == 0.7

    def test_infer_subcommand_with_input_file(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "infer",
            "--input", "text.txt",
            "--model-path", "model/",
            "--hybrid",
            "--confidence-threshold", "0.8",
            "--pretty",
        ])
        assert args.command == "infer"
        assert args.input == "text.txt"
        assert args.hybrid is True
        assert args.confidence_threshold == 0.8
        assert args.pretty is True

    def test_infer_text_and_input_mutually_exclusive(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "infer",
                "--text", "hello",
                "--input", "file.txt",
                "--model-path", "model/",
            ])

    def test_disagree_subcommand(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "disagree",
            "--predictions", "preds.jsonl",
            "--labels", "labels.jsonl",
            "--output", "disagree.jsonl",
        ])
        assert args.command == "disagree"
        assert args.predictions == "preds.jsonl"
        assert args.labels == "labels.jsonl"
        assert args.output == "disagree.jsonl"
        assert args.threshold == 0.3

    def test_disagree_custom_threshold(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "disagree",
            "--predictions", "p.jsonl",
            "--labels", "l.jsonl",
            "--output", "o.jsonl",
            "--threshold", "0.5",
        ])
        assert args.threshold == 0.5

    def test_merge_subcommand(self) -> None:
        from export_training_data import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "merge",
            "--base", "base.jsonl",
            "--disagreements", "disagree.jsonl",
            "--output", "merged.jsonl",
        ])
        assert args.command == "merge"
        assert args.base == "base.jsonl"
        assert args.disagreements == "disagree.jsonl"
        assert args.output == "merged.jsonl"

    def test_no_subcommand_exits(self) -> None:
        from export_training_data import main

        with pytest.raises(SystemExit):
            main([])
