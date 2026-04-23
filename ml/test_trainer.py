# Feature: ml-distillation-layer, Property 8: Multi-task loss as weighted sum

"""Property-based and unit tests for ml/trainer.py.

Tests the trainer module: multi-task loss as weighted sum (Property 8),
default config values, differential learning rates, checkpoint save/load,
evaluation metrics, and uncertainty weighting.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3,
15.1, 15.2, 15.3, 15.4**
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.optim import AdamW
from transformers import BertConfig, BertModel

from ml.model import LinguisticModel, _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS
from ml.trainer import (
    TrainConfig,
    _compute_loss,
    _f1_per_type,
    _load_checkpoint,
    _save_checkpoint,
    _spearman_rank_correlation,
)


# ---------------------------------------------------------------------------
# Helper: build a tiny model (same pattern as test_model.py)
# ---------------------------------------------------------------------------


def _make_tiny_model() -> LinguisticModel:
    """Create a LinguisticModel backed by a 2-layer, hidden_dim=32 encoder."""
    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=1000,
    )
    tiny_encoder = BertModel(config)

    model = LinguisticModel.__new__(LinguisticModel)
    nn.Module.__init__(model)
    model.encoder = tiny_encoder
    hidden = tiny_encoder.config.hidden_size
    model.scores_head = nn.Linear(hidden, 5)
    model.issues_head = nn.Linear(hidden, 17)
    model.diagnoses_head = nn.Linear(hidden, 8)
    return model


# ===========================================================================
# Property 8: Multi-task loss as weighted sum
# ===========================================================================

# Feature: ml-distillation-layer, Property 8: Multi-task loss as weighted sum
# **Validates: Requirements 12.1**


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    w2=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    w3=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_multi_task_loss_weighted_sum(
    batch_size: int, w1: float, w2: float, w3: float
) -> None:
    """For any set of prediction tensors and target tensors, and any positive
    weight triple (w1, w2, w3), the multi-task loss equals
    w1 * MSE(pred_scores, target_scores) + w2 * BCE(pred_issues, target_issues)
    + w3 * BCE(pred_diagnoses, target_diagnoses)."""

    # Generate random prediction and target tensors in [0.01, 0.99]
    # to avoid exact 0/1 for BCE stability
    pred_scores = torch.rand(batch_size, 5) * 0.98 + 0.01
    pred_issues = torch.rand(batch_size, 17) * 0.98 + 0.01
    pred_diagnoses = torch.rand(batch_size, 8) * 0.98 + 0.01

    tgt_scores = torch.rand(batch_size, 5) * 0.98 + 0.01
    tgt_issues = torch.rand(batch_size, 17) * 0.98 + 0.01
    tgt_diagnoses = torch.rand(batch_size, 8) * 0.98 + 0.01

    predictions = {
        "scores": pred_scores,
        "issues": pred_issues,
        "diagnoses": pred_diagnoses,
    }
    targets = {
        "scores": tgt_scores,
        "issues": tgt_issues,
        "diagnoses": tgt_diagnoses,
    }

    # Compute via _compute_loss
    actual = _compute_loss(predictions, targets, (w1, w2, w3))

    # Compute expected manually
    expected_mse = F.mse_loss(pred_scores, tgt_scores)
    expected_bce_issues = F.binary_cross_entropy(pred_issues, tgt_issues)
    expected_bce_diag = F.binary_cross_entropy(pred_diagnoses, tgt_diagnoses)
    expected = w1 * expected_mse + w2 * expected_bce_issues + w3 * expected_bce_diag

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Loss mismatch: actual={actual.item():.6f}, expected={expected.item():.6f}"
    )


# ===========================================================================
# Unit tests for trainer module (Task 5.3)
# ===========================================================================


class TestTrainerUnit:
    """Unit tests for TrainConfig, optimizer, checkpointing, metrics, and
    uncertainty weighting."""

    # -- Test: default config values match specification -------------------

    def test_default_config_values(self) -> None:
        """TrainConfig defaults match the design specification."""
        cfg = TrainConfig()

        assert cfg.encoder_name == "dicta-il/dictabert"
        assert cfg.batch_size == 16
        assert cfg.encoder_lr == 2e-5
        assert cfg.heads_lr == 1e-3
        assert cfg.epochs == 3
        assert cfg.warmup_fraction == 0.1
        assert cfg.max_seq_length == 512
        assert cfg.loss_weights == (1.0, 1.5, 2.0, 1.5, 1.5)
        assert cfg.sentence_heads_lr == 5e-3
        assert cfg.use_uncertainty_weighting is False
        assert cfg.val_split == 0.1
        assert cfg.seed == 42

    # -- Test: differential learning rates in optimizer --------------------

    def test_differential_learning_rates(self) -> None:
        """Optimizer parameter groups use encoder_lr for encoder params and
        heads_lr for head params."""
        model = _make_tiny_model()
        cfg = TrainConfig()

        encoder_params = list(model.encoder.parameters())
        head_params = (
            list(model.scores_head.parameters())
            + list(model.issues_head.parameters())
            + list(model.diagnoses_head.parameters())
        )
        param_groups = [
            {"params": encoder_params, "lr": cfg.encoder_lr},
            {"params": head_params, "lr": cfg.heads_lr},
        ]
        optimizer = AdamW(param_groups)

        # Verify the optimizer has 2 parameter groups
        assert len(optimizer.param_groups) == 2

        # Group 0: encoder params with encoder_lr
        assert optimizer.param_groups[0]["lr"] == cfg.encoder_lr
        encoder_param_count = sum(1 for _ in model.encoder.parameters())
        assert len(optimizer.param_groups[0]["params"]) == encoder_param_count

        # Group 1: head params with heads_lr
        assert optimizer.param_groups[1]["lr"] == cfg.heads_lr
        head_param_count = sum(
            1
            for m in [model.scores_head, model.issues_head, model.diagnoses_head]
            for _ in m.parameters()
        )
        assert len(optimizer.param_groups[1]["params"]) == head_param_count

    # -- Test: checkpoint save/load round-trip -----------------------------

    def test_checkpoint_save_load_roundtrip(self, tmp_path) -> None:
        """Save a tiny model checkpoint, load it back, verify state dict
        matches."""
        model = _make_tiny_model()
        cfg = TrainConfig()

        # Build a minimal optimizer
        optimizer = AdamW(model.parameters(), lr=1e-3)

        # We need a tokenizer-like object for save_pretrained.
        # Use a minimal BertTokenizer from the tiny config.
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        checkpoint_dir = str(tmp_path / "checkpoint")
        _save_checkpoint(checkpoint_dir, model, tokenizer, cfg, optimizer, epoch=1)

        # Verify files exist
        import os

        assert os.path.exists(os.path.join(checkpoint_dir, "model.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "config.json"))
        assert os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "epoch.txt"))
        assert os.path.isdir(os.path.join(checkpoint_dir, "tokenizer"))

        # Load into a fresh model
        model2 = _make_tiny_model()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        start_epoch = _load_checkpoint(checkpoint_dir, model2, optimizer2)

        # Epoch should be 2 (saved epoch 1, resume from next)
        assert start_epoch == 2

        # State dicts should match
        sd1 = model.state_dict()
        sd2 = model2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for key in sd1:
            assert torch.allclose(sd1[key], sd2[key], atol=1e-6), (
                f"State dict mismatch for key: {key}"
            )

        # Config should round-trip
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            loaded_cfg = json.load(f)
        assert loaded_cfg["encoder_name"] == cfg.encoder_name
        assert loaded_cfg["batch_size"] == cfg.batch_size
        assert loaded_cfg["epochs"] == cfg.epochs

    # -- Test: evaluation metrics with known values ------------------------

    def test_rmse_with_known_values(self) -> None:
        """RMSE computation: known pred/target → expected RMSE."""
        # pred = [0.0, 0.5, 1.0], target = [0.0, 0.5, 1.0] → RMSE = 0
        pred = torch.tensor([[0.0, 0.5, 1.0, 0.3, 0.7]])
        target = torch.tensor([[0.0, 0.5, 1.0, 0.3, 0.7]])
        for i in range(5):
            mse = F.mse_loss(pred[:, i], target[:, i]).item()
            rmse = math.sqrt(mse)
            assert rmse == 0.0

        # pred = [1.0], target = [0.0] → RMSE = 1.0
        pred2 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        target2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        mse = F.mse_loss(pred2[:, 0], target2[:, 0]).item()
        rmse = math.sqrt(mse)
        assert abs(rmse - 1.0) < 1e-6

    def test_f1_with_known_values(self) -> None:
        """F1 computation with known predictions and targets."""
        # 4 samples, 2 types
        # Type 0: pred=[0.8, 0.3, 0.6, 0.2], target=[0.8, 0.1, 0.7, 0.3]
        # At threshold 0.5:
        #   pred_bin  = [1, 0, 1, 0]
        #   tgt_bin   = [1, 0, 1, 0]
        #   TP=2, FP=0, FN=0 → F1=1.0
        # Type 1: pred=[0.1, 0.9, 0.4, 0.6], target=[0.6, 0.8, 0.2, 0.1]
        # At threshold 0.5:
        #   pred_bin  = [0, 1, 0, 1]
        #   tgt_bin   = [1, 1, 0, 0]
        #   TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        preds = torch.tensor([
            [0.8, 0.1],
            [0.3, 0.9],
            [0.6, 0.4],
            [0.2, 0.6],
        ])
        targets = torch.tensor([
            [0.8, 0.6],
            [0.1, 0.8],
            [0.7, 0.2],
            [0.3, 0.1],
        ])
        f1s = _f1_per_type(preds, targets, threshold=0.5)
        assert len(f1s) == 2
        assert abs(f1s[0] - 1.0) < 1e-6
        assert abs(f1s[1] - 0.5) < 1e-6

    def test_spearman_with_known_values(self) -> None:
        """Spearman correlation with perfectly correlated and anti-correlated
        inputs."""
        # Perfect positive correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        rho = _spearman_rank_correlation(x, y)
        assert abs(rho - 1.0) < 1e-6

        # Perfect negative correlation
        y_neg = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        rho_neg = _spearman_rank_correlation(x, y_neg)
        assert abs(rho_neg - (-1.0)) < 1e-6

        # Single element → 0.0
        rho_single = _spearman_rank_correlation(
            torch.tensor([1.0]), torch.tensor([2.0])
        )
        assert rho_single == 0.0

    # -- Test: uncertainty weighting log-variance parameters ---------------

    def test_uncertainty_weighting_differs_from_fixed(self) -> None:
        """_compute_loss with log_vars produces a different result than
        without, confirming uncertainty weighting is active."""
        pred_scores = torch.rand(2, 5) * 0.98 + 0.01
        pred_issues = torch.rand(2, 17) * 0.98 + 0.01
        pred_diagnoses = torch.rand(2, 8) * 0.98 + 0.01

        tgt_scores = torch.rand(2, 5) * 0.98 + 0.01
        tgt_issues = torch.rand(2, 17) * 0.98 + 0.01
        tgt_diagnoses = torch.rand(2, 8) * 0.98 + 0.01

        predictions = {
            "scores": pred_scores,
            "issues": pred_issues,
            "diagnoses": pred_diagnoses,
        }
        targets = {
            "scores": tgt_scores,
            "issues": tgt_issues,
            "diagnoses": tgt_diagnoses,
        }

        weights = (1.0, 1.5, 2.0)

        # Fixed-weight loss
        loss_fixed = _compute_loss(predictions, targets, weights, log_vars=None)

        # Uncertainty-weighted loss with non-zero log_vars
        log_vars = torch.tensor([0.5, -0.3, 1.0])
        loss_uw = _compute_loss(predictions, targets, weights, log_vars=log_vars)

        # They should differ (extremely unlikely to be equal with random data
        # and non-zero log_vars)
        assert not torch.allclose(loss_fixed, loss_uw), (
            "Uncertainty-weighted loss should differ from fixed-weight loss"
        )

    def test_uncertainty_weighting_formula(self) -> None:
        """Verify uncertainty weighting follows L_i/(2*exp(s_i)) + s_i/2."""
        pred_scores = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
        pred_issues = torch.tensor([[0.5] * 17])
        pred_diagnoses = torch.tensor([[0.5] * 8])

        tgt_scores = torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.3]])
        tgt_issues = torch.tensor([[0.3] * 17])
        tgt_diagnoses = torch.tensor([[0.3] * 8])

        predictions = {
            "scores": pred_scores,
            "issues": pred_issues,
            "diagnoses": pred_diagnoses,
        }
        targets = {
            "scores": tgt_scores,
            "issues": tgt_issues,
            "diagnoses": tgt_diagnoses,
        }

        log_vars = torch.tensor([0.5, -0.3, 1.0])

        actual = _compute_loss(predictions, targets, (1.0, 1.0, 1.0), log_vars=log_vars)

        # Compute expected manually
        l_scores = F.mse_loss(pred_scores, tgt_scores)
        l_issues = F.binary_cross_entropy(pred_issues, tgt_issues)
        l_diag = F.binary_cross_entropy(pred_diagnoses, tgt_diagnoses)

        losses = torch.stack([l_scores, l_issues, l_diag])
        precisions = torch.exp(-log_vars)
        expected = (precisions * losses / 2.0 + log_vars / 2.0).sum()

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"Uncertainty loss mismatch: actual={actual.item():.6f}, "
            f"expected={expected.item():.6f}"
        )
