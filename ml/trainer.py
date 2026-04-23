"""Shared training logic for the ML Distillation Layer (Layer 6).

Contains the :class:`TrainConfig` dataclass, multi-task loss computation,
evaluation metrics, and the :func:`train` entry-point invoked by both the
local training script and the SageMaker entry point.

Requirements implemented: 12.1–12.4, 13.1–13.4, 14.5, 15.1–15.5,
25.1, 25.2, 25.5, 27.1, 27.6.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ml.dataset import LinguisticDataset, linguistic_collate_fn
from ml.model import (
    LinguisticModel,
    _DIAGNOSIS_KEYS,
    _ISSUE_KEYS,
    _SCORE_KEYS,
)

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyper-parameters with sensible defaults.

    See the design document §4 (Trainer) for rationale behind each default.
    """

    encoder_name: str = "dicta-il/dictabert"
    batch_size: int = 16
    encoder_lr: float = 2e-5
    heads_lr: float = 1e-3
    sentence_heads_lr: float = 5e-3
    epochs: int = 3
    warmup_fraction: float = 0.1
    max_seq_length: int = 512
    loss_weights: tuple[float, float, float, float, float] = (1.0, 1.5, 2.0, 1.5, 1.5)
    use_uncertainty_weighting: bool = False
    val_split: float = 0.1
    seed: int = 42


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _detect_device(device: str | None) -> torch.device:
    """Auto-detect the best available device: CUDA > CPU.

    MPS (Apple Silicon GPU) is skipped by default because DictaBERT
    (~738 MB) exceeds the typical MPS memory budget (~6.8 GB shared)
    when combined with optimizer state and activations.  Use
    ``device="mps"`` explicitly to force MPS if you know your system
    has enough headroom.

    Parameters
    ----------
    device:
        Explicit device string (e.g. ``"cuda"``, ``"cpu"``, ``"mps"``).
        When *None*, the function probes hardware in priority order
        (CUDA > CPU — MPS is skipped).

    Returns
    -------
    torch.device
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is intentionally skipped in auto-detection — DictaBERT is too
    # large for the default MPS memory limit on most Apple Silicon Macs.
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _compute_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: tuple[float, ...],
    log_vars: torch.Tensor | None = None,
    sentence_preds: list[torch.Tensor] | None = None,
    pair_preds: list[torch.Tensor] | None = None,
    sentence_targets: list[torch.Tensor] | None = None,
    pair_targets: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute the multi-task loss.

    ``L = w1 * MSE(scores) + w2 * BCE(issues) + w3 * BCE(diagnoses)
         + w4 * BCE(sentence_complexity) + w5 * BCE(weak_cohesion)``

    When *log_vars* is provided (uncertainty weighting, Kendall et al. 2018),
    each head loss is scaled as ``L_i / (2 * exp(s_i)) + s_i / 2``.

    Parameters
    ----------
    predictions:
        Dict with keys ``"scores"``, ``"issues"``, ``"diagnoses"`` — model
        outputs (sigmoid-activated, values in [0, 1]).
    targets:
        Dict with the same keys — ground-truth tensors.
    weights:
        Weight tuple ``(w_scores, w_issues, w_diagnoses, w_sentence, w_pairs)``.
        When only 3 weights are provided, sentence-level terms are skipped.
    log_vars:
        Optional tensor of shape ``(3,)`` containing learned log-variance
        parameters for uncertainty weighting (applied to the first 3 terms).
    sentence_preds:
        Optional list of per-batch-item sentence complexity predictions.
    pair_preds:
        Optional list of per-batch-item pair cohesion predictions.
    sentence_targets:
        Optional list of per-batch-item sentence complexity targets.
    pair_targets:
        Optional list of per-batch-item pair cohesion targets.

    Returns
    -------
    Scalar loss tensor.
    """
    loss_scores = F.mse_loss(predictions["scores"], targets["scores"])
    loss_issues = F.binary_cross_entropy(
        predictions["issues"].clamp(1e-7, 1 - 1e-7),
        targets["issues"].clamp(0.0, 1.0),
    )
    loss_diagnoses = F.binary_cross_entropy(
        predictions["diagnoses"].clamp(1e-7, 1 - 1e-7),
        targets["diagnoses"].clamp(0.0, 1.0),
    )

    if log_vars is not None:
        # Uncertainty weighting: L_i / (2 * exp(s_i)) + s_i / 2
        losses = torch.stack([loss_scores, loss_issues, loss_diagnoses])
        precisions = torch.exp(-log_vars)
        total = (precisions * losses / 2.0 + log_vars / 2.0).sum()
    else:
        w1 = weights[0] if len(weights) > 0 else 1.0
        w2 = weights[1] if len(weights) > 1 else 1.5
        w3 = weights[2] if len(weights) > 2 else 2.0
        total = w1 * loss_scores + w2 * loss_issues + w3 * loss_diagnoses

    # Sentence-level loss terms
    w4 = weights[3] if len(weights) > 3 else 1.5
    w5 = weights[4] if len(weights) > 4 else 1.5

    if sentence_preds is not None and sentence_targets is not None:
        sent_losses: list[torch.Tensor] = []
        for sp, st_ in zip(sentence_preds, sentence_targets):
            if sp.numel() > 0 and st_.numel() > 0:
                min_len = min(sp.shape[0], st_.shape[0])
                sent_losses.append(
                    F.binary_cross_entropy(
                        sp[:min_len].clamp(1e-7, 1 - 1e-7),
                        st_[:min_len].clamp(0.0, 1.0),
                    )
                )
        if sent_losses:
            total = total + w4 * torch.stack(sent_losses).mean()

    if pair_preds is not None and pair_targets is not None:
        pair_losses: list[torch.Tensor] = []
        for pp, pt_ in zip(pair_preds, pair_targets):
            if pp.numel() > 0 and pt_.numel() > 0:
                min_len = min(pp.shape[0], pt_.shape[0])
                pair_losses.append(
                    F.binary_cross_entropy(
                        pp[:min_len].clamp(1e-7, 1 - 1e-7),
                        pt_[:min_len].clamp(0.0, 1.0),
                    )
                )
        if pair_losses:
            total = total + w5 * torch.stack(pair_losses).mean()

    return total


# ---------------------------------------------------------------------------
# Manual metric helpers (avoid sklearn / scipy dependency)
# ---------------------------------------------------------------------------


def _f1_per_type(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> list[float]:
    """Compute per-column F1 at the given threshold.

    Parameters
    ----------
    preds:
        Predicted probabilities, shape ``(N, C)``.
    targets:
        Ground-truth probabilities, shape ``(N, C)``.
    threshold:
        Binarisation threshold.

    Returns
    -------
    List of *C* F1 values.
    """
    pred_bin = (preds >= threshold).float()
    tgt_bin = (targets >= threshold).float()
    f1s: list[float] = []
    for c in range(preds.shape[1]):
        tp = (pred_bin[:, c] * tgt_bin[:, c]).sum().item()
        fp = (pred_bin[:, c] * (1 - tgt_bin[:, c])).sum().item()
        fn = ((1 - pred_bin[:, c]) * tgt_bin[:, c]).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return f1s


def _spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman rank correlation between two 1-D tensors.

    Uses the standard formula: ``1 - 6 * sum(d^2) / (n * (n^2 - 1))``.
    Returns 0.0 when the input has fewer than 2 elements.
    """
    n = x.numel()
    if n < 2:
        return 0.0

    def _rank(t: torch.Tensor) -> torch.Tensor:
        order = t.argsort()
        ranks = torch.empty_like(t)
        ranks[order] = torch.arange(1, n + 1, dtype=t.dtype, device=t.device)
        return ranks

    rx = _rank(x.float())
    ry = _rank(y.float())
    d = rx - ry
    rho = 1.0 - 6.0 * (d * d).sum().item() / (n * (n * n - 1))
    return rho


def _binary_f1(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute binary F1 for 1-D prediction and target tensors."""
    pred_bin = (preds >= threshold).float()
    tgt_bin = (targets >= threshold).float()
    tp = (pred_bin * tgt_bin).sum().item()
    fp = (pred_bin * (1 - tgt_bin)).sum().item()
    fn = ((1 - pred_bin) * tgt_bin).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model: LinguisticModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run the model on *dataloader* and compute evaluation metrics.

    Returns
    -------
    dict with keys:
        ``rmse_scores``   — dict mapping each score name to its RMSE.
        ``f1_issues``     — dict mapping each issue name to its F1.
        ``f1_diagnoses``  — dict mapping each diagnosis name to its F1.
        ``spearman_issues``    — Spearman ρ across all issue severities.
        ``spearman_diagnoses`` — Spearman ρ across all diagnosis severities.
        ``val_loss``      — mean validation loss (fixed weights 1,1,1).
        ``f1_sentence_complexity`` — per-sentence F1 (when data available).
        ``f1_weak_cohesion``       — per-pair F1 (when data available).
    """
    model.eval()
    all_pred_scores: list[torch.Tensor] = []
    all_tgt_scores: list[torch.Tensor] = []
    all_pred_issues: list[torch.Tensor] = []
    all_tgt_issues: list[torch.Tensor] = []
    all_pred_diag: list[torch.Tensor] = []
    all_tgt_diag: list[torch.Tensor] = []
    all_sent_preds: list[torch.Tensor] = []
    all_sent_targets: list[torch.Tensor] = []
    all_pair_preds: list[torch.Tensor] = []
    all_pair_targets: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {
                "scores": batch["scores"].to(device),
                "issues": batch["issues"].to(device),
                "diagnoses": batch["diagnoses"].to(device),
            }

            # Pass sentence boundaries if available
            sentence_boundaries = batch.get("sentence_boundaries")
            preds = model(input_ids, attention_mask, sentence_boundaries=sentence_boundaries)

            loss = _compute_loss(preds, targets, (1.0, 1.0, 1.0))
            total_loss += loss.item()
            n_batches += 1

            all_pred_scores.append(preds["scores"].cpu())
            all_tgt_scores.append(targets["scores"].cpu())
            all_pred_issues.append(preds["issues"].cpu())
            all_tgt_issues.append(targets["issues"].cpu())
            all_pred_diag.append(preds["diagnoses"].cpu())
            all_tgt_diag.append(targets["diagnoses"].cpu())

            # Collect sentence-level predictions and targets
            if "sentence_complexity" in preds and sentence_boundaries is not None:
                sent_targets = batch.get("sentence_complexities", [])
                for sp in preds["sentence_complexity"]:
                    all_sent_preds.append(sp.cpu())
                for st_ in sent_targets:
                    all_sent_targets.append(st_.to("cpu") if isinstance(st_, torch.Tensor) else st_)

            if "weak_cohesion" in preds and sentence_boundaries is not None:
                pair_targets = batch.get("cohesion_pairs", [])
                for pp in preds["weak_cohesion"]:
                    all_pair_preds.append(pp.cpu())
                for pt_ in pair_targets:
                    all_pair_targets.append(pt_.to("cpu") if isinstance(pt_, torch.Tensor) else pt_)

    pred_scores = torch.cat(all_pred_scores, dim=0)
    tgt_scores = torch.cat(all_tgt_scores, dim=0)
    pred_issues = torch.cat(all_pred_issues, dim=0)
    tgt_issues = torch.cat(all_tgt_issues, dim=0)
    pred_diag = torch.cat(all_pred_diag, dim=0)
    tgt_diag = torch.cat(all_tgt_diag, dim=0)

    # RMSE per score type
    rmse_scores: dict[str, float] = {}
    for i, key in enumerate(_SCORE_KEYS):
        mse = F.mse_loss(pred_scores[:, i], tgt_scores[:, i]).item()
        rmse_scores[key] = math.sqrt(mse)

    # F1 per issue type at threshold 0.5
    issue_f1_list = _f1_per_type(pred_issues, tgt_issues, threshold=0.5)
    f1_issues = dict(zip(_ISSUE_KEYS, issue_f1_list))

    # F1 per diagnosis type at threshold 0.5
    diag_f1_list = _f1_per_type(pred_diag, tgt_diag, threshold=0.5)
    f1_diagnoses = dict(zip(_DIAGNOSIS_KEYS, diag_f1_list))

    # Spearman rank correlation for issues and diagnoses
    spearman_issues = _spearman_rank_correlation(
        pred_issues.flatten(), tgt_issues.flatten()
    )
    spearman_diagnoses = _spearman_rank_correlation(
        pred_diag.flatten(), tgt_diag.flatten()
    )

    avg_loss = total_loss / max(n_batches, 1)

    result = {
        "rmse_scores": rmse_scores,
        "f1_issues": f1_issues,
        "f1_diagnoses": f1_diagnoses,
        "spearman_issues": spearman_issues,
        "spearman_diagnoses": spearman_diagnoses,
        "val_loss": avg_loss,
    }

    # Sentence-level F1 metrics
    if all_sent_preds and all_sent_targets:
        flat_sp = torch.cat([s for s in all_sent_preds if s.numel() > 0])
        flat_st = torch.cat([s for s in all_sent_targets if s.numel() > 0])
        if flat_sp.numel() > 0 and flat_st.numel() > 0:
            min_len = min(flat_sp.shape[0], flat_st.shape[0])
            result["f1_sentence_complexity"] = _binary_f1(
                flat_sp[:min_len], flat_st[:min_len], threshold=0.3
            )

    if all_pair_preds and all_pair_targets:
        flat_pp = torch.cat([p for p in all_pair_preds if p.numel() > 0])
        flat_pt = torch.cat([p for p in all_pair_targets if p.numel() > 0])
        if flat_pp.numel() > 0 and flat_pt.numel() > 0:
            min_len = min(flat_pp.shape[0], flat_pt.shape[0])
            result["f1_weak_cohesion"] = _binary_f1(
                flat_pp[:min_len], flat_pt[:min_len], threshold=0.3
            )

    return result


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    output_dir: str,
    model: LinguisticModel,
    tokenizer: AutoTokenizer,
    config: TrainConfig,
    optimizer: AdamW,
    epoch: int,
) -> None:
    """Persist a training checkpoint to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    with open(os.path.join(output_dir, "epoch.txt"), "w") as f:
        f.write(str(epoch))


def _load_checkpoint(
    checkpoint_dir: str,
    model: LinguisticModel,
    optimizer: AdamW,
) -> int:
    """Restore model + optimizer state from *checkpoint_dir*.

    Returns the epoch number to resume from (i.e. the next epoch to run).

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or required files are missing.
    """
    cp = Path(checkpoint_dir)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model_path = cp / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))

    opt_path = cp / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(str(opt_path), map_location="cpu", weights_only=True))

    epoch_path = cp / "epoch.txt"
    if epoch_path.exists():
        start_epoch = int(epoch_path.read_text().strip()) + 1
    else:
        start_epoch = 0

    return start_epoch


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------


def train(
    data_path: str,
    output_dir: str,
    config: TrainConfig = TrainConfig(),
    device: str | None = None,
    resume_from: str | None = None,
) -> dict:
    """Run the full training pipeline.

    Parameters
    ----------
    data_path:
        Path to a JSONL file produced by :func:`ml.export.export_training_data`.
    output_dir:
        Directory where checkpoints and the final model are saved.
    config:
        Training hyper-parameters.
    device:
        Explicit device string.  *None* triggers auto-detection
        (CUDA > MPS > CPU).
    resume_from:
        Path to a checkpoint directory to resume training from.

    Returns
    -------
    dict — final evaluation metrics from the last epoch.
    """
    dev = _detect_device(device)
    logger.info("Using device: %s", dev)

    # Reproducibility
    torch.manual_seed(config.seed)

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    full_dataset = LinguisticDataset(
        data_path, tokenizer, max_length=config.max_seq_length
    )

    # Train / val split
    total = len(full_dataset)
    val_size = max(1, int(total * config.val_split))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=linguistic_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=linguistic_collate_fn,
    )

    # Model
    model = LinguisticModel(encoder_name=config.encoder_name)
    model.to(dev)

    # Uncertainty weighting (optional learned log-variance parameters)
    log_vars: torch.Tensor | None = None
    extra_params: list[torch.Tensor] = []
    if config.use_uncertainty_weighting:
        log_vars = nn.Parameter(torch.zeros(3, device=dev))
        extra_params.append(log_vars)

    # Optimizer with differential learning rates
    encoder_params = list(model.encoder.parameters())
    doc_head_params = (
        list(model.scores_head.parameters())
        + list(model.issues_head.parameters())
        + list(model.diagnoses_head.parameters())
    )
    sentence_head_params = (
        list(model.sentence_head.parameters())
        + list(model.pair_head.parameters())
    )
    param_groups = [
        {"params": encoder_params, "lr": config.encoder_lr},
        {"params": doc_head_params + extra_params, "lr": config.heads_lr},
        {"params": sentence_head_params, "lr": config.sentence_heads_lr},
    ]
    optimizer = AdamW(param_groups)

    # Scheduler: linear warmup + linear decay
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint
    start_epoch = 0
    if resume_from is not None:
        start_epoch = _load_checkpoint(resume_from, model, optimizer)
        model.to(dev)
        logger.info("Resumed from checkpoint at epoch %d", start_epoch)
        # Advance scheduler to the correct step
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    # Training loop
    metrics: dict = {}
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            targets = {
                "scores": batch["scores"].to(dev),
                "issues": batch["issues"].to(dev),
                "diagnoses": batch["diagnoses"].to(dev),
            }

            sentence_boundaries = batch.get("sentence_boundaries")
            preds = model(input_ids, attention_mask, sentence_boundaries=sentence_boundaries)

            # Prepare sentence-level targets
            sent_targets = None
            pair_targets = None
            sent_preds = preds.get("sentence_complexity")
            pair_preds_batch = preds.get("weak_cohesion")

            if sent_preds is not None and "sentence_complexities" in batch:
                sent_targets = [t.to(dev) for t in batch["sentence_complexities"]]
            if pair_preds_batch is not None and "cohesion_pairs" in batch:
                pair_targets = [t.to(dev) for t in batch["cohesion_pairs"]]

            loss = _compute_loss(
                preds, targets, config.loss_weights, log_vars,
                sentence_preds=sent_preds,
                pair_preds=pair_preds_batch,
                sentence_targets=sent_targets,
                pair_targets=pair_targets,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d/%d — train loss: %.4f",
            epoch + 1,
            config.epochs,
            avg_train_loss,
        )

        # Validation
        metrics = _evaluate(model, val_loader, dev)
        logger.info(
            "Epoch %d/%d — val loss: %.4f | spearman issues: %.3f | spearman diagnoses: %.3f",
            epoch + 1,
            config.epochs,
            metrics["val_loss"],
            metrics["spearman_issues"],
            metrics["spearman_diagnoses"],
        )

        # Save checkpoint after each epoch
        _save_checkpoint(output_dir, model, tokenizer, config, optimizer, epoch)
        logger.info("Checkpoint saved to %s", output_dir)

    # Save final model + tokenizer + config
    _save_checkpoint(output_dir, model, tokenizer, config, optimizer, config.epochs - 1)
    logger.info("Training complete. Final model saved to %s", output_dir)

    return metrics
