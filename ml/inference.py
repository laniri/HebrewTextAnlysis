"""Inference module for the ML Distillation Layer (Layer 6).

Provides fast-path prediction from raw Hebrew text, hybrid mode with
confidence-based fallback to the full pipeline, intervention derivation
from predicted diagnoses, and JSON serialization of inference output.

Requirements implemented: 16.1–16.5, 17.1–17.4, 18.1–18.5, 19.1–19.4,
27.1, 27.6.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from analysis.diagnosis_models import Diagnosis
from analysis.intervention_mapper import map_interventions
from ml.model import LinguisticModel, _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS
from ml.sentence_utils import find_token_boundaries, split_into_sentences
from ml.trainer import _detect_device

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_model(
    model_path: str, device: torch.device
) -> tuple[LinguisticModel, PreTrainedTokenizer]:
    """Load a trained LinguisticModel and its tokenizer from *model_path*.

    Expects the checkpoint directory to contain:
    - ``config.json`` with at least an ``"encoder_name"`` field
    - ``model.pt`` with the model state dict
    - ``tokenizer/`` sub-directory with a saved HuggingFace tokenizer

    Parameters
    ----------
    model_path:
        Path to the checkpoint directory.
    device:
        Target device for the model.

    Returns
    -------
    tuple of (LinguisticModel, PreTrainedTokenizer)

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or required files are missing.
    """
    cp = Path(model_path)
    if not cp.exists():
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_path}")

    config_file = cp / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_name = config.get("encoder_name", "dicta-il/dictabert")

    model = LinguisticModel(encoder_name=encoder_name)

    model_file = cp / "model.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"model.pt not found in {model_path}")

    state_dict = torch.load(str(model_file), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    tokenizer_dir = cp / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer/ directory not found in {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    model.to(device)
    model.eval()

    return model, tokenizer


def _predictions_to_dicts(output: dict[str, torch.Tensor]) -> dict:
    """Convert raw model tensor outputs to named dicts.

    Parameters
    ----------
    output:
        Dict with keys ``"scores"`` *(1, 5)*, ``"issues"`` *(1, 17)*,
        ``"diagnoses"`` *(1, 8)* — single-sample batch from the model.
        May also contain ``"sentence_complexity"`` and ``"weak_cohesion"``
        as lists of tensors.

    Returns
    -------
    dict with keys ``"scores"``, ``"issues"``, ``"diagnoses"``, each
    mapping canonical key names to float values.  When sentence-level
    predictions are present, also includes ``"sentence_complexity"``
    and ``"weak_cohesion"`` as lists of dicts.
    """
    scores_tensor = output["scores"][0]
    issues_tensor = output["issues"][0]
    diagnoses_tensor = output["diagnoses"][0]

    scores_dict = {
        key: float(scores_tensor[i].item())
        for i, key in enumerate(_SCORE_KEYS)
    }
    issues_dict = {
        key: float(issues_tensor[i].item())
        for i, key in enumerate(_ISSUE_KEYS)
    }
    diagnoses_dict = {
        key: float(diagnoses_tensor[i].item())
        for i, key in enumerate(_DIAGNOSIS_KEYS)
    }

    result = {
        "scores": scores_dict,
        "issues": issues_dict,
        "diagnoses": diagnoses_dict,
    }

    # Sentence-level predictions
    if "sentence_complexity" in output:
        sent_scores = output["sentence_complexity"][0]
        result["sentence_complexity"] = [
            {"sentence": i, "severity": float(s.item())}
            for i, s in enumerate(sent_scores)
            if s.item() > 0.3
        ]
        # Update document-level issue with max of per-sentence predictions
        if sent_scores.numel() > 0:
            result["issues"]["sentence_complexity"] = float(
                sent_scores.max().item()
            )

    if "weak_cohesion" in output:
        pair_scores = output["weak_cohesion"][0]
        result["weak_cohesion"] = [
            {"pair": [i, i + 1], "severity": float(s.item())}
            for i, s in enumerate(pair_scores)
            if s.item() > 0.3
        ]
        # Update document-level issue with max of per-pair predictions
        if pair_scores.numel() > 0:
            result["issues"]["weak_cohesion"] = float(
                pair_scores.max().item()
            )

    return result


def _derive_interventions(diagnoses_dict: dict[str, float]) -> list[dict]:
    """Derive interventions from predicted diagnosis severities.

    Converts diagnosis severities exceeding 0.5 to ``Diagnosis`` objects
    and passes them through ``map_interventions()`` from the existing
    analysis layer.  The resulting ``Intervention`` objects are serialized
    to dicts matching the ``serialize_interpretation()`` format.

    Parameters
    ----------
    diagnoses_dict:
        Mapping of diagnosis type names to predicted severity floats.

    Returns
    -------
    list of intervention dicts, sorted by priority descending.
    """
    active_diagnoses: list[Diagnosis] = []
    for diag_type, severity in diagnoses_dict.items():
        if severity > 0.5:
            active_diagnoses.append(
                Diagnosis(
                    type=diag_type,
                    confidence=severity,
                    severity=severity,
                )
            )

    interventions = map_interventions(active_diagnoses)

    serialized: list[dict] = []
    for interv in interventions:
        serialized.append({
            "type": interv.type,
            "priority": float(interv.priority),
            "target_diagnosis": interv.target_diagnosis,
            "actions": list(interv.actions),
            "exercises": list(interv.exercises),
            "focus_features": list(interv.focus_features),
        })

    return serialized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(
    text: str,
    model_path: str,
    device: str | None = None,
) -> dict:
    """Fast-path inference: tokenize → forward pass → named output dict.

    Loads the model from *model_path*, tokenizes *text*, runs a single
    forward pass, converts tensor outputs to named dicts, and derives
    interventions from predicted diagnoses.

    Parameters
    ----------
    text:
        Raw Hebrew text to analyse.
    model_path:
        Path to a checkpoint directory containing ``config.json``,
        ``model.pt``, and ``tokenizer/``.
    device:
        Explicit device string (e.g. ``"cuda"``, ``"cpu"``).  When *None*,
        auto-detects (CUDA > MPS > CPU).

    Returns
    -------
    dict with keys ``"scores"`` (5 floats), ``"issues"`` (17 floats),
    ``"diagnoses"`` (8 floats), ``"interventions"`` (list of dicts).
    """
    dev = _detect_device(device)
    model, tokenizer = _load_model(model_path, dev)

    encoding = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(dev)
    attention_mask = encoding["attention_mask"].to(dev)

    # Detect sentence boundaries
    sentences = split_into_sentences(text)
    sentence_boundaries = find_token_boundaries(sentences, tokenizer, text)

    with torch.no_grad():
        raw_output = model(
            input_ids, attention_mask,
            sentence_boundaries=[sentence_boundaries],
        )

    named = _predictions_to_dicts(raw_output)
    interventions = _derive_interventions(named["diagnoses"])

    result = {
        "scores": named["scores"],
        "issues": named["issues"],
        "diagnoses": named["diagnoses"],
        "interventions": interventions,
    }

    # Include sentence-level predictions when available
    if "sentence_complexity" in named:
        result["sentence_complexity"] = named["sentence_complexity"]
    if "weak_cohesion" in named:
        result["weak_cohesion"] = named["weak_cohesion"]

    return result


def predict_hybrid(
    text: str,
    model_path: str,
    confidence_threshold: float = 0.7,
    pipeline_config: "PipelineConfig | None" = None,
    device: str | None = None,
) -> dict:
    """Hybrid inference: use model if confident, fall back to pipeline.

    Runs :func:`predict` first.  If the model's confidence (mean of the
    maximum predicted diagnosis severity) exceeds *confidence_threshold*,
    returns the model output.  Otherwise invokes the full pipeline via
    ``run_analysis_pipeline`` and returns its output.

    Parameters
    ----------
    text:
        Raw Hebrew text to analyse.
    model_path:
        Path to a checkpoint directory.
    confidence_threshold:
        Minimum confidence to trust the model output.  Default 0.7.
    pipeline_config:
        Optional ``PipelineConfig`` for the pipeline fallback.  When
        *None*, a default ``PipelineConfig()`` is constructed.
    device:
        Explicit device string.  *None* triggers auto-detection.

    Returns
    -------
    dict with the same structure as :func:`predict`, plus a ``"source"``
    field set to ``"model"`` or ``"pipeline"``.
    """
    model_output = predict(text, model_path, device=device)

    # Confidence = mean of max predicted diagnosis severity
    diag_values = list(model_output["diagnoses"].values())
    confidence = sum(diag_values) / len(diag_values) if diag_values else 0.0

    if confidence > confidence_threshold:
        model_output["source"] = "model"
        return model_output

    # Fall back to full pipeline
    from analysis.analysis_pipeline import run_analysis_pipeline
    from analysis.diagnosis_engine import run_diagnoses
    from analysis.issue_detector import detect_issues
    from analysis.serialization import serialize_interpretation
    from analysis.statistics import FeatureStats, flatten_corpus_json
    from hebrew_profiler.models import PipelineConfig

    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    analysis_input = run_analysis_pipeline(text, pipeline_config)

    issues = detect_issues(
        analysis_input.raw_features,
        analysis_input.sentence_metrics,
        FeatureStats.load_default(),
    )
    diagnoses = run_diagnoses(issues, analysis_input.scores)
    interventions = map_interventions(diagnoses)

    # Build output matching the predict() format
    scores_dict = {k: float(v) if v is not None else 0.0 for k, v in analysis_input.scores.items()}

    issues_dict: dict[str, float] = {k: 0.0 for k in _ISSUE_KEYS}
    for issue in issues:
        if issue.type in issues_dict:
            issues_dict[issue.type] = max(issues_dict[issue.type], issue.severity)

    diagnoses_dict: dict[str, float] = {k: 0.0 for k in _DIAGNOSIS_KEYS}
    for diag in diagnoses:
        if diag.type in diagnoses_dict:
            diagnoses_dict[diag.type] = diag.severity

    serialized_interventions: list[dict] = []
    for interv in interventions:
        serialized_interventions.append({
            "type": interv.type,
            "priority": float(interv.priority),
            "target_diagnosis": interv.target_diagnosis,
            "actions": list(interv.actions),
            "exercises": list(interv.exercises),
            "focus_features": list(interv.focus_features),
        })

    return {
        "scores": scores_dict,
        "issues": issues_dict,
        "diagnoses": diagnoses_dict,
        "interventions": serialized_interventions,
        "source": "pipeline",
    }


def serialize_prediction(output: dict) -> str:
    """Serialize an inference output dict to a JSON string.

    Follows the same conventions as ``serialize_interpretation()`` in
    ``analysis/serialization.py``: uses ``ensure_ascii=False`` to
    preserve Hebrew characters.

    Parameters
    ----------
    output:
        Dict produced by :func:`predict` or :func:`predict_hybrid`.

    Returns
    -------
    JSON string.
    """
    return json.dumps(output, ensure_ascii=False)
