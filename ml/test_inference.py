# Feature: ml-distillation-layer, Property 10: Inference output round-trip serialization
# Feature: ml-distillation-layer, Property 11: Diagnosis threshold conversion for interventions
# Feature: ml-distillation-layer, Property 12: Hybrid mode confidence-based routing

"""Property-based and unit tests for ml/inference.py.

Tests the inference module: output round-trip serialization (Property 10),
diagnosis threshold conversion for interventions (Property 11), hybrid mode
confidence-based routing (Property 12), and unit tests for internal helpers.

**Validates: Requirements 16.3, 16.5, 17.1, 17.2, 18.1, 18.2, 18.3,
18.5, 19.1, 19.3, 19.4**
"""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.diagnosis_models import Diagnosis
from analysis.intervention_mapper import INTERVENTION_MAP, map_interventions
from ml.inference import _derive_interventions, _predictions_to_dicts, serialize_prediction
from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_INTERVENTION_TYPES = ["vocabulary_expansion", "pronoun_clarification",
                       "sentence_simplification", "cohesion_improvement"]

# Strategy for a single intervention dict matching the serialized format
_intervention_strategy = st.fixed_dictionaries({
    "type": st.sampled_from(_INTERVENTION_TYPES),
    "priority": st.floats(min_value=0.0, max_value=1.0,
                          allow_nan=False, allow_infinity=False),
    "target_diagnosis": st.sampled_from(list(_DIAGNOSIS_KEYS)),
    "actions": st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=3),
    "exercises": st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=3),
    "focus_features": st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=3),
})

# Strategy for a full inference output dict
_inference_output_strategy = st.fixed_dictionaries({
    "scores": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False)
        for k in _SCORE_KEYS
    }),
    "issues": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False)
        for k in _ISSUE_KEYS
    }),
    "diagnoses": st.fixed_dictionaries({
        k: st.floats(min_value=0.0, max_value=1.0,
                     allow_nan=False, allow_infinity=False)
        for k in _DIAGNOSIS_KEYS
    }),
    "interventions": st.lists(_intervention_strategy, max_size=4),
})

# Strategy for diagnosis severity dicts
_diagnosis_severities_strategy = st.fixed_dictionaries({
    k: st.floats(min_value=0.0, max_value=1.0,
                 allow_nan=False, allow_infinity=False)
    for k in _DIAGNOSIS_KEYS
})


# ===========================================================================
# Property 10: Inference output round-trip serialization
# ===========================================================================

# Feature: ml-distillation-layer, Property 10: Inference output round-trip serialization
# **Validates: Requirements 19.1, 19.4**


@given(output=_inference_output_strategy)
@settings(max_examples=100)
def test_inference_output_round_trip_serialization(output: dict) -> None:
    """For any valid inference output dict (containing scores, issues,
    diagnoses, and interventions), serializing to JSON with ensure_ascii=False
    and deserializing produces an equivalent dict."""
    serialized = serialize_prediction(output)
    restored = json.loads(serialized)

    # Top-level keys preserved
    assert set(restored.keys()) == set(output.keys())

    # Scores round-trip
    for key in _SCORE_KEYS:
        assert abs(restored["scores"][key] - output["scores"][key]) < 1e-12

    # Issues round-trip
    for key in _ISSUE_KEYS:
        assert abs(restored["issues"][key] - output["issues"][key]) < 1e-12

    # Diagnoses round-trip
    for key in _DIAGNOSIS_KEYS:
        assert abs(restored["diagnoses"][key] - output["diagnoses"][key]) < 1e-12

    # Interventions round-trip
    assert len(restored["interventions"]) == len(output["interventions"])
    for orig, rest in zip(output["interventions"], restored["interventions"]):
        assert rest["type"] == orig["type"]
        assert abs(rest["priority"] - orig["priority"]) < 1e-12
        assert rest["target_diagnosis"] == orig["target_diagnosis"]
        assert rest["actions"] == orig["actions"]
        assert rest["exercises"] == orig["exercises"]
        assert rest["focus_features"] == orig["focus_features"]


# ===========================================================================
# Property 11: Diagnosis threshold conversion for interventions
# ===========================================================================

# Feature: ml-distillation-layer, Property 11: Diagnosis threshold conversion for interventions
# **Validates: Requirements 17.1, 17.2**


@given(severities=_diagnosis_severities_strategy)
@settings(max_examples=100)
def test_diagnosis_threshold_conversion_for_interventions(
    severities: dict[str, float],
) -> None:
    """For any dict of predicted diagnosis severities, converting to Diagnosis
    objects using threshold 0.5 produces Diagnosis objects only for entries
    where severity > 0.5, and passing through map_interventions() produces
    interventions consistent with INTERVENTION_MAP."""
    result = _derive_interventions(severities)

    # Determine which diagnosis types should be active (severity > 0.5)
    active_types = {k for k, v in severities.items() if v > 0.5}

    # Only active types that exist in INTERVENTION_MAP can produce interventions
    expected_types = {k for k in active_types if k in INTERVENTION_MAP}

    # Each expected type should have exactly one intervention
    result_target_diagnoses = {iv["target_diagnosis"] for iv in result}
    assert result_target_diagnoses == expected_types, (
        f"Expected interventions for {expected_types}, "
        f"got interventions for {result_target_diagnoses}"
    )

    # Each intervention's type must match INTERVENTION_MAP
    for iv in result:
        target = iv["target_diagnosis"]
        expected_type = INTERVENTION_MAP[target]["type"]
        assert iv["type"] == expected_type, (
            f"Intervention for '{target}' has type '{iv['type']}', "
            f"expected '{expected_type}'"
        )

    # Each intervention's priority must equal the diagnosis severity
    for iv in result:
        target = iv["target_diagnosis"]
        expected_priority = severities[target]
        assert abs(iv["priority"] - expected_priority) < 1e-12, (
            f"Intervention for '{target}' has priority {iv['priority']}, "
            f"expected {expected_priority}"
        )

    # Interventions should be sorted by priority descending
    priorities = [iv["priority"] for iv in result]
    assert priorities == sorted(priorities, reverse=True), (
        "Interventions are not sorted by priority descending"
    )

    # No intervention should exist for types with severity <= 0.5
    inactive_types = {k for k, v in severities.items() if v <= 0.5}
    for iv in result:
        assert iv["target_diagnosis"] not in inactive_types


# ===========================================================================
# Property 12: Hybrid mode confidence-based routing
# ===========================================================================

# Feature: ml-distillation-layer, Property 12: Hybrid mode confidence-based routing
# **Validates: Requirements 18.1, 18.2, 18.3**


def _make_model_output(diagnoses_dict: dict[str, float]) -> dict:
    """Build a minimal model output dict with the given diagnoses."""
    return {
        "scores": {k: 0.5 for k in _SCORE_KEYS},
        "issues": {k: 0.0 for k in _ISSUE_KEYS},
        "diagnoses": diagnoses_dict,
        "interventions": [],
    }


# Strategy that generates diagnosis dicts where the mean is ABOVE a threshold.
# We generate floats in (0.7, 1.0] to ensure the mean strictly exceeds 0.7.
# Using exclude_min=True ensures no value is exactly 0.7.
_high_confidence_strategy = st.fixed_dictionaries({
    k: st.floats(min_value=0.7, max_value=1.0,
                 allow_nan=False, allow_infinity=False,
                 exclude_min=True)
    for k in _DIAGNOSIS_KEYS
})

# Strategy that generates diagnosis dicts where the mean is AT or BELOW a threshold.
# We generate floats in [0.0, 0.7] to ensure the mean stays at or below 0.7.
_low_confidence_strategy = st.fixed_dictionaries({
    k: st.floats(min_value=0.0, max_value=0.7,
                 allow_nan=False, allow_infinity=False)
    for k in _DIAGNOSIS_KEYS
})


@given(diagnoses=_high_confidence_strategy)
@settings(max_examples=100, deadline=None)
def test_hybrid_mode_routes_to_model_when_confident(
    diagnoses: dict[str, float],
) -> None:
    """For any model prediction where confidence (mean of diagnosis severities)
    exceeds threshold, hybrid mode returns source='model'."""
    model_output = _make_model_output(diagnoses)

    with patch("ml.inference.predict", return_value=model_output):
        from ml.inference import predict_hybrid

        result = predict_hybrid(
            text="טקסט לבדיקה",
            model_path="/fake/path",
            confidence_threshold=0.7,
            device="cpu",
        )

    assert result["source"] == "model"
    assert result["diagnoses"] == diagnoses


@given(diagnoses=_low_confidence_strategy)
@settings(max_examples=100, deadline=None)
def test_hybrid_mode_routes_to_pipeline_when_not_confident(
    diagnoses: dict[str, float],
) -> None:
    """For any model prediction where confidence (mean of diagnosis severities)
    does not exceed threshold, hybrid mode invokes the pipeline fallback and
    returns source='pipeline'."""
    model_output = _make_model_output(diagnoses)

    # Build a fake pipeline result
    pipeline_result = {
        "scores": {k: 0.3 for k in _SCORE_KEYS},
        "issues": {k: 0.0 for k in _ISSUE_KEYS},
        "diagnoses": {k: 0.0 for k in _DIAGNOSIS_KEYS},
        "interventions": [],
        "source": "pipeline",
    }

    # Mock predict() to return low-confidence output, and mock the pipeline
    # fallback modules that predict_hybrid imports when confidence is low
    with patch("ml.inference.predict", return_value=model_output), \
         patch("analysis.analysis_pipeline.run_analysis_pipeline") as mock_pipeline, \
         patch("analysis.issue_detector.detect_issues", return_value=[]), \
         patch("analysis.diagnosis_engine.run_diagnoses", return_value=[]), \
         patch("analysis.statistics.FeatureStats") as mock_stats_cls:

        # Configure the pipeline mock to return a minimal analysis result
        mock_analysis = MagicMock()
        mock_analysis.raw_features = {}
        mock_analysis.sentence_metrics = []
        mock_analysis.scores = {k: 0.3 for k in _SCORE_KEYS}
        mock_pipeline.return_value = mock_analysis
        mock_stats_cls.load_default.return_value = MagicMock()

        from ml.inference import predict_hybrid

        result = predict_hybrid(
            text="טקסט לבדיקה",
            model_path="/fake/path",
            confidence_threshold=0.7,
            device="cpu",
        )

    assert result["source"] == "pipeline"


# ===========================================================================
# Unit tests for inference module (Task 7.5)
# ===========================================================================


class TestInferenceUnit:
    """Unit tests for inference module helpers."""

    # -- Test: _derive_interventions with known diagnosis severities --------

    def test_derive_interventions_known_severities(self) -> None:
        """_derive_interventions with known severities produces expected
        interventions matching INTERVENTION_MAP."""
        diagnoses = {k: 0.0 for k in _DIAGNOSIS_KEYS}
        # Set two diagnoses above threshold
        diagnoses["sentence_over_complexity"] = 0.78
        diagnoses["low_cohesion"] = 0.65

        result = _derive_interventions(diagnoses)

        # Should produce exactly 2 interventions
        assert len(result) == 2

        # Sorted by priority descending: sentence_over_complexity (0.78) first
        assert result[0]["target_diagnosis"] == "sentence_over_complexity"
        assert result[0]["type"] == "sentence_simplification"
        assert abs(result[0]["priority"] - 0.78) < 1e-12

        assert result[1]["target_diagnosis"] == "low_cohesion"
        assert result[1]["type"] == "cohesion_improvement"
        assert abs(result[1]["priority"] - 0.65) < 1e-12

    def test_derive_interventions_none_above_threshold(self) -> None:
        """_derive_interventions with all severities <= 0.5 returns empty list."""
        diagnoses = {k: 0.3 for k in _DIAGNOSIS_KEYS}
        result = _derive_interventions(diagnoses)
        assert result == []

    def test_derive_interventions_boundary_at_threshold(self) -> None:
        """_derive_interventions with severity exactly 0.5 does NOT produce
        an intervention (threshold is strictly > 0.5)."""
        diagnoses = {k: 0.0 for k in _DIAGNOSIS_KEYS}
        diagnoses["low_cohesion"] = 0.5  # exactly at threshold
        result = _derive_interventions(diagnoses)
        assert result == []

    def test_derive_interventions_all_above_threshold(self) -> None:
        """_derive_interventions with all severities > 0.5 produces
        interventions for all 8 diagnosis types in INTERVENTION_MAP."""
        diagnoses = {k: 0.8 for k in _DIAGNOSIS_KEYS}
        result = _derive_interventions(diagnoses)
        # All 8 diagnosis types are in INTERVENTION_MAP
        assert len(result) == 8
        result_targets = {iv["target_diagnosis"] for iv in result}
        assert result_targets == set(_DIAGNOSIS_KEYS)

    # -- Test: _predictions_to_dicts with known tensors --------------------

    def test_predictions_to_dicts_known_tensors(self) -> None:
        """_predictions_to_dicts converts tensor outputs to named dicts
        with correct key-value mapping."""
        scores_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        issues_vals = [float(i) / 17.0 for i in range(17)]
        diagnoses_vals = [float(i) / 8.0 for i in range(8)]

        output = {
            "scores": torch.tensor([scores_vals]),
            "issues": torch.tensor([issues_vals]),
            "diagnoses": torch.tensor([diagnoses_vals]),
        }

        result = _predictions_to_dicts(output)

        # Check scores mapping
        assert set(result["scores"].keys()) == set(_SCORE_KEYS)
        for i, key in enumerate(_SCORE_KEYS):
            assert abs(result["scores"][key] - scores_vals[i]) < 1e-6

        # Check issues mapping
        assert set(result["issues"].keys()) == set(_ISSUE_KEYS)
        for i, key in enumerate(_ISSUE_KEYS):
            assert abs(result["issues"][key] - issues_vals[i]) < 1e-6

        # Check diagnoses mapping
        assert set(result["diagnoses"].keys()) == set(_DIAGNOSIS_KEYS)
        for i, key in enumerate(_DIAGNOSIS_KEYS):
            assert abs(result["diagnoses"][key] - diagnoses_vals[i]) < 1e-6

    # -- Test: serialize_prediction with ensure_ascii=False and Hebrew -----

    def test_serialize_prediction_hebrew(self) -> None:
        """serialize_prediction preserves Hebrew characters (no \\uXXXX)."""
        output = {
            "scores": {k: 0.5 for k in _SCORE_KEYS},
            "issues": {k: 0.0 for k in _ISSUE_KEYS},
            "diagnoses": {k: 0.0 for k in _DIAGNOSIS_KEYS},
            "interventions": [
                {
                    "type": "vocabulary_expansion",
                    "priority": 0.8,
                    "target_diagnosis": "low_lexical_diversity",
                    "actions": ["הרחבת אוצר מילים"],
                    "exercises": ["תרגול מילים חדשות"],
                    "focus_features": ["lemma_diversity"],
                }
            ],
        }

        serialized = serialize_prediction(output)

        # Hebrew characters should appear directly
        assert "הרחבת" in serialized
        assert "תרגול" in serialized
        # Should not contain unicode escapes for Hebrew
        assert "\\u05d4" not in serialized

        # Round-trip should preserve content
        restored = json.loads(serialized)
        assert restored["interventions"][0]["actions"][0] == "הרחבת אוצר מילים"

    def test_serialize_prediction_round_trip(self) -> None:
        """serialize_prediction output can be deserialized back to equivalent dict."""
        output = {
            "scores": {"difficulty": 0.72, "style": 0.21, "fluency": 0.41,
                       "cohesion": 0.33, "complexity": 0.59},
            "issues": {k: 0.0 for k in _ISSUE_KEYS},
            "diagnoses": {k: 0.0 for k in _DIAGNOSIS_KEYS},
            "interventions": [],
        }

        serialized = serialize_prediction(output)
        restored = json.loads(serialized)

        assert restored == output
