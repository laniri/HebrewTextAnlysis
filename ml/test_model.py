# Feature: ml-distillation-layer, Property 6: Model output shape and range invariants

"""Property-based and unit tests for ml/model.py.

Tests the student model: output shape and range invariants (Property 6),
CLS token extraction, fixed key orderings, and model attributes.

**Validates: Requirements 7.2, 7.3, 7.4, 7.5, 9.1, 9.2, 10.1, 10.2,
11.1, 11.2**
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import BertConfig, BertModel

from ml.model import (
    LinguisticModel,
    _DIAGNOSIS_KEYS,
    _ISSUE_KEYS,
    _SCORE_KEYS,
)

# ---------------------------------------------------------------------------
# Helper: build a LinguisticModel with a tiny randomly-initialized encoder
# so tests run fast without downloading DictaBERT.
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
    model.sentence_head = nn.Linear(hidden, 1)
    model.pair_head = nn.Linear(hidden * 2, 1)
    return model


# ===========================================================================
# Property 6: Model output shape and range invariants
# ===========================================================================

# Feature: ml-distillation-layer, Property 6: Model output shape and range invariants
# **Validates: Requirements 9.1, 9.2, 10.1, 10.2, 11.1, 11.2**


# Module-level model instance shared across property test examples to avoid
# re-creating the encoder on every Hypothesis iteration.
_SHARED_MODEL = _make_tiny_model()
_SHARED_MODEL.eval()


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=100, deadline=None)
def test_model_output_shape_and_range(batch_size: int, seq_len: int) -> None:
    """For any batch of input tensors with shape (B, seq_len) where B >= 1
    and seq_len <= 512, the model's forward pass produces outputs where
    scores has shape (B, 5), issues has shape (B, 17), diagnoses has shape
    (B, 8), and all output values are in [0, 1]."""
    model = _SHARED_MODEL

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Check keys
    assert set(output.keys()) == {"scores", "issues", "diagnoses"}

    # Check shapes
    assert output["scores"].shape == (batch_size, 5)
    assert output["issues"].shape == (batch_size, 17)
    assert output["diagnoses"].shape == (batch_size, 8)

    # Check all values in [0, 1]
    for key in ("scores", "issues", "diagnoses"):
        tensor = output[key]
        assert tensor.min().item() >= 0.0, f"{key} has values < 0"
        assert tensor.max().item() <= 1.0, f"{key} has values > 1"


# ===========================================================================
# Unit tests for student model (Task 3.3)
# ===========================================================================


class TestModelUnit:
    """Unit tests for the LinguisticModel and canonical key orderings."""

    # -- Test: CLS token extraction ----------------------------------------

    def test_cls_token_extraction(self) -> None:
        """Verify the model uses position 0 of last_hidden_state (CLS token).

        We hook into the linear heads to capture the input they receive and
        confirm it matches encoder_output.last_hidden_state[:, 0, :].
        """
        model = _make_tiny_model()
        model.eval()

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Run encoder directly to get the expected CLS representation
        with torch.no_grad():
            encoder_output = model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            expected_cls = encoder_output.last_hidden_state[:, 0, :]

        # Capture what the heads actually receive by hooking into scores_head
        captured_input = []

        def hook_fn(module, inp, out):
            captured_input.append(inp[0].clone())

        handle = model.scores_head.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        assert len(captured_input) == 1
        actual_cls = captured_input[0]
        assert torch.allclose(actual_cls, expected_cls, atol=1e-6), (
            "scores_head input does not match CLS token (position 0)"
        )

    # -- Test: fixed key orderings match specification ---------------------

    def test_score_keys_count_and_content(self) -> None:
        """_SCORE_KEYS has exactly 5 items matching the specification."""
        assert len(_SCORE_KEYS) == 5
        expected = ["difficulty", "style", "fluency", "cohesion", "complexity"]
        assert _SCORE_KEYS == expected

    def test_issue_keys_count_and_content(self) -> None:
        """_ISSUE_KEYS has exactly 17 items matching the specification."""
        assert len(_ISSUE_KEYS) == 17
        expected = [
            "agreement_errors",
            "morphological_ambiguity",
            "low_morphological_diversity",
            "sentence_complexity",
            "dependency_spread",
            "excessive_branching",
            "low_lexical_diversity",
            "rare_word_overuse",
            "low_content_density",
            "sentence_length_variability",
            "punctuation_issues",
            "fragmentation",
            "weak_cohesion",
            "missing_connectives",
            "pronoun_ambiguity",
            "structural_inconsistency",
            "sentence_progression_drift",
        ]
        assert _ISSUE_KEYS == expected

    def test_diagnosis_keys_count_and_content(self) -> None:
        """_DIAGNOSIS_KEYS has exactly 8 items matching the specification."""
        assert len(_DIAGNOSIS_KEYS) == 8
        expected = [
            "low_lexical_diversity",
            "pronoun_overuse",
            "low_cohesion",
            "sentence_over_complexity",
            "structural_inconsistency",
            "low_morphological_richness",
            "fragmented_writing",
            "punctuation_deficiency",
        ]
        assert _DIAGNOSIS_KEYS == expected

    # -- Test: LinguisticModel has expected attributes ----------------------

    def test_model_has_expected_attributes(self) -> None:
        """LinguisticModel has encoder, scores_head, issues_head, diagnoses_head, sentence_head, pair_head."""
        model = _make_tiny_model()

        assert hasattr(model, "encoder")
        assert hasattr(model, "scores_head")
        assert hasattr(model, "issues_head")
        assert hasattr(model, "diagnoses_head")
        assert hasattr(model, "sentence_head")
        assert hasattr(model, "pair_head")

        # Verify head output dimensions
        assert model.scores_head.out_features == 5
        assert model.issues_head.out_features == 17
        assert model.diagnoses_head.out_features == 8
        assert model.sentence_head.out_features == 1
        assert model.pair_head.out_features == 1

        # Verify heads are nn.Linear instances
        assert isinstance(model.scores_head, nn.Linear)
        assert isinstance(model.issues_head, nn.Linear)
        assert isinstance(model.diagnoses_head, nn.Linear)
        assert isinstance(model.sentence_head, nn.Linear)
        assert isinstance(model.pair_head, nn.Linear)
