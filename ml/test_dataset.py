# Feature: ml-distillation-layer, Property 7: Dataset tensor shape correctness
# Feature: ml-distillation-layer, Property 9: Tokenization truncation bound

"""Property-based and unit tests for ml/dataset.py.

Tests the dataset module: tensor shape correctness (Property 7),
tokenization truncation bound (Property 9), padding behaviour for short
sequences, and canonical key ordering in tensors.

**Validates: Requirements 8.2, 8.3, 8.4, 14.2, 14.3, 14.4**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer

from ml.dataset import LinguisticDataset
from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS

# ---------------------------------------------------------------------------
# Shared tokenizer — loaded once for all tests (small download, no model).
# ---------------------------------------------------------------------------

_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

# ---------------------------------------------------------------------------
# Strategies for generating valid Training_Records
# ---------------------------------------------------------------------------

_scores_strategy = st.fixed_dictionaries(
    {k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False) for k in _SCORE_KEYS}
)

_issues_strategy = st.fixed_dictionaries(
    {k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False) for k in _ISSUE_KEYS}
)

_diagnoses_strategy = st.fixed_dictionaries(
    {k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False) for k in _DIAGNOSIS_KEYS}
)

_training_record_strategy = st.fixed_dictionaries(
    {
        "text": st.text(min_size=1, max_size=200),
        "scores": _scores_strategy,
        "issues": _issues_strategy,
        "diagnoses": _diagnoses_strategy,
    }
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===========================================================================
# Property 7: Dataset tensor shape correctness
# ===========================================================================

# Feature: ml-distillation-layer, Property 7: Dataset tensor shape correctness
# **Validates: Requirements 14.2, 14.3, 14.4**

MAX_LENGTH = 32  # small value for fast testing


@given(record=_training_record_strategy)
@settings(max_examples=100, deadline=None)
def test_dataset_tensor_shape_correctness(record: dict, tmp_path_factory) -> None:
    """For any valid Training_Record loaded by LinguisticDataset, the returned
    tensors have shapes: input_ids and attention_mask of length max_length,
    scores of shape (5,), issues of shape (17,), diagnoses of shape (8,),
    with label tensor values matching the source dict values in the canonical
    key ordering."""
    tmp_dir = tmp_path_factory.mktemp("data")
    jsonl_path = tmp_dir / "train.jsonl"
    _write_jsonl(jsonl_path, [record])

    dataset = LinguisticDataset(
        jsonl_path=str(jsonl_path),
        tokenizer=_TOKENIZER,
        max_length=MAX_LENGTH,
    )

    assert len(dataset) == 1
    item = dataset[0]

    # --- Shape checks ---
    assert item["input_ids"].shape == (MAX_LENGTH,), (
        f"input_ids shape {item['input_ids'].shape} != ({MAX_LENGTH},)"
    )
    assert item["attention_mask"].shape == (MAX_LENGTH,), (
        f"attention_mask shape {item['attention_mask'].shape} != ({MAX_LENGTH},)"
    )
    assert item["scores"].shape == (5,)
    assert item["issues"].shape == (17,)
    assert item["diagnoses"].shape == (8,)

    # --- Dtype checks ---
    assert item["scores"].dtype == torch.float
    assert item["issues"].dtype == torch.float
    assert item["diagnoses"].dtype == torch.float

    # --- Value correctness: label tensors match source dict in canonical order ---
    for i, key in enumerate(_SCORE_KEYS):
        assert abs(item["scores"][i].item() - record["scores"][key]) < 1e-6, (
            f"scores[{i}] ({key}): expected {record['scores'][key]}, got {item['scores'][i].item()}"
        )

    for i, key in enumerate(_ISSUE_KEYS):
        assert abs(item["issues"][i].item() - record["issues"][key]) < 1e-6, (
            f"issues[{i}] ({key}): expected {record['issues'][key]}, got {item['issues'][i].item()}"
        )

    for i, key in enumerate(_DIAGNOSIS_KEYS):
        assert abs(item["diagnoses"][i].item() - record["diagnoses"][key]) < 1e-6, (
            f"diagnoses[{i}] ({key}): expected {record['diagnoses'][key]}, got {item['diagnoses'][i].item()}"
        )


# ===========================================================================
# Property 9: Tokenization truncation bound
# ===========================================================================

# Feature: ml-distillation-layer, Property 9: Tokenization truncation bound
# **Validates: Requirements 8.2, 8.4**


@given(text=st.text(min_size=1, max_size=5000))
@settings(max_examples=100, deadline=None)
def test_tokenization_truncation_bound(text: str) -> None:
    """For any input text string (including very long strings), the tokenized
    output has at most max_length tokens."""
    max_length = MAX_LENGTH

    encoding = _TOKENIZER(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    # Total length is exactly max_length (padded or truncated)
    assert input_ids.shape == (max_length,), (
        f"input_ids length {input_ids.shape[0]} != {max_length}"
    )
    assert attention_mask.shape == (max_length,), (
        f"attention_mask length {attention_mask.shape[0]} != {max_length}"
    )

    # Number of real (non-padding) tokens is at most max_length
    real_tokens = attention_mask.sum().item()
    assert real_tokens <= max_length


# ===========================================================================
# Unit tests for dataset module (Task 4.4)
# ===========================================================================


class TestDatasetUnit:
    """Unit tests for the LinguisticDataset."""

    @staticmethod
    def _make_record(text: str = "hello world this is a test") -> dict:
        """Return a minimal valid Training_Record."""
        return {
            "text": text,
            "scores": {k: float(i) / 10.0 for i, k in enumerate(_SCORE_KEYS)},
            "issues": {k: float(i) / 20.0 for i, k in enumerate(_ISSUE_KEYS)},
            "diagnoses": {k: float(i) / 10.0 for i, k in enumerate(_DIAGNOSIS_KEYS)},
        }

    # -- Test: padding behaviour for short sequences -----------------------

    def test_padding_short_sequence(self, tmp_path: Path) -> None:
        """Short text is padded to max_length with zeros in attention_mask."""
        record = self._make_record("hi")
        jsonl_path = tmp_path / "train.jsonl"
        _write_jsonl(jsonl_path, [record])

        max_length = 32
        dataset = LinguisticDataset(
            jsonl_path=str(jsonl_path),
            tokenizer=_TOKENIZER,
            max_length=max_length,
        )

        item = dataset[0]

        # input_ids is padded to max_length
        assert item["input_ids"].shape == (max_length,)
        assert item["attention_mask"].shape == (max_length,)

        # "hi" tokenizes to [CLS] hi [SEP] → 3 real tokens, rest is padding
        real_count = item["attention_mask"].sum().item()
        assert real_count < max_length, (
            f"Expected padding but got {real_count} real tokens out of {max_length}"
        )

        # Padding positions should have attention_mask == 0
        padding_positions = item["attention_mask"][int(real_count):]
        assert (padding_positions == 0).all(), "Padding positions should have mask 0"

        # Padding positions in input_ids should be the pad token id
        pad_token_id = _TOKENIZER.pad_token_id
        padding_ids = item["input_ids"][int(real_count):]
        assert (padding_ids == pad_token_id).all(), (
            f"Padding input_ids should be {pad_token_id}"
        )

    # -- Test: canonical key ordering in tensors ---------------------------

    def test_canonical_key_ordering(self, tmp_path: Path) -> None:
        """Tensor values match dict values in the canonical key ordering."""
        # Use distinct values so we can verify ordering
        record = {
            "text": "test ordering",
            "scores": {k: (i + 1) * 0.1 for i, k in enumerate(_SCORE_KEYS)},
            "issues": {k: (i + 1) * 0.05 for i, k in enumerate(_ISSUE_KEYS)},
            "diagnoses": {k: (i + 1) * 0.1 for i, k in enumerate(_DIAGNOSIS_KEYS)},
        }
        jsonl_path = tmp_path / "train.jsonl"
        _write_jsonl(jsonl_path, [record])

        dataset = LinguisticDataset(
            jsonl_path=str(jsonl_path),
            tokenizer=_TOKENIZER,
            max_length=32,
        )

        item = dataset[0]

        # Verify scores ordering
        for i, key in enumerate(_SCORE_KEYS):
            expected = record["scores"][key]
            actual = item["scores"][i].item()
            assert abs(actual - expected) < 1e-6, (
                f"scores[{i}] ({key}): expected {expected}, got {actual}"
            )

        # Verify issues ordering
        for i, key in enumerate(_ISSUE_KEYS):
            expected = record["issues"][key]
            actual = item["issues"][i].item()
            assert abs(actual - expected) < 1e-6, (
                f"issues[{i}] ({key}): expected {expected}, got {actual}"
            )

        # Verify diagnoses ordering
        for i, key in enumerate(_DIAGNOSIS_KEYS):
            expected = record["diagnoses"][key]
            actual = item["diagnoses"][i].item()
            assert abs(actual - expected) < 1e-6, (
                f"diagnoses[{i}] ({key}): expected {expected}, got {actual}"
            )
