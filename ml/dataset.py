"""Dataset module for the ML Distillation Layer (Layer 6).

Loads JSONL training data produced by :mod:`ml.export` and converts each
record into PyTorch tensors suitable for the :class:`ml.model.LinguisticModel`.

The canonical key orderings for scores, issues, and diagnoses are imported
from :mod:`ml.model` so that tensor positions always match the model's
output heads.

Requirements implemented: 8.1, 8.2, 8.3, 8.4, 14.1, 14.2, 14.3, 14.4, 27.1, 27.6.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ml.model import _DIAGNOSIS_KEYS, _ISSUE_KEYS, _SCORE_KEYS
from ml.sentence_utils import find_token_boundaries, split_into_sentences


class LinguisticDataset(Dataset):
    """PyTorch dataset that reads JSONL training records and yields tensors.

    Each record is a JSON line with keys ``"text"``, ``"scores"`` (5 floats),
    ``"issues"`` (17 floats), and ``"diagnoses"`` (8 floats).  Optionally
    includes ``"sentence_complexities"`` and ``"cohesion_pairs"`` for
    sentence-level training.

    The dataset tokenises the text at ``__getitem__`` time and converts the
    label dicts to float tensors using the canonical key orderings from
    :mod:`ml.model`.

    Parameters
    ----------
    jsonl_path:
        Path to a JSONL file produced by :func:`ml.export.export_training_data`.
    tokenizer:
        A HuggingFace tokenizer matching the encoder used by
        :class:`ml.model.LinguisticModel`.
    max_length:
        Maximum token sequence length.  Sequences are truncated from the
        right and padded to this length.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            self.records: list[dict] = [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]

        # Tokenize text
        encoding = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze the batch dimension (1, seq_len) → (seq_len,)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Convert label dicts to float tensors in canonical key order
        # Scores use MSE (no range constraint), but issues/diagnoses use BCE
        # which requires [0, 1] — clamp to be safe.
        scores = torch.tensor(
            [record["scores"][k] for k in _SCORE_KEYS],
            dtype=torch.float,
        )
        issues = torch.tensor(
            [max(0.0, min(1.0, record["issues"][k])) for k in _ISSUE_KEYS],
            dtype=torch.float,
        )
        diagnoses = torch.tensor(
            [max(0.0, min(1.0, record["diagnoses"][k])) for k in _DIAGNOSIS_KEYS],
            dtype=torch.float,
        )

        result: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scores": scores,
            "issues": issues,
            "diagnoses": diagnoses,
        }

        # Sentence-level data
        text = record["text"]
        sentences = split_into_sentences(text)
        boundaries = find_token_boundaries(
            sentences, self.tokenizer, text, max_length=self.max_length
        )
        result["sentence_boundaries"] = boundaries

        # Per-sentence and per-pair labels (backward compatible — default to
        # empty lists when fields are absent from older training records)
        sent_complexities = record.get("sentence_complexities", [])
        coh_pairs = record.get("cohesion_pairs", [])

        # Trim to match the number of boundaries (sentences may be truncated)
        num_sents = len(boundaries)
        if sent_complexities:
            sent_complexities = sent_complexities[:num_sents]
        else:
            sent_complexities = [0.0] * num_sents

        num_pairs = max(0, num_sents - 1)
        if coh_pairs:
            coh_pairs = coh_pairs[:num_pairs]
        else:
            coh_pairs = [0.0] * num_pairs

        result["sentence_complexities"] = torch.tensor(
            [max(0.0, min(1.0, v)) for v in sent_complexities], dtype=torch.float
        )
        result["cohesion_pairs"] = torch.tensor(
            [max(0.0, min(1.0, v)) for v in coh_pairs], dtype=torch.float
        )

        return result


def linguistic_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length sentence data.

    Fixed-size tensors (``input_ids``, ``attention_mask``, ``scores``,
    ``issues``, ``diagnoses``) are stacked normally.  Variable-length
    fields (``sentence_boundaries``, ``sentence_complexities``,
    ``cohesion_pairs``) are kept as lists.

    Parameters
    ----------
    batch:
        List of dicts from :meth:`LinguisticDataset.__getitem__`.

    Returns
    -------
    dict suitable for passing to the model and loss function.
    """
    collated: dict = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "scores": torch.stack([item["scores"] for item in batch]),
        "issues": torch.stack([item["issues"] for item in batch]),
        "diagnoses": torch.stack([item["diagnoses"] for item in batch]),
        # Variable-length — keep as lists
        "sentence_boundaries": [item["sentence_boundaries"] for item in batch],
        "sentence_complexities": [item["sentence_complexities"] for item in batch],
        "cohesion_pairs": [item["cohesion_pairs"] for item in batch],
    }
    return collated
