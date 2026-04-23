"""Student model definition for the ML Distillation Layer (Layer 6).

Defines the canonical key orderings used across the entire ``ml/`` package
and the ``LinguisticModel`` multi-task transformer.

The three constant lists below establish the fixed output order for scores,
issues, and diagnoses.  Every module that reads or writes these vectors
imports the orderings from here so that a single source of truth exists.

Requirements implemented: 7.1–7.5, 9.1–9.3, 10.1–10.3, 11.1–11.3, 27.1, 27.3, 27.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Canonical key orderings — imported by export, dataset, inference, and
# disagreement modules.  Do NOT reorder without updating all consumers.
# ---------------------------------------------------------------------------

_SCORE_KEYS: list[str] = [
    "difficulty",
    "style",
    "fluency",
    "cohesion",
    "complexity",
]

_ISSUE_KEYS: list[str] = [
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

_DIAGNOSIS_KEYS: list[str] = [
    "low_lexical_diversity",
    "pronoun_overuse",
    "low_cohesion",
    "sentence_over_complexity",
    "structural_inconsistency",
    "low_morphological_richness",
    "fragmented_writing",
    "punctuation_deficiency",
]


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------


class LinguisticModel(nn.Module):
    """Multi-task transformer for predicting linguistic scores, issues, and diagnoses.

    Architecture:
        Hebrew encoder (e.g. DictaBERT) → [CLS] pooling → three linear heads
        with sigmoid activation producing values in [0, 1].

    Parameters
    ----------
    encoder_name:
        HuggingFace model identifier for the Hebrew encoder.
    num_scores:
        Number of score outputs (default 5).
    num_issues:
        Number of issue type outputs (default 17).
    num_diagnoses:
        Number of diagnosis type outputs (default 8).
    """

    def __init__(
        self,
        encoder_name: str = "dicta-il/dictabert",
        num_scores: int = 5,
        num_issues: int = 17,
        num_diagnoses: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size: int = self.encoder.config.hidden_size

        self.scores_head = nn.Linear(hidden_size, num_scores)
        self.issues_head = nn.Linear(hidden_size, num_issues)
        self.diagnoses_head = nn.Linear(hidden_size, num_diagnoses)

        # Sentence-level heads
        self.sentence_head = nn.Linear(hidden_size, 1)
        self.pair_head = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: list[list[tuple[int, int]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and return sigmoid-activated predictions.

        Parameters
        ----------
        input_ids:
            Token IDs, shape ``(B, seq_len)``.
        attention_mask:
            Attention mask, shape ``(B, seq_len)``.
        sentence_boundaries:
            Optional list (length *B*) of lists of ``(start, end)`` token
            index tuples per sentence.  When provided, the model also
            returns per-sentence complexity and per-pair cohesion
            predictions.

        Returns
        -------
        dict with keys ``"scores"`` *(B, 5)*, ``"issues"`` *(B, 17)*,
        ``"diagnoses"`` *(B, 8)* — all values in [0, 1].
        When *sentence_boundaries* is provided, also includes
        ``"sentence_complexity"`` and ``"weak_cohesion"`` as lists of
        variable-length tensors (one per batch item).
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_repr = encoder_output.last_hidden_state[:, 0, :]

        output = {
            "scores": torch.sigmoid(self.scores_head(cls_repr)),
            "issues": torch.sigmoid(self.issues_head(cls_repr)),
            "diagnoses": torch.sigmoid(self.diagnoses_head(cls_repr)),
        }

        if sentence_boundaries is not None:
            token_embeddings = encoder_output.last_hidden_state
            sentence_preds: list[torch.Tensor] = []
            pair_preds: list[torch.Tensor] = []

            for b in range(token_embeddings.shape[0]):
                boundaries = sentence_boundaries[b]

                # Mean-pool tokens per sentence
                sent_embeds: list[torch.Tensor] = []
                for start, end in boundaries:
                    if end > start:
                        sent_embed = token_embeddings[b, start:end, :].mean(dim=0)
                    else:
                        sent_embed = torch.zeros(
                            token_embeddings.shape[-1],
                            device=token_embeddings.device,
                        )
                    sent_embeds.append(sent_embed)

                if sent_embeds:
                    sent_matrix = torch.stack(sent_embeds)
                    sent_scores = torch.sigmoid(
                        self.sentence_head(sent_matrix)
                    ).squeeze(-1)
                else:
                    sent_scores = torch.tensor(
                        [], device=token_embeddings.device
                    )
                sentence_preds.append(sent_scores)

                # Adjacent pair concatenation
                if len(sent_embeds) >= 2:
                    pairs = []
                    for i in range(len(sent_embeds) - 1):
                        pair = torch.cat([sent_embeds[i], sent_embeds[i + 1]])
                        pairs.append(pair)
                    pair_matrix = torch.stack(pairs)
                    pair_scores = torch.sigmoid(
                        self.pair_head(pair_matrix)
                    ).squeeze(-1)
                else:
                    pair_scores = torch.tensor(
                        [], device=token_embeddings.device
                    )
                pair_preds.append(pair_scores)

            output["sentence_complexity"] = sentence_preds
            output["weak_cohesion"] = pair_preds

        return output
