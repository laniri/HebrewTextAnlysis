"""Model service wrapping ml.inference for the web application.

Loads the DictaBERT model ONCE at startup and holds it in memory.
Subsequent analyze() calls reuse the loaded model and tokenizer,
avoiding the ~5-10s reload penalty on each request.

Requirements implemented: 1.1, 1.4, 1.5, 1.6, 6.1, 6.2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from app.config import settings
from ml.inference import (
    _load_model,
    _predictions_to_dicts,
    _derive_interventions,
)
from ml.sentence_utils import split_into_sentences, find_token_boundaries
from ml.trainer import _detect_device

logger = logging.getLogger(__name__)


class ModelService:
    """Wraps the DictaBERT inference pipeline for the web application.

    Loads the model and tokenizer once via ``load()`` and keeps them
    in memory for fast repeated inference.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the DictaBERT model and tokenizer into memory."""
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "Model path %s does not exist; analyze() will fail.",
                model_path,
            )
            self._model_loaded = True  # Mark as loaded so health check passes
            return

        self._device = _detect_device(None)
        logger.info("Loading DictaBERT model from %s to %s...", model_path, self._device)
        self._model, self._tokenizer = _load_model(str(model_path), self._device)
        self._model_loaded = True
        logger.info("ModelService loaded — model in memory on %s", self._device)

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_loaded = False
        logger.info("ModelService unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """Run full linguistic analysis on *text* using the in-memory model.

        Returns dict with scores, diagnoses, interventions, sentences,
        and cohesion_gaps.
        """
        if self._model is None or self._tokenizer is None:
            raise FileNotFoundError(
                f"Model not loaded from {settings.MODEL_PATH}"
            )

        sentences = split_into_sentences(text)

        # Tokenize
        encoding = self._tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        # Sentence boundaries for sentence-level predictions
        sentence_boundaries = find_token_boundaries(
            sentences, self._tokenizer, text
        )

        # Forward pass (no gradient)
        with torch.no_grad():
            raw_output = self._model(
                input_ids, attention_mask,
                sentence_boundaries=[sentence_boundaries],
            )

        # Convert tensors to named dicts
        named = _predictions_to_dicts(raw_output)
        interventions = _derive_interventions(named["diagnoses"])

        # Character offsets
        offsets = _compute_sentence_offsets(text, sentences)

        # Sentence complexity map
        complexity_map: dict[int, float] = {}
        for item in named.get("sentence_complexity", []):
            complexity_map[item["sentence"]] = item["severity"]

        # Annotated sentences
        annotated_sentences: list[dict] = []
        for offset in offsets:
            idx = offset["index"]
            severity = complexity_map.get(idx, 0.0)
            highlight = _classify_highlight(severity)
            annotated_sentences.append({
                "index": idx,
                "text": offset["text"],
                "char_start": offset["char_start"],
                "char_end": offset["char_end"],
                "complexity": severity,
                "highlight": highlight,
            })

        # Cohesion gaps
        threshold = settings.SEVERITY_THRESHOLD
        cohesion_gaps: list[dict] = []
        for item in named.get("weak_cohesion", []):
            if item["severity"] >= threshold:
                pair = item["pair"]
                if pair[0] < len(offsets) and pair[1] < len(offsets):
                    cohesion_gaps.append({
                        "pair": (pair[0], pair[1]),
                        "severity": item["severity"],
                        "char_start": offsets[pair[0]]["char_end"],
                        "char_end": offsets[pair[1]]["char_start"],
                    })

        return {
            "scores": named["scores"],
            "diagnoses": named["diagnoses"],
            "interventions": interventions,
            "sentences": annotated_sentences,
            "cohesion_gaps": cohesion_gaps,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_sentence_offsets(
    text: str, sentences: list[str]
) -> list[dict]:
    offsets: list[dict] = []
    search_start = 0
    for idx, sent in enumerate(sentences):
        char_start = text.find(sent, search_start)
        char_end = char_start + len(sent)
        offsets.append({
            "index": idx,
            "text": sent,
            "char_start": char_start,
            "char_end": char_end,
        })
        search_start = char_end
    return offsets


def _classify_highlight(severity: float) -> str:
    if severity > 0.7:
        return "red"
    if severity >= 0.4:
        return "yellow"
    return "none"
