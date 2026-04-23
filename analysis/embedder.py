"""Sentence embedding module for semantic cohesion detection.

Wraps sentence-transformers with lazy loading and a module-level singleton
so the model is loaded once per process.

Falls back gracefully when sentence-transformers is not installed — callers
check the return value of get_embedder() for None before using it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    pass

# Default multilingual model — supports Hebrew, ~420 MB download on first use.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_singleton: "SentenceEmbedder | None" = None
_singleton_model: str | None = None


def get_embedder(model_name: str = DEFAULT_MODEL) -> "SentenceEmbedder | None":
    """Return the singleton SentenceEmbedder, or None if unavailable.

    On first call the model is downloaded and loaded (~420 MB, one-time).
    Subsequent calls return the cached instance immediately.

    Returns None when sentence-transformers is not installed, allowing
    callers to fall back to Jaccard-based cohesion detection.
    """
    global _singleton, _singleton_model
    if _singleton is not None and _singleton_model == model_name:
        return _singleton
    try:
        _singleton = SentenceEmbedder(model_name)
        _singleton_model = model_name
        return _singleton
    except ImportError:
        return None


class SentenceEmbedder:
    """Thin wrapper around a SentenceTransformer model.

    All vectors are L2-normalised at encode time, so cosine similarity
    reduces to a plain dot product.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences into L2-normalised float32 vectors.

        Args:
            sentences: List of plain-text sentences.

        Returns:
            np.ndarray of shape (N, D) where D is the model's embedding
            dimension (768 for mpnet-base).
        """
        vecs = self._model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )
        return np.array(vecs, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two L2-normalised vectors (dot product).

        Args:
            a: L2-normalised embedding vector.
            b: L2-normalised embedding vector.

        Returns:
            Float in [-1.0, 1.0]; 1.0 = identical direction.
        """
        return float(np.dot(a, b))
