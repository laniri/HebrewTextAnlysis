"""Per-sentence metrics extracted from the IR for the probabilistic analysis layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from hebrew_profiler.models import IntermediateRepresentation

if TYPE_CHECKING:
    from analysis.embedder import SentenceEmbedder


@dataclass
class SentenceMetrics:
    index: int              # sentence position in the document (0-based)
    token_count: int        # len(sentence.tokens)
    tree_depth: float       # max depth of dep tree for this sentence (0.0 if no tree)
    lemma_set: frozenset    # lemma set for Jaccard fallback
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # L2-normalised sentence embedding vector; None when embedder is unavailable


def _compute_tree_depth(dep_tree) -> float:
    """BFS from root node to compute max depth. Returns 0.0 if dep_tree is None."""
    if dep_tree is None or not dep_tree.nodes:
        return 0.0

    nodes = dep_tree.nodes

    children: dict[int, list[int]] = {}
    root_id: int | None = None

    for node in nodes:
        children.setdefault(node.id, [])
        if node.head == 0:
            root_id = node.id
        else:
            children.setdefault(node.head, [])
            children[node.head].append(node.id)

    if root_id is None:
        return 0.0

    max_depth = 0
    queue: deque[tuple[int, int]] = deque([(root_id, 0)])
    visited: set[int] = set()

    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        if depth > max_depth:
            max_depth = depth
        for child_id in children.get(node_id, []):
            if child_id not in visited:
                queue.append((child_id, depth + 1))

    return float(max_depth)


def extract_sentence_metrics(
    ir: IntermediateRepresentation,
    sentences: List[str] | None = None,
    embedder: "SentenceEmbedder | None" = None,
) -> List[SentenceMetrics]:
    """Extract per-sentence metrics from the IR.

    For each IRSentence:
    - token_count = len(sentence.tokens)
    - tree_depth = max depth of sentence.dep_tree (BFS from root), 0.0 if dep_tree is None
    - lemma_set = frozenset of lemmas (falling back to surface form when morph is None)
    - embedding = L2-normalised sentence vector when embedder and sentences are provided

    Args:
        ir: The intermediate representation from a single pipeline run.
        sentences: Original sentence texts (from normalized text split). Required
            for embedding — must align 1:1 with ir.sentences.
        embedder: Optional SentenceEmbedder. When provided alongside sentences,
            each SentenceMetrics will have its embedding field populated.

    Returns:
        Exactly len(ir.sentences) SentenceMetrics objects.
    """
    result: List[SentenceMetrics] = []

    for i, sentence in enumerate(ir.sentences):
        token_count = len(sentence.tokens)
        tree_depth = _compute_tree_depth(sentence.dep_tree)
        lemma_set = frozenset(
            token.morph.lemma if token.morph is not None else token.surface
            for token in sentence.tokens
        )
        result.append(SentenceMetrics(
            index=i,
            token_count=token_count,
            tree_depth=tree_depth,
            lemma_set=lemma_set,
        ))

    # Populate embeddings when both embedder and sentence texts are available
    if embedder is not None and sentences is not None and len(sentences) > 0:
        # Align: use only as many sentences as we have IR sentences
        texts = sentences[:len(result)]
        if texts:
            embeddings = embedder.embed(texts)
            for sm, vec in zip(result, embeddings):
                sm.embedding = vec

    return result
