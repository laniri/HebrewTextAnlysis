"""Sentence boundary utilities for the ML Distillation Layer (Layer 6).

Provides sentence splitting and token-boundary mapping shared by
:mod:`ml.dataset` (training) and :mod:`ml.inference` (prediction).

Requirements implemented: 27.1, 27.3.
"""

from __future__ import annotations

import re


def split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentences using sentence-ending punctuation (.!?).

    Uses ``re.split(r'(?<=[.!?])\\s+', text)`` and filters empty strings.

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    List of non-empty sentence strings.
    """
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in parts if s]


def find_token_boundaries(
    sentences: list[str],
    tokenizer,
    full_text: str,
    max_length: int = 512,
) -> list[tuple[int, int]]:
    """Map sentence character spans to token index spans.

    Tokenizes *full_text* with ``return_offsets_mapping=True`` and maps
    each sentence's character range to the corresponding token positions.
    Sentences whose tokens fall entirely beyond the *max_length* truncation
    boundary are excluded from the result.

    Parameters
    ----------
    sentences:
        List of sentence strings (from :func:`split_into_sentences`).
    tokenizer:
        A HuggingFace tokenizer that supports ``return_offsets_mapping``.
    full_text:
        The original unsplit text.
    max_length:
        Maximum token sequence length (matches the model's truncation).

    Returns
    -------
    List of ``(start_token_idx, end_token_idx)`` tuples, one per sentence
    that fits within the truncation window.
    """
    encoding = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"]  # list of (char_start, char_end)
    num_tokens = len(offset_mapping)

    # Build character-start → sentence boundaries
    boundaries: list[tuple[int, int]] = []
    search_start = 0
    for sent in sentences:
        char_start = full_text.find(sent, search_start)
        if char_start == -1:
            # Sentence not found in text — skip
            continue
        char_end = char_start + len(sent)
        search_start = char_end

        # Find the first token whose character span overlaps with this sentence
        tok_start = None
        tok_end = None
        for tok_idx in range(num_tokens):
            t_start, t_end = offset_mapping[tok_idx]
            if t_start == 0 and t_end == 0:
                # Special token ([CLS], [SEP], [PAD]) — skip
                continue
            if t_end > char_start and t_start < char_end:
                if tok_start is None:
                    tok_start = tok_idx
                tok_end = tok_idx + 1

        if tok_start is not None and tok_end is not None:
            boundaries.append((tok_start, tok_end))

    return boundaries
