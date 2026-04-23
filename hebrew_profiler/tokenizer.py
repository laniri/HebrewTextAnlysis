"""Hebrew-aware tokenization module for the Hebrew Linguistic Profiling Engine.

Splits normalized text into surface tokens with character offsets and annotates
Hebrew prefix particles and suffix pronoun patterns.
"""

from __future__ import annotations

import re

from hebrew_profiler.models import TokenizationResult

# Hebrew letter Unicode range: aleph (\u05D0) through tav (\u05EA)
_HEBREW_LETTER = r"[\u05D0-\u05EA]"

# Tokenization regex: match sequences of non-whitespace, non-punctuation chars.
# We split on Unicode whitespace and common punctuation boundaries.
_TOKEN_RE = re.compile(
    r"[^\s\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
    r"\u00A0-\u00BF\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F\uFE10-\uFE6F"
    r"\uFF01-\uFF60]+"
)

# Prefix particles: vav, bet, lamed, kaf, he, mem, shin
_PREFIX_PARTICLES = frozenset("\u05D5\u05D1\u05DC\u05DB\u05D4\u05DE\u05E9")

# Suffix pronoun patterns, ordered longest first so two-char suffixes match before
# single-char ones.
_SUFFIX_PATTERNS: list[str] = [
    "\u05E0\u05D5",  # נו (nu)
    "\u05DB\u05DD",  # כם (khem)
    "\u05DB\u05DF",  # כן (khen)
    "\u05D4\u05DD",  # הם (hem)
    "\u05D4\u05DF",  # הן (hen)
    "\u05D5",        # ו (o/av)
    "\u05D4",        # ה (ah)
    "\u05DD",        # ם (am)
    "\u05DF",        # ן (an)
    "\u05D9",        # י (i)
]

_HEBREW_LETTER_RE = re.compile(_HEBREW_LETTER)


def _detect_prefixes(token: str) -> list[str]:
    """Detect Hebrew prefix particles at the start of a token.

    A prefix is recognized when the token starts with a known prefix particle
    AND is followed by at least one more Hebrew letter. A single-letter token
    that is just the prefix character itself is NOT annotated.
    """
    prefixes: list[str] = []
    if len(token) < 2:
        return prefixes

    if token[0] in _PREFIX_PARTICLES and _HEBREW_LETTER_RE.match(token[1]):
        prefixes.append(token[0])

    return prefixes


_SUFFIX_SET = frozenset(_SUFFIX_PATTERNS)


def _detect_suffix(token: str) -> str | None:
    """Detect a Hebrew suffix pronoun pattern at the end of a token.

    A suffix is recognized when the token ends with a known suffix pattern
    AND has at least one Hebrew letter before the suffix. A token that is
    ONLY a recognized suffix pattern (of any length) is NOT annotated.
    """
    # If the entire token is a recognized suffix pattern, skip annotation.
    if token in _SUFFIX_SET:
        return None

    for suffix in _SUFFIX_PATTERNS:
        slen = len(suffix)
        if len(token) <= slen:
            continue
        if token.endswith(suffix):
            # Check that there is at least one Hebrew letter before the suffix
            preceding = token[:-slen]
            if _HEBREW_LETTER_RE.search(preceding):
                return suffix
    return None


def tokenize(normalized_text: str) -> TokenizationResult:
    """Split normalized text into surface tokens with character offsets.

    Annotates Hebrew prefix particles and suffix pronouns for each token.

    Args:
        normalized_text: Text that has already been through the normalizer.

    Returns:
        TokenizationResult with tokens, character_offsets,
        prefix_annotations, and suffix_annotations.
    """
    if not normalized_text:
        return TokenizationResult(
            tokens=[],
            character_offsets=[],
            prefix_annotations=[],
            suffix_annotations=[],
        )

    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    prefix_annotations: list[list[str]] = []
    suffix_annotations: list[str | None] = []

    for match in _TOKEN_RE.finditer(normalized_text):
        token = match.group()
        start = match.start()
        end = match.end()

        tokens.append(token)
        offsets.append((start, end))
        prefix_annotations.append(_detect_prefixes(token))
        suffix_annotations.append(_detect_suffix(token))

    return TokenizationResult(
        tokens=tokens,
        character_offsets=offsets,
        prefix_annotations=prefix_annotations,
        suffix_annotations=suffix_annotations,
    )
