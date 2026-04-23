"""Text normalization module for the Hebrew Linguistic Profiling Engine.

Applies Unicode NFKC normalization and standardizes typographic punctuation
to ASCII equivalents.
"""

import unicodedata

from hebrew_profiler.models import NormalizationResult

# Punctuation mapping: typographic characters → ASCII equivalents.
# str.translate handles single-char → single-char replacements.
# Ellipsis (U+2026 → "...") is handled separately via str.replace
# since str.translate cannot map one char to multiple chars.
_PUNCTUATION_MAP = str.maketrans(
    {
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u00A0": " ",   # non-breaking space
    }
)


def normalize(text: str) -> NormalizationResult:
    """Apply NFKC normalization and punctuation standardization.

    Args:
        text: Raw input text (may contain typographic punctuation and
              non-standard Unicode forms).

    Returns:
        NormalizationResult with the normalized_text field set.
    """
    if not text:
        return NormalizationResult(normalized_text="")

    # Step 1: Unicode NFKC normalization
    result = unicodedata.normalize("NFKC", text)

    # Step 2: Translate single-char punctuation replacements
    result = result.translate(_PUNCTUATION_MAP)

    # Step 3: Replace ellipsis (1 char → 3 chars, can't use translate)
    result = result.replace("\u2026", "...")

    return NormalizationResult(normalized_text=result)
