"""Property-based tests for the normalizer module."""

from hypothesis import given, settings
import hypothesis.strategies as st

from hebrew_profiler.normalizer import normalize


# Feature: hebrew-linguistic-profiling-engine, Property 1: Normalization idempotence
# **Validates: Requirements 1.1**
@settings(max_examples=100)
@given(text=st.text())
def test_normalization_idempotence(text):
    """For any Unicode string, applying normalize twice produces the same result as once."""
    result = normalize(text)
    assert normalize(result.normalized_text) == result


# Non-standard punctuation characters that must be replaced by the normalizer.
NON_STANDARD_CHARS = [
    "\u201c",  # left double quotation mark
    "\u201d",  # right double quotation mark
    "\u2018",  # left single quotation mark
    "\u2019",  # right single quotation mark
    "\u2014",  # em dash
    "\u2013",  # en dash
    "\u2026",  # horizontal ellipsis
    "\u00A0",  # non-breaking space
]

# Strategy: generate text that contains at least some non-standard characters
# mixed with arbitrary text.
_non_standard_char_strategy = st.sampled_from(NON_STANDARD_CHARS)

_mixed_text_strategy = st.lists(
    st.one_of(
        st.text(min_size=0, max_size=10),
        _non_standard_char_strategy,
    ),
    min_size=1,
    max_size=20,
).map("".join).filter(
    lambda s: any(c in s for c in NON_STANDARD_CHARS)
)


# Feature: hebrew-linguistic-profiling-engine, Property 2: Punctuation standardization completeness
# **Validates: Requirements 1.2**
@settings(max_examples=100)
@given(text=_mixed_text_strategy)
def test_punctuation_standardization_completeness(text):
    """For any string containing non-standard punctuation characters,
    the Normalizer output SHALL contain none of those non-standard characters."""
    result = normalize(text)
    for char in NON_STANDARD_CHARS:
        assert char not in result.normalized_text, (
            f"Non-standard character U+{ord(char):04X} found in output"
        )
