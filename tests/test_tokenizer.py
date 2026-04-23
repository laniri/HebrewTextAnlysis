"""Unit tests for the Hebrew-aware tokenizer."""

from hebrew_profiler.tokenizer import tokenize


class TestTokenizeEmptyInput:
    def test_empty_string_returns_empty_lists(self):
        result = tokenize("")
        assert result.tokens == []
        assert result.character_offsets == []
        assert result.prefix_annotations == []
        assert result.suffix_annotations == []


class TestTokenizeSplitting:
    def test_single_word(self):
        result = tokenize("\u05E9\u05DC\u05D5\u05DD")  # שלום
        assert result.tokens == ["\u05E9\u05DC\u05D5\u05DD"]
        assert result.character_offsets == [(0, 4)]

    def test_two_words_separated_by_space(self):
        text = "\u05E9\u05DC\u05D5\u05DD \u05E2\u05D5\u05DC\u05DD"  # שלום עולם
        result = tokenize(text)
        assert result.tokens == ["\u05E9\u05DC\u05D5\u05DD", "\u05E2\u05D5\u05DC\u05DD"]
        assert len(result.character_offsets) == 2

    def test_punctuation_splits_tokens(self):
        text = "\u05D0\u05D1,\u05D2\u05D3"  # אב,גד
        result = tokenize(text)
        assert result.tokens == ["\u05D0\u05D1", "\u05D2\u05D3"]

    def test_multiple_spaces(self):
        text = "\u05D0\u05D1   \u05D2\u05D3"
        result = tokenize(text)
        assert result.tokens == ["\u05D0\u05D1", "\u05D2\u05D3"]


class TestTokenizeOffsets:
    def test_offsets_round_trip(self):
        text = "\u05E9\u05DC\u05D5\u05DD \u05E2\u05D5\u05DC\u05DD"  # שלום עולם
        result = tokenize(text)
        for token, (start, end) in zip(result.tokens, result.character_offsets):
            assert text[start:end] == token

    def test_offsets_with_punctuation(self):
        text = "\u05D0\u05D1.\u05D2\u05D3"  # אב.גד
        result = tokenize(text)
        for token, (start, end) in zip(result.tokens, result.character_offsets):
            assert text[start:end] == token


class TestPrefixAnnotation:
    def test_vav_prefix_detected(self):
        # ואדם = vav + aleph + dalet + mem → prefix ו
        token = "\u05D5\u05D0\u05D3\u05DD"
        result = tokenize(token)
        assert result.prefix_annotations[0] == ["\u05D5"]

    def test_bet_prefix_detected(self):
        # בבית = bet + bet + yod + tav → prefix ב
        token = "\u05D1\u05D1\u05D9\u05EA"
        result = tokenize(token)
        assert result.prefix_annotations[0] == ["\u05D1"]

    def test_lamed_prefix_detected(self):
        # לבית = lamed + bet + yod + tav → prefix ל
        token = "\u05DC\u05D1\u05D9\u05EA"
        result = tokenize(token)
        assert result.prefix_annotations[0] == ["\u05DC"]

    def test_single_prefix_char_not_annotated(self):
        # A single ו should NOT be annotated as prefix
        result = tokenize("\u05D5")
        assert result.prefix_annotations[0] == []

    def test_prefix_followed_by_non_hebrew_not_annotated(self):
        # ו followed by a digit — not a prefix
        result = tokenize("\u05D53")
        assert result.prefix_annotations[0] == []

    def test_no_prefix_for_non_prefix_start(self):
        # אדם starts with aleph, not a prefix particle
        token = "\u05D0\u05D3\u05DD"
        result = tokenize(token)
        assert result.prefix_annotations[0] == []


class TestSuffixAnnotation:
    def test_single_char_suffix_vav(self):
        # אדמו = aleph + dalet + mem + vav → suffix ו
        token = "\u05D0\u05D3\u05DE\u05D5"
        result = tokenize(token)
        assert result.suffix_annotations[0] == "\u05D5"

    def test_two_char_suffix_nu(self):
        # אדמנו = aleph + dalet + mem + nun + vav → suffix נו
        token = "\u05D0\u05D3\u05DE\u05E0\u05D5"
        result = tokenize(token)
        assert result.suffix_annotations[0] == "\u05E0\u05D5"

    def test_two_char_suffix_khem(self):
        # אדמכם = aleph + dalet + mem + kaf + mem-sofit → suffix כם
        token = "\u05D0\u05D3\u05DE\u05DB\u05DD"
        result = tokenize(token)
        assert result.suffix_annotations[0] == "\u05DB\u05DD"

    def test_suffix_only_token_not_annotated(self):
        # Just ו alone should NOT be annotated as suffix
        result = tokenize("\u05D5")
        assert result.suffix_annotations[0] is None

    def test_two_char_suffix_only_not_annotated(self):
        # Just נו alone should NOT be annotated as suffix
        result = tokenize("\u05E0\u05D5")
        assert result.suffix_annotations[0] is None

    def test_no_suffix_when_no_match(self):
        # אבג = aleph + bet + gimel → no suffix match
        token = "\u05D0\u05D1\u05D2"
        result = tokenize(token)
        assert result.suffix_annotations[0] is None

    def test_longer_suffix_preferred_over_shorter(self):
        # Token ending in הם — should match הם (2-char) not ם (1-char)
        # אבהם = aleph + bet + he + mem
        token = "\u05D0\u05D1\u05D4\u05DD"
        result = tokenize(token)
        assert result.suffix_annotations[0] == "\u05D4\u05DD"


class TestResultListLengths:
    def test_all_lists_same_length(self):
        text = "\u05E9\u05DC\u05D5\u05DD \u05E2\u05D5\u05DC\u05DD \u05D5\u05D0\u05D3\u05DD"
        result = tokenize(text)
        n = len(result.tokens)
        assert len(result.character_offsets) == n
        assert len(result.prefix_annotations) == n
        assert len(result.suffix_annotations) == n


# --- Property-Based Tests ---

from hypothesis import given, settings
import hypothesis.strategies as st

from hebrew_profiler.tokenizer import tokenize


# Feature: hebrew-linguistic-profiling-engine, Property 3: Token-offset round-trip consistency
# **Validates: Requirements 2.1, 2.2**
@settings(max_examples=100)
@given(
    text=st.text(
        alphabet=st.sampled_from(
            # Hebrew letters aleph-tav
            [chr(c) for c in range(0x05D0, 0x05EB)]
            # Spaces and common punctuation
            + list(" .,;:!?")
        ),
        min_size=1,
    )
)
def test_token_offset_round_trip(text):
    """For any text with Hebrew characters, spaces, and punctuation,
    extracting substrings using each token's (start, end) offset pair
    yields exactly the corresponding token surface form."""
    result = tokenize(text)
    assert len(result.tokens) == len(result.character_offsets)
    for token, (start, end) in zip(result.tokens, result.character_offsets):
        assert text[start:end] == token


_PREFIX_CHARS = list("ובלכהמש")
_HEBREW_LETTERS = [chr(c) for c in range(0x05D0, 0x05EB)]


# Feature: hebrew-linguistic-profiling-engine, Property 4: Prefix particle annotation
# **Validates: Requirements 2.3**
@settings(max_examples=100)
@given(
    prefix=st.sampled_from(_PREFIX_CHARS),
    rest=st.text(alphabet=st.sampled_from(_HEBREW_LETTERS), min_size=1, max_size=10),
)
def test_prefix_particle_annotation(prefix, rest):
    """For any Hebrew token that begins with one of the recognized prefix
    particles (ו, ב, ל, כ, ה, מ, ש) followed by at least one additional
    Hebrew letter, the Tokenizer SHALL include that prefix character in the
    token's prefix annotation list."""
    token = prefix + rest
    result = tokenize(token)
    assert len(result.tokens) >= 1
    assert prefix in result.prefix_annotations[0]


# Hebrew letters that do NOT appear in any suffix pronoun pattern.
# This avoids the stem accidentally combining with the suffix to form
# a longer recognised suffix, keeping the property assertion precise.
_SAFE_STEM_LETTERS = [
    "\u05D0",  # א aleph
    "\u05D1",  # ב bet
    "\u05D2",  # ג gimel
    "\u05D3",  # ד dalet
    "\u05D6",  # ז zayin
    "\u05D7",  # ח chet
    "\u05D8",  # ט tet
    "\u05DC",  # ל lamed
    "\u05DE",  # מ mem (non-final)
    "\u05E1",  # ס samekh
    "\u05E2",  # ע ayin
    "\u05E4",  # פ pe
    "\u05E6",  # צ tsadi
    "\u05E7",  # ק qof
    "\u05E8",  # ר resh
    "\u05E9",  # ש shin
    "\u05EA",  # ת tav
]

_SUFFIX_PRONOUN_PATTERNS = [
    "\u05E0\u05D5",  # נו
    "\u05DB\u05DD",  # כם
    "\u05DB\u05DF",  # כן
    "\u05D4\u05DD",  # הם
    "\u05D4\u05DF",  # הן
    "\u05D5",        # ו
    "\u05D4",        # ה
    "\u05DD",        # ם
    "\u05DF",        # ן
    "\u05D9",        # י
]


# Feature: hebrew-linguistic-profiling-engine, Property 5: Suffix pronoun annotation
# **Validates: Requirements 2.4**
@settings(max_examples=100)
@given(
    stem=st.text(
        alphabet=st.sampled_from(_SAFE_STEM_LETTERS),
        min_size=1,
        max_size=8,
    ),
    suffix=st.sampled_from(_SUFFIX_PRONOUN_PATTERNS),
)
def test_suffix_pronoun_annotation(stem, suffix):
    """For any Hebrew token that ends with a recognized suffix pronoun
    pattern, the Tokenizer SHALL annotate the token with the detected
    suffix — provided the stem contains at least one Hebrew letter and
    the full token is not just the suffix pattern alone."""
    token = stem + suffix
    result = tokenize(token)
    assert len(result.tokens) >= 1
    assert result.suffix_annotations[0] == suffix
