"""Unit tests for the normalizer module."""

import unicodedata

from hebrew_profiler.models import NormalizationResult
from hebrew_profiler.normalizer import normalize


class TestNormalizeEmptyInput:
    def test_empty_string_returns_empty_result(self):
        result = normalize("")
        assert result == NormalizationResult(normalized_text="")

    def test_returns_normalization_result_type(self):
        result = normalize("")
        assert isinstance(result, NormalizationResult)


class TestNormalizeNFKC:
    def test_nfkc_applied(self):
        # U+FB01 (fi ligature) decomposes to "fi" under NFKC
        result = normalize("\ufb01")
        assert result.normalized_text == "fi"

    def test_hebrew_text_preserved(self):
        text = "שלום עולם"
        result = normalize(text)
        assert result.normalized_text == "שלום עולם"


class TestNormalizePunctuation:
    def test_left_double_quote(self):
        result = normalize("\u201cהלו\u201d")
        assert result.normalized_text == '"הלו"'

    def test_right_double_quote(self):
        result = normalize("word\u201d")
        assert result.normalized_text == 'word"'

    def test_left_single_quote(self):
        result = normalize("\u2018test\u2019")
        assert result.normalized_text == "'test'"

    def test_em_dash(self):
        result = normalize("a\u2014b")
        assert result.normalized_text == "a-b"

    def test_en_dash(self):
        result = normalize("a\u2013b")
        assert result.normalized_text == "a-b"

    def test_ellipsis(self):
        result = normalize("wait\u2026")
        assert result.normalized_text == "wait..."

    def test_non_breaking_space(self):
        result = normalize("hello\u00A0world")
        assert result.normalized_text == "hello world"

    def test_multiple_replacements(self):
        text = "\u201cשלום\u201d \u2014 \u2026"
        result = normalize(text)
        assert result.normalized_text == '"שלום" - ...'

    def test_no_special_chars_unchanged(self):
        text = "plain ascii text 123"
        result = normalize(text)
        assert result.normalized_text == text


class TestNormalizeIdempotence:
    def test_double_normalize_same_result(self):
        text = "\u201cשלום\u201d \u2014 \u2026 \u00A0"
        first = normalize(text)
        second = normalize(first.normalized_text)
        assert first.normalized_text == second.normalized_text

    def test_already_normalized_text(self):
        text = '"hello" - ... plain'
        result = normalize(text)
        assert result.normalized_text == text
