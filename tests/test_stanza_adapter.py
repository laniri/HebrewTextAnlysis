"""Tests for the Stanza morphological analysis adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hebrew_profiler.models import MorphAnalysis, StanzaError, StanzaResult
from hebrew_profiler.stanza_adapter import (
    _extract_mwt_prefixes_and_suffix,
    _parse_feats,
    _word_to_morph_analysis,
    analyze_morphology,
)


# ---------------------------------------------------------------------------
# Helpers – lightweight fakes that mimic Stanza's object model
# ---------------------------------------------------------------------------

def _make_word(
    word_id: int = 1,
    text: str = "מילה",
    lemma: str = "מילה",
    upos: str = "NOUN",
    feats: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(id=word_id, text=text, lemma=lemma, upos=upos, feats=feats)


def _make_token(token_id: int | tuple, text: str = "") -> SimpleNamespace:
    """Create a fake Stanza Token.  ``token_id`` may be an int or a tuple for MWT."""
    if isinstance(token_id, int):
        tid = (token_id,)
    else:
        tid = token_id
    return SimpleNamespace(id=tid, text=text)


def _make_sentence(words: list, tokens: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(words=words, tokens=tokens or [])


def _make_doc(sentences: list) -> SimpleNamespace:
    return SimpleNamespace(sentences=sentences)


# ---------------------------------------------------------------------------
# _parse_feats
# ---------------------------------------------------------------------------

class TestParseFeats:
    def test_none_returns_empty(self):
        assert _parse_feats(None) == {}

    def test_empty_string_returns_empty(self):
        assert _parse_feats("") == {}

    def test_single_feature(self):
        assert _parse_feats("Gender=Masc") == {"Gender": "Masc"}

    def test_multiple_features(self):
        result = _parse_feats("Gender=Masc|Number=Sing|Tense=Past|HebBinyan=PAAL")
        assert result == {
            "Gender": "Masc",
            "Number": "Sing",
            "Tense": "Past",
            "HebBinyan": "PAAL",
        }

    def test_feature_with_equals_in_value(self):
        result = _parse_feats("Key=Val=ue")
        assert result == {"Key": "Val=ue"}


# ---------------------------------------------------------------------------
# _extract_mwt_prefixes_and_suffix
# ---------------------------------------------------------------------------

class TestExtractMwtPrefixesAndSuffix:
    def test_no_mwt_returns_empty(self):
        word = _make_word(word_id=1)
        sentence = _make_sentence(words=[word], tokens=[_make_token(1)])
        prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, word)
        assert prefixes == []
        assert suffix is None

    def test_mwt_prefix_detected(self):
        """When a token is split into (prefix, main), the prefix is captured."""
        w1 = _make_word(word_id=1, text="ו", lemma="ו", upos="CCONJ")
        w2 = _make_word(word_id=2, text="הלך", lemma="הלך", upos="VERB")
        mwt_token = _make_token((1, 2), text="והלך")
        sentence = _make_sentence(words=[w1, w2], tokens=[mwt_token])

        # For the main word (w2), w1 should appear as a prefix
        prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, w2)
        assert prefixes == ["ו"]
        assert suffix is None

    def test_mwt_suffix_detected(self):
        """When a token is split into (main, suffix), the suffix is captured."""
        w1 = _make_word(word_id=1, text="ספר", lemma="ספר", upos="NOUN")
        w2 = _make_word(word_id=2, text="ו", lemma="הוא", upos="PRON")
        mwt_token = _make_token((1, 2), text="ספרו")
        sentence = _make_sentence(words=[w1, w2], tokens=[mwt_token])

        # For the main word (w1), w2 should appear as a suffix
        prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, w1)
        assert prefixes == []
        assert suffix == "ו"

    def test_mwt_prefix_and_suffix(self):
        """Three-way split: prefix + main + suffix."""
        w1 = _make_word(word_id=1, text="ו", lemma="ו", upos="CCONJ")
        w2 = _make_word(word_id=2, text="ספר", lemma="ספר", upos="NOUN")
        w3 = _make_word(word_id=3, text="ו", lemma="הוא", upos="PRON")
        mwt_token = _make_token((1, 3), text="וספרו")
        sentence = _make_sentence(words=[w1, w2, w3], tokens=[mwt_token])

        prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, w2)
        assert prefixes == ["ו"]
        assert suffix == "ו"

    def test_sentence_without_tokens_attr(self):
        """Gracefully handle sentence objects without a tokens attribute."""
        word = _make_word(word_id=1)
        sentence = SimpleNamespace(words=[word])  # no 'tokens'
        prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, word)
        assert prefixes == []
        assert suffix is None


# ---------------------------------------------------------------------------
# _word_to_morph_analysis
# ---------------------------------------------------------------------------

class TestWordToMorphAnalysis:
    def test_basic_word(self):
        word = _make_word(
            word_id=1,
            text="הלך",
            lemma="הלך",
            upos="VERB",
            feats="Gender=Masc|Number=Sing|Tense=Past|HebBinyan=PAAL",
        )
        sentence = _make_sentence(words=[word], tokens=[_make_token(1)])
        result = _word_to_morph_analysis(sentence, word)

        assert isinstance(result, MorphAnalysis)
        assert result.surface == "הלך"
        assert result.lemma == "הלך"
        assert result.pos == "VERB"
        assert result.gender == "Masc"
        assert result.number == "Sing"
        assert result.tense == "Past"
        assert result.binyan == "PAAL"
        assert result.ambiguity_count == 1
        assert result.top_k_analyses == []

    def test_word_without_feats(self):
        word = _make_word(word_id=1, text="את", lemma="את", upos="ADP", feats=None)
        sentence = _make_sentence(words=[word], tokens=[_make_token(1)])
        result = _word_to_morph_analysis(sentence, word)

        assert result.gender is None
        assert result.number is None
        assert result.binyan is None
        assert result.tense is None

    def test_word_with_none_lemma(self):
        word = _make_word(word_id=1, text="xyz", upos="X")
        word.lemma = None
        sentence = _make_sentence(words=[word], tokens=[_make_token(1)])
        result = _word_to_morph_analysis(sentence, word)
        # Falls back to surface text
        assert result.lemma == "xyz"


# ---------------------------------------------------------------------------
# analyze_morphology
# ---------------------------------------------------------------------------

class TestAnalyzeMorphology:
    def test_empty_string_returns_empty_result(self):
        result = analyze_morphology("")
        assert isinstance(result, StanzaResult)
        assert result.analyses == []

    def test_whitespace_only_returns_empty_result(self):
        result = analyze_morphology("   ")
        assert isinstance(result, StanzaResult)
        assert result.analyses == []

    def test_success_with_provided_pipeline(self):
        """Passing a fake pipeline produces a StanzaResult."""
        w1 = _make_word(word_id=1, text="שלום", lemma="שלום", upos="NOUN", feats="Gender=Masc|Number=Sing")
        sentence = _make_sentence(words=[w1], tokens=[_make_token(1)])
        doc = _make_doc(sentences=[sentence])

        fake_pipeline = lambda text: doc  # noqa: E731
        result = analyze_morphology("שלום", pipeline=fake_pipeline)

        assert isinstance(result, StanzaResult)
        assert len(result.analyses) == 1
        assert result.analyses[0].surface == "שלום"
        assert result.analyses[0].gender == "Masc"

    def test_multiple_sentences(self):
        w1 = _make_word(word_id=1, text="הוא", lemma="הוא", upos="PRON")
        w2 = _make_word(word_id=2, text="הלך", lemma="הלך", upos="VERB", feats="Tense=Past")
        s1 = _make_sentence(words=[w1, w2], tokens=[_make_token(1), _make_token(2)])

        w3 = _make_word(word_id=1, text="היא", lemma="היא", upos="PRON")
        s2 = _make_sentence(words=[w3], tokens=[_make_token(1)])

        doc = _make_doc(sentences=[s1, s2])
        fake_pipeline = lambda text: doc  # noqa: E731

        result = analyze_morphology("הוא הלך. היא.", pipeline=fake_pipeline)
        assert isinstance(result, StanzaResult)
        assert len(result.analyses) == 3

    def test_pipeline_exception_returns_stanza_error(self):
        """When the pipeline raises, we get a StanzaError dataclass."""
        def failing_pipeline(text):
            raise RuntimeError("model crashed")

        result = analyze_morphology("טקסט", pipeline=failing_pipeline)
        assert isinstance(result, StanzaError)
        assert result.error_type == "RuntimeError"
        assert "model crashed" in result.message

    @patch("hebrew_profiler.stanza_adapter.ensure_stanza_pipeline")
    def test_uses_ensure_pipeline_when_none(self, mock_ensure):
        """When no pipeline is provided, ensure_stanza_pipeline is called."""
        w = _make_word(word_id=1, text="בדיקה", lemma="בדיקה", upos="NOUN")
        s = _make_sentence(words=[w], tokens=[_make_token(1)])
        doc = _make_doc(sentences=[s])
        mock_ensure.return_value = lambda text: doc

        result = analyze_morphology("בדיקה")
        mock_ensure.assert_called_once()
        assert isinstance(result, StanzaResult)
        assert len(result.analyses) == 1

    @patch("hebrew_profiler.stanza_adapter.ensure_stanza_pipeline")
    def test_setup_error_returns_stanza_error(self, mock_ensure):
        """StanzaSetupError from ensure_stanza_pipeline is caught."""
        from hebrew_profiler.errors import StanzaSetupError
        mock_ensure.side_effect = StanzaSetupError("model not found")

        result = analyze_morphology("טקסט")
        assert isinstance(result, StanzaError)
        assert result.error_type == "StanzaSetupError"

    def test_mwt_expansion_captured(self):
        """MWT expansion produces prefix annotations on the main word."""
        w_prefix = _make_word(word_id=1, text="ב", lemma="ב", upos="ADP")
        w_main = _make_word(word_id=2, text="בית", lemma="בית", upos="NOUN", feats="Gender=Masc|Number=Sing")
        mwt_tok = _make_token((1, 2), text="בבית")
        sentence = _make_sentence(words=[w_prefix, w_main], tokens=[mwt_tok])
        doc = _make_doc(sentences=[sentence])

        fake_pipeline = lambda text: doc  # noqa: E731
        result = analyze_morphology("בבית", pipeline=fake_pipeline)

        assert isinstance(result, StanzaResult)
        # The main word should have the prefix captured
        main_analysis = result.analyses[1]  # w_main is second word
        assert main_analysis.prefixes == ["ב"]
        assert main_analysis.surface == "בית"


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Valid values for Stanza UD feature fields
_GENDERS = ["Masc", "Fem"]
_NUMBERS = ["Sing", "Plur", "Dual"]
_TENSES = ["Past", "Pres", "Fut", "Imp"]
_BINYANIM = ["PAAL", "PIEL", "HIFIL", "HITPAEL", "NIFAL", "PUAL", "HUFAL"]
_UPOS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET", "CCONJ", "SCONJ", "NUM", "PUNCT", "X"]


def _build_feats_string(
    gender: str | None,
    number: str | None,
    tense: str | None,
    binyan: str | None,
) -> str | None:
    """Build a UD-style feature string from optional field values."""
    parts: list[str] = []
    if gender is not None:
        parts.append(f"Gender={gender}")
    if number is not None:
        parts.append(f"Number={number}")
    if tense is not None:
        parts.append(f"Tense={tense}")
    if binyan is not None:
        parts.append(f"HebBinyan={binyan}")
    return "|".join(parts) if parts else None


@st.composite
def stanza_word_strategy(draw: st.DrawFn, word_id: int = 1):
    """Generate a fake Stanza word with random morphological features."""
    text = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=0x0590, max_codepoint=0x05FF),
        min_size=1,
        max_size=10,
    ))
    lemma = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",), min_codepoint=0x0590, max_codepoint=0x05FF),
        min_size=1,
        max_size=10,
    ))
    upos = draw(st.sampled_from(_UPOS_TAGS))
    gender = draw(st.one_of(st.none(), st.sampled_from(_GENDERS)))
    number = draw(st.one_of(st.none(), st.sampled_from(_NUMBERS)))
    tense = draw(st.one_of(st.none(), st.sampled_from(_TENSES)))
    binyan = draw(st.one_of(st.none(), st.sampled_from(_BINYANIM)))

    feats = _build_feats_string(gender, number, tense, binyan)
    word = _make_word(word_id=word_id, text=text, lemma=lemma, upos=upos, feats=feats)
    return word, gender, number, tense, binyan


@st.composite
def stanza_document_strategy(draw: st.DrawFn):
    """Generate a fake Stanza Document with 1-3 sentences, each with 1-5 words."""
    num_sentences = draw(st.integers(min_value=1, max_value=3))
    sentences = []
    expected: list[tuple] = []  # (word, gender, number, tense, binyan) per word

    for _ in range(num_sentences):
        num_words = draw(st.integers(min_value=1, max_value=5))
        words = []
        tokens = []
        for wid in range(1, num_words + 1):
            word, gender, number, tense, binyan = draw(stanza_word_strategy(word_id=wid))
            words.append(word)
            tokens.append(_make_token(wid))
            expected.append((word, gender, number, tense, binyan))
        sentences.append(_make_sentence(words=words, tokens=tokens))

    doc = _make_doc(sentences=sentences)
    return doc, expected


class TestStanzaResponseParsingCompleteness:
    """Property 6: Stanza response parsing completeness.

    **Validates: Requirements 4.2, 4.3**

    For any well-formed Stanza Document output containing morphological
    analyses for tokens, the Stanza_Adapter SHALL extract all required
    fields (surface, lemma, POS, gender, number, prefixes, suffix, binyan,
    tense) from the Stanza feature string.
    """

    @given(data=stanza_document_strategy())
    @settings(max_examples=150)
    def test_all_required_fields_extracted(self, data):
        """Every generated word's morphological features are faithfully
        extracted into the corresponding MorphAnalysis object."""
        doc, expected = data

        fake_pipeline = lambda text: doc  # noqa: E731
        result = analyze_morphology("dummy text", pipeline=fake_pipeline)

        assert isinstance(result, StanzaResult)
        assert len(result.analyses) == len(expected)

        for analysis, (word, exp_gender, exp_number, exp_tense, exp_binyan) in zip(
            result.analyses, expected
        ):
            # Surface and lemma must match the word object
            assert analysis.surface == word.text
            assert analysis.lemma == word.lemma

            # POS must match
            assert analysis.pos == word.upos

            # Gender, number, tense, binyan extracted from feature string
            assert analysis.gender == exp_gender
            assert analysis.number == exp_number
            assert analysis.tense == exp_tense
            assert analysis.binyan == exp_binyan

            # Prefixes and suffix must be present as lists / None
            assert isinstance(analysis.prefixes, list)
            # suffix is either None or a string
            assert analysis.suffix is None or isinstance(analysis.suffix, str)
