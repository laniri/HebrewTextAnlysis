"""Stanza morphological analysis adapter for Hebrew text.

Processes Hebrew text through the Stanza NLP pipeline and parses the output
into structured MorphAnalysis objects with all required morphological fields.
"""

from __future__ import annotations

from typing import Any

from hebrew_profiler.models import MorphAnalysis, StanzaError, StanzaResult
from hebrew_profiler.stanza_setup import ensure_stanza_pipeline


def _parse_feats(feats_str: str | None) -> dict[str, str]:
    """Parse a Universal Dependencies feature string into a dict.

    Example input: ``"Gender=Masc|Number=Sing|Tense=Past|HebBinyan=PAAL"``
    Returns: ``{"Gender": "Masc", "Number": "Sing", "Tense": "Past", "HebBinyan": "PAAL"}``
    """
    if not feats_str:
        return {}
    result: dict[str, str] = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key] = value
    return result


def _extract_mwt_prefixes_and_suffix(
    sentence: Any,
    word: Any,
) -> tuple[list[str], str | None]:
    """Derive prefix and suffix annotations from Stanza MWT expansion.

    When Stanza splits a surface token into multiple sub-words via multi-word
    token (MWT) expansion, the leading sub-words (before the main content word)
    are treated as prefixes and the trailing sub-words (after the main content
    word) are treated as suffixes.

    Returns a ``(prefixes, suffix)`` tuple.
    """
    prefixes: list[str] = []
    suffix: str | None = None

    # Check if this word belongs to a multi-word token
    if not hasattr(sentence, "tokens"):
        return prefixes, suffix

    for token in sentence.tokens:
        # MWT tokens have an id range like (1, 2) covering multiple words
        if not hasattr(token, "id") or not isinstance(token.id, tuple):
            continue
        start_id, end_id = token.id[0], token.id[-1]

        # Check if the current word falls within this MWT range
        word_id = word.id
        if word_id < start_id or word_id > end_id:
            continue

        # This word is part of an MWT expansion.
        # Sub-words before the current word are prefixes;
        # sub-words after are suffixes.
        mwt_word_ids = list(range(start_id, end_id + 1))
        word_idx = mwt_word_ids.index(word_id)

        # Collect prefix texts (sub-words before this one)
        for pid in mwt_word_ids[:word_idx]:
            for w in sentence.words:
                if w.id == pid:
                    prefixes.append(w.text)
                    break

        # Collect suffix text (sub-words after this one, joined)
        suffix_parts: list[str] = []
        for sid in mwt_word_ids[word_idx + 1 :]:
            for w in sentence.words:
                if w.id == sid:
                    suffix_parts.append(w.text)
                    break
        if suffix_parts:
            suffix = "".join(suffix_parts)

        break  # Found the MWT containing this word

    return prefixes, suffix



def _word_to_morph_analysis(sentence: Any, word: Any) -> MorphAnalysis:
    """Convert a single Stanza Word object into a ``MorphAnalysis``."""
    feats = _parse_feats(word.feats)

    gender = feats.get("Gender")
    number = feats.get("Number")
    binyan = feats.get("HebBinyan")
    tense = feats.get("Tense")

    prefixes, suffix = _extract_mwt_prefixes_and_suffix(sentence, word)

    return MorphAnalysis(
        surface=word.text,
        lemma=word.lemma if word.lemma else word.text,
        pos=word.upos if word.upos else "",
        gender=gender,
        number=number,
        prefixes=prefixes,
        suffix=suffix,
        binyan=binyan,
        tense=tense,
        ambiguity_count=1,  # Stanza disambiguates internally
        top_k_analyses=[],  # Stanza returns single best analysis
    )


def analyze_morphology(
    text: str, pipeline: Any = None
) -> StanzaResult | StanzaError:
    """Process text through Stanza Hebrew pipeline, return morphological analyses.

    Each token gets: surface, lemma, POS, gender, number, prefixes,
    suffix, binyan, tense, ambiguity_count.
    Uses a cached pipeline instance if provided, otherwise initializes one.

    Args:
        text: The Hebrew text to analyse.
        pipeline: An optional pre-initialised ``stanza.Pipeline``.  When
            ``None``, :func:`ensure_stanza_pipeline` is called to obtain
            (or create) a cached instance.

    Returns:
        A :class:`StanzaResult` on success, or a :class:`StanzaError`
        dataclass (from ``hebrew_profiler.models``) on failure.
    """
    if not text or not text.strip():
        return StanzaResult(analyses=[])

    try:
        if pipeline is None:
            pipeline = ensure_stanza_pipeline()

        doc = pipeline(text)

        analyses: list[MorphAnalysis] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                analyses.append(_word_to_morph_analysis(sentence, word))

        return StanzaResult(analyses=analyses)

    except Exception as exc:
        error_type = type(exc).__name__
        return StanzaError(
            error_type=error_type,
            message=str(exc),
        )
