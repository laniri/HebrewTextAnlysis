"""Intermediate Representation builder for the Hebrew Linguistic Profiling Engine.

Combines tokenization, Stanza morphological analysis, and YAP syntactic parsing
outputs into a unified IntermediateRepresentation. Handles partial data gracefully
when upstream modules fail.
"""

from __future__ import annotations

from hebrew_profiler.models import (
    DepTreeNode,
    IntermediateRepresentation,
    IRSentence,
    IRToken,
    MorphAnalysis,
    NormalizationResult,
    SentenceTree,
    StanzaError,
    StanzaResult,
    TokenizationResult,
    YAPError,
    YAPResult,
)


def _get_morph(analyses: list[MorphAnalysis], index: int) -> MorphAnalysis | None:
    """Safely retrieve a MorphAnalysis by global token index."""
    if 0 <= index < len(analyses):
        return analyses[index]
    return None


def _get_token_surface(tokenization: TokenizationResult, index: int) -> str:
    """Get surface form from tokenizer output, falling back to empty string."""
    if 0 <= index < len(tokenization.tokens):
        return tokenization.tokens[index]
    return ""


def _get_token_offset(
    tokenization: TokenizationResult, index: int
) -> tuple[int, int]:
    """Get character offset from tokenizer output, falling back to (0, 0)."""
    if 0 <= index < len(tokenization.character_offsets):
        return tokenization.character_offsets[index]
    return (0, 0)


def _get_token_prefixes(tokenization: TokenizationResult, index: int) -> list[str]:
    """Get prefix annotations from tokenizer output."""
    if 0 <= index < len(tokenization.prefix_annotations):
        return tokenization.prefix_annotations[index]
    return []


def _get_token_suffix(tokenization: TokenizationResult, index: int) -> str | None:
    """Get suffix annotation from tokenizer output."""
    if 0 <= index < len(tokenization.suffix_annotations):
        return tokenization.suffix_annotations[index]
    return None


def _build_both_succeed(
    tokenization: TokenizationResult,
    stanza_result: StanzaResult,
    yap_result: YAPResult,
) -> list[IRSentence]:
    """Build IR sentences when both Stanza and YAP succeed.

    Aligns Stanza morphological analyses with YAP dependency tree nodes
    by matching on global token position across sentences.
    """
    sentences: list[IRSentence] = []
    global_idx = 0

    for sentence_tree in yap_result.sentences:
        tokens: list[IRToken] = []
        for node in sentence_tree.nodes:
            morph = _get_morph(stanza_result.analyses, global_idx)
            surface = node.form
            offset = _get_token_offset(tokenization, global_idx)
            prefixes = _get_token_prefixes(tokenization, global_idx)
            suffix = _get_token_suffix(tokenization, global_idx)

            tokens.append(IRToken(
                surface=surface,
                offset=offset,
                morph=morph,
                dep_node=node,
                prefixes=prefixes,
                suffix=suffix,
            ))
            global_idx += 1

        sentences.append(IRSentence(
            tokens=tokens,
            dep_tree=sentence_tree,
        ))

    return sentences


def _build_stanza_only(
    tokenization: TokenizationResult,
    stanza_result: StanzaResult,
) -> list[IRSentence]:
    """Build IR when only Stanza succeeds (YAP failed).

    Creates a single IRSentence with all tokens, morph data present,
    dep_node=None, dep_tree=None.
    """
    tokens: list[IRToken] = []
    num_tokens = max(len(tokenization.tokens), len(stanza_result.analyses))

    for i in range(num_tokens):
        morph = _get_morph(stanza_result.analyses, i)
        surface = _get_token_surface(tokenization, i)
        if not surface and morph is not None:
            surface = morph.surface
        offset = _get_token_offset(tokenization, i)
        prefixes = _get_token_prefixes(tokenization, i)
        suffix = _get_token_suffix(tokenization, i)

        tokens.append(IRToken(
            surface=surface,
            offset=offset,
            morph=morph,
            dep_node=None,
            prefixes=prefixes,
            suffix=suffix,
        ))

    return [IRSentence(tokens=tokens, dep_tree=None)] if tokens else []


def _build_yap_only(
    tokenization: TokenizationResult,
    yap_result: YAPResult,
) -> list[IRSentence]:
    """Build IR when only YAP succeeds (Stanza failed).

    Creates IRSentences from YAP structure, morph=None for each token.
    """
    sentences: list[IRSentence] = []
    global_idx = 0

    for sentence_tree in yap_result.sentences:
        tokens: list[IRToken] = []
        for node in sentence_tree.nodes:
            surface = node.form
            offset = _get_token_offset(tokenization, global_idx)
            prefixes = _get_token_prefixes(tokenization, global_idx)
            suffix = _get_token_suffix(tokenization, global_idx)

            tokens.append(IRToken(
                surface=surface,
                offset=offset,
                morph=None,
                dep_node=node,
                prefixes=prefixes,
                suffix=suffix,
            ))
            global_idx += 1

        sentences.append(IRSentence(
            tokens=tokens,
            dep_tree=sentence_tree,
        ))

    return sentences


def _build_both_fail(
    tokenization: TokenizationResult,
) -> list[IRSentence]:
    """Build IR when both Stanza and YAP fail.

    Creates a single IRSentence from tokenizer output with morph=None,
    dep_node=None, dep_tree=None.
    """
    tokens: list[IRToken] = []

    for i, surface in enumerate(tokenization.tokens):
        offset = _get_token_offset(tokenization, i)
        prefixes = _get_token_prefixes(tokenization, i)
        suffix = _get_token_suffix(tokenization, i)

        tokens.append(IRToken(
            surface=surface,
            offset=offset,
            morph=None,
            dep_node=None,
            prefixes=prefixes,
            suffix=suffix,
        ))

    return [IRSentence(tokens=tokens, dep_tree=None)] if tokens else []


def _apply_yap_ambiguity_counts(
    sentences: list[IRSentence],
    ambiguity_counts: list[int],
) -> None:
    """Override morph.ambiguity_count with YAP MA lattice counts.

    Stanza always returns ambiguity_count=1 (it disambiguates internally).
    YAP's MA lattice contains all candidate analyses per token — the count
    per token_id is the real morphological ambiguity.
    """
    global_idx = 0
    for sentence in sentences:
        for token in sentence.tokens:
            if global_idx < len(ambiguity_counts):
                count = ambiguity_counts[global_idx]
                if token.morph is not None:
                    token.morph.ambiguity_count = count
            global_idx += 1


def build_ir(
    original_text: str,
    normalization: NormalizationResult,
    tokenization: TokenizationResult,
    stanza_result: StanzaResult | StanzaError,
    yap_result: YAPResult | YAPError,
) -> IntermediateRepresentation:
    """Combine all upstream outputs into a single IntermediateRepresentation.

    Aligns morphological analyses with dependency tree nodes via token IDs.
    Handles partial data when upstream modules fail.

    Args:
        original_text: The raw input text before any processing.
        normalization: Result from the normalizer stage.
        tokenization: Result from the tokenizer stage.
        stanza_result: Stanza morphological analysis or error.
        yap_result: YAP syntactic parsing result or error.

    Returns:
        An IntermediateRepresentation combining all available data.
    """
    missing_layers: list[str] = []

    stanza_ok = isinstance(stanza_result, StanzaResult)
    yap_ok = isinstance(yap_result, YAPResult)

    if not stanza_ok:
        missing_layers.append("stanza")
    if not yap_ok:
        missing_layers.append("yap")

    if stanza_ok and yap_ok:
        sentences = _build_both_succeed(tokenization, stanza_result, yap_result)
        _apply_yap_ambiguity_counts(sentences, yap_result.ambiguity_counts)
    elif stanza_ok and not yap_ok:
        sentences = _build_stanza_only(tokenization, stanza_result)
    elif not stanza_ok and yap_ok:
        sentences = _build_yap_only(tokenization, yap_result)
    else:
        sentences = _build_both_fail(tokenization)

    return IntermediateRepresentation(
        original_text=original_text,
        normalized_text=normalization.normalized_text,
        sentences=sentences,
        missing_layers=missing_layers,
    )
