"""Tests for the IR Builder module."""

from __future__ import annotations

from hebrew_profiler.ir_builder import build_ir
from hebrew_profiler.models import (
    DepTreeNode,
    IntermediateRepresentation,
    MorphAnalysis,
    NormalizationResult,
    SentenceTree,
    StanzaError,
    StanzaResult,
    TokenizationResult,
    YAPError,
    YAPResult,
)


def _make_morph(surface: str, pos: str = "NN") -> MorphAnalysis:
    return MorphAnalysis(
        surface=surface,
        lemma=surface,
        pos=pos,
        gender=None,
        number=None,
        prefixes=[],
        suffix=None,
        binyan=None,
        tense=None,
        ambiguity_count=1,
        top_k_analyses=[],
    )


def _make_dep_node(token_id: int, form: str) -> DepTreeNode:
    return DepTreeNode(
        id=token_id,
        form=form,
        lemma=form,
        cpostag="NN",
        postag="NN",
        features={},
        head=0,
        deprel="root" if token_id == 1 else "dep",
    )


def _make_tokenization(tokens: list[str]) -> TokenizationResult:
    offsets = []
    pos = 0
    for t in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t) + 1
    return TokenizationResult(
        tokens=tokens,
        character_offsets=offsets,
        prefix_annotations=[[] for _ in tokens],
        suffix_annotations=[None for _ in tokens],
    )


class TestBuildIRBothSucceed:
    """Tests for when both Stanza and YAP succeed."""

    def test_single_sentence(self):
        tokens = ["hello", "world"]
        tok = _make_tokenization(tokens)
        norm = NormalizationResult(normalized_text="hello world")
        stanza = StanzaResult(analyses=[_make_morph("hello"), _make_morph("world")])
        yap = YAPResult(
            morphological_disambiguation=[],
            sentences=[SentenceTree(nodes=[
                _make_dep_node(1, "hello"),
                _make_dep_node(2, "world"),
            ])],
            ambiguity_counts=[3, 5],
        )

        ir = build_ir("hello world", norm, tok, stanza, yap)

        assert ir.original_text == "hello world"
        assert ir.normalized_text == "hello world"
        assert ir.missing_layers == []
        assert len(ir.sentences) == 1
        assert len(ir.sentences[0].tokens) == 2
        assert ir.sentences[0].dep_tree is not None
        # Check alignment
        assert ir.sentences[0].tokens[0].morph is not None
        assert ir.sentences[0].tokens[0].dep_node is not None
        assert ir.sentences[0].tokens[0].morph.surface == "hello"
        assert ir.sentences[0].tokens[0].dep_node.form == "hello"
        # Verify YAP ambiguity counts override Stanza's default of 1
        assert ir.sentences[0].tokens[0].morph.ambiguity_count == 3
        assert ir.sentences[0].tokens[1].morph.ambiguity_count == 5

    def test_multi_sentence(self):
        tokens = ["a", "b", "c"]
        tok = _make_tokenization(tokens)
        norm = NormalizationResult(normalized_text="a b c")
        stanza = StanzaResult(analyses=[
            _make_morph("a"), _make_morph("b"), _make_morph("c"),
        ])
        yap = YAPResult(
            morphological_disambiguation=[],
            sentences=[
                SentenceTree(nodes=[_make_dep_node(1, "a")]),
                SentenceTree(nodes=[_make_dep_node(1, "b"), _make_dep_node(2, "c")]),
            ],
            ambiguity_counts=[2, 4, 3],
        )

        ir = build_ir("a b c", norm, tok, stanza, yap)

        assert len(ir.sentences) == 2
        assert len(ir.sentences[0].tokens) == 1
        assert len(ir.sentences[1].tokens) == 2
        assert ir.sentences[0].tokens[0].morph.surface == "a"
        assert ir.sentences[1].tokens[0].morph.surface == "b"
        assert ir.sentences[1].tokens[1].morph.surface == "c"


class TestBuildIRStanzaOnly:
    """Tests for when only Stanza succeeds (YAP fails)."""

    def test_yap_error(self):
        tokens = ["hello", "world"]
        tok = _make_tokenization(tokens)
        norm = NormalizationResult(normalized_text="hello world")
        stanza = StanzaResult(analyses=[_make_morph("hello"), _make_morph("world")])
        yap_err = YAPError(
            error_type="YAPConnectionError",
            http_status=None,
            message="Connection refused",
        )

        ir = build_ir("hello world", norm, tok, stanza, yap_err)

        assert ir.missing_layers == ["yap"]
        assert len(ir.sentences) == 1
        assert len(ir.sentences[0].tokens) == 2
        assert ir.sentences[0].dep_tree is None
        assert ir.sentences[0].tokens[0].morph is not None
        assert ir.sentences[0].tokens[0].dep_node is None


class TestBuildIRYAPOnly:
    """Tests for when only YAP succeeds (Stanza fails)."""

    def test_stanza_error(self):
        tokens = ["hello", "world"]
        tok = _make_tokenization(tokens)
        norm = NormalizationResult(normalized_text="hello world")
        stanza_err = StanzaError(error_type="RuntimeError", message="Pipeline failed")
        yap = YAPResult(
            morphological_disambiguation=[],
            sentences=[SentenceTree(nodes=[
                _make_dep_node(1, "hello"),
                _make_dep_node(2, "world"),
            ])],
            ambiguity_counts=[2, 3],
        )

        ir = build_ir("hello world", norm, tok, stanza_err, yap)

        assert ir.missing_layers == ["stanza"]
        assert len(ir.sentences) == 1
        assert len(ir.sentences[0].tokens) == 2
        assert ir.sentences[0].dep_tree is not None
        assert ir.sentences[0].tokens[0].morph is None
        assert ir.sentences[0].tokens[0].dep_node is not None
        assert ir.sentences[0].tokens[0].dep_node.form == "hello"


class TestBuildIRBothFail:
    """Tests for when both Stanza and YAP fail."""

    def test_both_errors(self):
        tokens = ["hello", "world"]
        tok = _make_tokenization(tokens)
        norm = NormalizationResult(normalized_text="hello world")
        stanza_err = StanzaError(error_type="RuntimeError", message="Failed")
        yap_err = YAPError(
            error_type="YAPConnectionError", http_status=None, message="Failed"
        )

        ir = build_ir("hello world", norm, tok, stanza_err, yap_err)

        assert sorted(ir.missing_layers) == ["stanza", "yap"]
        assert len(ir.sentences) == 1
        assert len(ir.sentences[0].tokens) == 2
        assert ir.sentences[0].dep_tree is None
        assert ir.sentences[0].tokens[0].morph is None
        assert ir.sentences[0].tokens[0].dep_node is None
        assert ir.sentences[0].tokens[0].surface == "hello"

    def test_empty_tokens_both_fail(self):
        tok = _make_tokenization([])
        norm = NormalizationResult(normalized_text="")
        stanza_err = StanzaError(error_type="RuntimeError", message="Failed")
        yap_err = YAPError(
            error_type="YAPConnectionError", http_status=None, message="Failed"
        )

        ir = build_ir("", norm, tok, stanza_err, yap_err)

        assert sorted(ir.missing_layers) == ["stanza", "yap"]
        assert len(ir.sentences) == 0


class TestBuildIRPreservesTokenizerData:
    """Tests that tokenizer prefix/suffix annotations are preserved."""

    def test_prefixes_and_suffixes_preserved(self):
        tok = TokenizationResult(
            tokens=["\u05D1\u05D1\u05D9\u05EA"],  # בבית
            character_offsets=[(0, 4)],
            prefix_annotations=[["\u05D1"]],  # ב prefix
            suffix_annotations=[None],
        )
        norm = NormalizationResult(normalized_text="\u05D1\u05D1\u05D9\u05EA")
        stanza = StanzaResult(analyses=[_make_morph("\u05D1\u05D1\u05D9\u05EA")])
        yap_err = YAPError(
            error_type="YAPConnectionError", http_status=None, message="Failed"
        )

        ir = build_ir("\u05D1\u05D1\u05D9\u05EA", norm, tok, stanza, yap_err)

        assert ir.sentences[0].tokens[0].prefixes == ["\u05D1"]
        assert ir.sentences[0].tokens[0].suffix is None


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

import hypothesis.strategies as st
from hypothesis import given, settings


# ---- Hypothesis strategies ------------------------------------------------

# Hebrew-ish text: mix of Hebrew letters, ASCII, spaces
_hebrew_chars = st.sampled_from(
    [chr(c) for c in range(0x05D0, 0x05EB)]  # aleph-tav
)
_token_char = st.one_of(_hebrew_chars, st.sampled_from(list("abcdefghij")))

_token_st = st.text(_token_char, min_size=1, max_size=8)

# List of 1..6 tokens
_token_list_st = st.lists(_token_st, min_size=1, max_size=6)

_pos_tags = st.sampled_from(["NN", "VB", "JJ", "RB", "IN", "DT"])
_deprels = st.sampled_from(["root", "dep", "nsubj", "obj", "advmod", "amod"])


@st.composite
def _morph_for_surface(draw, surface: str) -> MorphAnalysis:
    """Generate a MorphAnalysis matching a given surface form."""
    return MorphAnalysis(
        surface=surface,
        lemma=draw(st.text(st.characters(whitelist_categories=("L",)), min_size=1, max_size=6)),
        pos=draw(_pos_tags),
        gender=draw(st.one_of(st.none(), st.sampled_from(["Masc", "Fem"]))),
        number=draw(st.one_of(st.none(), st.sampled_from(["Sing", "Plur"]))),
        prefixes=draw(st.lists(st.text(min_size=1, max_size=2), max_size=2)),
        suffix=draw(st.one_of(st.none(), st.text(min_size=1, max_size=3))),
        binyan=draw(st.one_of(st.none(), st.sampled_from(["PAAL", "PIEL", "HIFIL"]))),
        tense=draw(st.one_of(st.none(), st.sampled_from(["Past", "Present", "Future"]))),
        ambiguity_count=draw(st.integers(min_value=1, max_value=5)),
        top_k_analyses=[],
    )


@st.composite
def _dep_node_for(draw, token_id: int, form: str) -> DepTreeNode:
    """Generate a DepTreeNode matching a given token ID and form."""
    return DepTreeNode(
        id=token_id,
        form=form,
        lemma=draw(st.text(st.characters(whitelist_categories=("L",)), min_size=1, max_size=6)),
        cpostag=draw(_pos_tags),
        postag=draw(_pos_tags),
        features=draw(st.fixed_dictionaries({}, optional={})),
        head=draw(st.integers(min_value=0, max_value=max(token_id, 1))),
        deprel=draw(_deprels),
    )


@st.composite
def _sentence_partition(draw, tokens: list[str]):
    """Partition a token list into 1+ sentence groups (non-empty sublists)."""
    n = len(tokens)
    if n <= 1:
        return [tokens] if tokens else [[]]
    # Choose cut points (between 1 and n-1 inclusive)
    num_cuts = draw(st.integers(min_value=0, max_value=min(n - 1, 3)))
    cuts = sorted(draw(st.lists(
        st.integers(min_value=1, max_value=n - 1),
        min_size=num_cuts,
        max_size=num_cuts,
        unique=True,
    )))
    parts: list[list[str]] = []
    prev = 0
    for c in cuts:
        parts.append(tokens[prev:c])
        prev = c
    parts.append(tokens[prev:])
    return [p for p in parts if p]  # filter empty


@st.composite
def _matching_stanza_yap(draw, tokens: list[str]):
    """Generate matching StanzaResult and YAPResult for a token list.

    Returns (StanzaResult, YAPResult, sentence_partition).
    """
    partition = draw(_sentence_partition(tokens))
    analyses: list[MorphAnalysis] = []
    sentences: list[SentenceTree] = []
    for sent_tokens in partition:
        nodes: list[DepTreeNode] = []
        for local_id, surface in enumerate(sent_tokens, start=1):
            analyses.append(draw(_morph_for_surface(surface)))
            nodes.append(draw(_dep_node_for(local_id, surface)))
        sentences.append(SentenceTree(nodes=nodes))
    return (
        StanzaResult(analyses=analyses),
        YAPResult(morphological_disambiguation=[], sentences=sentences, ambiguity_counts=[]),
        partition,
    )


_stanza_error_st = st.builds(
    StanzaError,
    error_type=st.just("RuntimeError"),
    message=st.text(min_size=1, max_size=20),
)

_yap_error_st = st.builds(
    YAPError,
    error_type=st.sampled_from(["YAPConnectionError", "YAPHTTPError"]),
    http_status=st.one_of(st.none(), st.integers(min_value=400, max_value=599)),
    message=st.text(min_size=1, max_size=20),
)


# ---- Property 9: IR structural completeness (Task 8.2) -------------------
# Validates: Requirements 6.1, 6.2

class TestProperty9IRStructuralCompleteness:
    """Property 9: IR structural completeness.

    For any valid combination of tokenization, Stanza, and YAP outputs,
    the IR_Builder SHALL produce an IntermediateRepresentation containing
    the original text, normalized text, and a list of sentences where each
    sentence contains tokens with both morphological and syntactic annotations.

    **Validates: Requirements 6.1, 6.2**
    """

    @given(data=st.data(), tokens=_token_list_st)
    @settings(max_examples=100)
    def test_ir_structural_completeness(self, data, tokens: list[str]):
        original_text = " ".join(tokens)
        norm = NormalizationResult(normalized_text=original_text)
        tok = _make_tokenization(tokens)

        stanza_result, yap_result, partition = data.draw(
            _matching_stanza_yap(tokens)
        )

        ir = build_ir(original_text, norm, tok, stanza_result, yap_result)

        # IR contains original and normalized text
        assert ir.original_text == original_text
        assert ir.normalized_text == original_text

        # No missing layers when both succeed
        assert ir.missing_layers == []

        # Sentences list is non-empty and matches partition count
        assert len(ir.sentences) == len(partition)

        # Every sentence has tokens; every token has morph AND dep_node
        total_ir_tokens = 0
        for sent_idx, ir_sent in enumerate(ir.sentences):
            assert len(ir_sent.tokens) == len(partition[sent_idx])
            assert ir_sent.dep_tree is not None
            for ir_tok in ir_sent.tokens:
                assert ir_tok.morph is not None, (
                    "morph should be present when Stanza succeeds"
                )
                assert ir_tok.dep_node is not None, (
                    "dep_node should be present when YAP succeeds"
                )
                total_ir_tokens += 1

        # Total tokens across sentences equals input token count
        assert total_ir_tokens == len(tokens)


# ---- Property 10: IR token-to-dependency alignment (Task 8.3) -------------
# Validates: Requirements 6.3

class TestProperty10IRTokenAlignment:
    """Property 10: IR token-to-dependency alignment.

    For any IntermediateRepresentation where both Stanza and YAP data are
    present, each IRToken's morphological analysis and dependency tree node
    SHALL reference the same token (matched by token ID), ensuring
    cross-layer consistency.

    **Validates: Requirements 6.3**
    """

    @given(data=st.data(), tokens=_token_list_st)
    @settings(max_examples=100)
    def test_ir_token_to_dependency_alignment(self, data, tokens: list[str]):
        original_text = " ".join(tokens)
        norm = NormalizationResult(normalized_text=original_text)
        tok = _make_tokenization(tokens)

        stanza_result, yap_result, partition = data.draw(
            _matching_stanza_yap(tokens)
        )

        ir = build_ir(original_text, norm, tok, stanza_result, yap_result)

        # Both layers present
        assert ir.missing_layers == []

        for ir_sent in ir.sentences:
            for ir_tok in ir_sent.tokens:
                assert ir_tok.morph is not None
                assert ir_tok.dep_node is not None
                # The morph surface and dep_node form must reference the same token
                assert ir_tok.morph.surface == ir_tok.dep_node.form, (
                    f"Morph surface '{ir_tok.morph.surface}' != "
                    f"dep_node form '{ir_tok.dep_node.form}'"
                )
                # The IR token surface should also match
                assert ir_tok.surface == ir_tok.dep_node.form


# ---- Property 11: IR graceful degradation (Task 8.4) ---------------------
# Validates: Requirements 6.4

class TestProperty11IRGracefulDegradation:
    """Property 11: IR graceful degradation.

    For any combination of upstream outputs where one or more modules return
    errors, the IR_Builder SHALL still produce a valid IntermediateRepresentation
    with available data, and `missing_layers` SHALL list exactly the modules
    that failed.

    **Validates: Requirements 6.4**
    """

    @given(
        tokens=_token_list_st,
        stanza_err=_stanza_error_st,
        yap_err=_yap_error_st,
    )
    @settings(max_examples=100)
    def test_both_fail(self, tokens: list[str], stanza_err: StanzaError, yap_err: YAPError):
        """When both Stanza and YAP fail, missing_layers has both, IR still valid."""
        original_text = " ".join(tokens)
        norm = NormalizationResult(normalized_text=original_text)
        tok = _make_tokenization(tokens)

        ir = build_ir(original_text, norm, tok, stanza_err, yap_err)

        assert isinstance(ir, IntermediateRepresentation)
        assert ir.original_text == original_text
        assert ir.normalized_text == original_text
        assert sorted(ir.missing_layers) == ["stanza", "yap"]

        # IR still has sentences with tokens from tokenizer
        all_surfaces = [
            ir_tok.surface for sent in ir.sentences for ir_tok in sent.tokens
        ]
        assert len(all_surfaces) == len(tokens)

        # All morph and dep_node are None
        for sent in ir.sentences:
            assert sent.dep_tree is None
            for ir_tok in sent.tokens:
                assert ir_tok.morph is None
                assert ir_tok.dep_node is None

    @given(
        data=st.data(),
        tokens=_token_list_st,
        yap_err=_yap_error_st,
    )
    @settings(max_examples=100)
    def test_yap_fails(self, data, tokens: list[str], yap_err: YAPError):
        """When only YAP fails, missing_layers=['yap'], morph present, dep absent."""
        original_text = " ".join(tokens)
        norm = NormalizationResult(normalized_text=original_text)
        tok = _make_tokenization(tokens)

        # Build a valid StanzaResult
        analyses = [data.draw(_morph_for_surface(t)) for t in tokens]
        stanza_result = StanzaResult(analyses=analyses)

        ir = build_ir(original_text, norm, tok, stanza_result, yap_err)

        assert isinstance(ir, IntermediateRepresentation)
        assert ir.missing_layers == ["yap"]

        all_tokens = [ir_tok for sent in ir.sentences for ir_tok in sent.tokens]
        assert len(all_tokens) == len(tokens)

        for sent in ir.sentences:
            assert sent.dep_tree is None
        for ir_tok in all_tokens:
            assert ir_tok.morph is not None
            assert ir_tok.dep_node is None

    @given(
        data=st.data(),
        tokens=_token_list_st,
        stanza_err=_stanza_error_st,
    )
    @settings(max_examples=100)
    def test_stanza_fails(self, data, tokens: list[str], stanza_err: StanzaError):
        """When only Stanza fails, missing_layers=['stanza'], dep present, morph absent."""
        original_text = " ".join(tokens)
        norm = NormalizationResult(normalized_text=original_text)
        tok = _make_tokenization(tokens)

        # Build a valid YAPResult with a single sentence
        nodes = [
            data.draw(_dep_node_for(i + 1, t))
            for i, t in enumerate(tokens)
        ]
        yap_result = YAPResult(
            morphological_disambiguation=[],
            sentences=[SentenceTree(nodes=nodes)],
            ambiguity_counts=[2] * len(nodes),
        )

        ir = build_ir(original_text, norm, tok, stanza_err, yap_result)

        assert isinstance(ir, IntermediateRepresentation)
        assert ir.missing_layers == ["stanza"]

        all_tokens = [ir_tok for sent in ir.sentences for ir_tok in sent.tokens]
        assert len(all_tokens) == len(tokens)

        for sent in ir.sentences:
            assert sent.dep_tree is not None
        for ir_tok in all_tokens:
            assert ir_tok.morph is None
            assert ir_tok.dep_node is not None
