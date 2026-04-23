"""Unit tests for hebrew_profiler.yap_adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hebrew_profiler.models import DepTreeNode, SentenceTree, YAPError, YAPResult
from hebrew_profiler.yap_adapter import (
    _parse_features,
    _parse_dep_tree,
    _parse_lattice,
    _segment_sentences,
    parse_syntax,
)


# ---------------------------------------------------------------------------
# _parse_features
# ---------------------------------------------------------------------------

class TestParseFeatures:
    def test_empty_string(self):
        assert _parse_features("") == {}

    def test_underscore(self):
        assert _parse_features("_") == {}

    def test_single_feature(self):
        assert _parse_features("gen=M") == {"gen": "M"}

    def test_multiple_features(self):
        assert _parse_features("gen=M|num=S|per=3") == {
            "gen": "M",
            "num": "S",
            "per": "3",
        }

    def test_value_with_equals(self):
        assert _parse_features("key=a=b") == {"key": "a=b"}


# ---------------------------------------------------------------------------
# _parse_dep_tree
# ---------------------------------------------------------------------------

SINGLE_CONLL_LINE = "1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t3\tsubj"

class TestParseDepTree:
    def test_single_line(self):
        nodes = _parse_dep_tree(SINGLE_CONLL_LINE)
        assert len(nodes) == 1
        n = nodes[0]
        assert isinstance(n, DepTreeNode)
        assert n.id == 1
        assert n.form == "הילד"
        assert n.lemma == "ילד"
        assert n.cpostag == "NN"
        assert n.postag == "NN"
        assert n.features == {"gen": "M", "num": "S"}
        assert n.head == 3
        assert n.deprel == "subj"

    def test_empty_input(self):
        assert _parse_dep_tree("") == []

    def test_skips_short_lines(self):
        assert _parse_dep_tree("too\tfew\tfields") == []


# ---------------------------------------------------------------------------
# _parse_lattice
# ---------------------------------------------------------------------------

LATTICE_LINE = "0\t1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t1"

class TestParseLattice:
    def test_single_record(self):
        records = _parse_lattice(LATTICE_LINE)
        assert len(records) == 1
        r = records[0]
        assert r["from"] == 0
        assert r["to"] == 1
        assert r["form"] == "הילד"
        assert r["lemma"] == "ילד"
        assert r["cpostag"] == "NN"
        assert r["postag"] == "NN"
        assert r["features"] == {"gen": "M", "num": "S"}
        assert r["token_id"] == 1

    def test_empty_input(self):
        assert _parse_lattice("") == []


# ---------------------------------------------------------------------------
# _segment_sentences
# ---------------------------------------------------------------------------

TWO_SENTENCE_CONLL = (
    "1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t2\tsubj\n"
    "2\tרץ\tרוץ\tVB\tVB\t_\t0\tROOT\n"
    "\n"
    "1\tהילדה\tילדה\tNN\tNN\tgen=F|num=S\t2\tsubj\n"
    "2\tאכלה\tאכל\tVB\tVB\t_\t0\tROOT\n"
)

class TestSegmentSentences:
    def test_two_sentences(self):
        sentences = _segment_sentences(TWO_SENTENCE_CONLL)
        assert len(sentences) == 2
        assert all(isinstance(s, SentenceTree) for s in sentences)
        assert len(sentences[0].nodes) == 2
        assert len(sentences[1].nodes) == 2
        assert sentences[0].nodes[0].form == "הילד"
        assert sentences[1].nodes[0].form == "הילדה"

    def test_single_sentence_no_trailing_blank(self):
        raw = "1\tהילד\tילד\tNN\tNN\t_\t2\tsubj\n2\tרץ\tרוץ\tVB\tVB\t_\t0\tROOT"
        sentences = _segment_sentences(raw)
        assert len(sentences) == 1
        assert len(sentences[0].nodes) == 2

    def test_empty_input(self):
        assert _segment_sentences("") == []

    def test_three_sentences(self):
        raw = (
            "1\ta\tb\tNN\tNN\t_\t0\tROOT\n\n"
            "1\tc\td\tNN\tNN\t_\t0\tROOT\n\n"
            "1\te\tf\tNN\tNN\t_\t0\tROOT\n"
        )
        sentences = _segment_sentences(raw)
        assert len(sentences) == 3


# ---------------------------------------------------------------------------
# parse_syntax — integration-level tests with mocked HTTP
# ---------------------------------------------------------------------------

MOCK_YAP_RESPONSE = {
    "ma_lattice": (
        "0\t1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t1\n"
        "0\t1\tהילד\tילד\tNNP\tNNP\tgen=M|num=S\t1\n"
        "0\t1\tהילד\tילד\tNN\tNN\tgen=F|num=S\t1\n"
        "1\t2\tרץ\tרוץ\tVB\tVB\t_\t2\n"
        "1\t2\tרץ\tריצה\tNN\tNN\t_\t2\n"
    ),
    "md_lattice": "0\t1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t1",
    "dep_tree": (
        "1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t2\tsubj\n"
        "2\tרץ\tרוץ\tVB\tVB\t_\t0\tROOT\n"
    ),
}


class TestParseSyntax:
    @patch("hebrew_profiler.yap_adapter.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = MOCK_YAP_RESPONSE
        mock_get.return_value = mock_resp

        result = parse_syntax("הילד רץ", "http://localhost:8000/yap/heb/joint")

        assert isinstance(result, YAPResult)
        assert len(result.morphological_disambiguation) == 1
        assert len(result.sentences) == 1
        assert len(result.sentences[0].nodes) == 2
        # Token 1 has 3 MA analyses, token 2 has 2
        assert result.ambiguity_counts == [3, 2]

        # Verify trailing spaces were appended
        call_data = mock_get.call_args
        import json
        sent_payload = json.loads(call_data.kwargs.get("data", call_data[1].get("data", "")))
        assert sent_payload["text"].endswith("  ")

    @patch("hebrew_profiler.yap_adapter.requests.get")
    def test_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("refused")

        result = parse_syntax("test", "http://localhost:8000/yap/heb/joint")

        assert isinstance(result, YAPError)
        assert result.error_type == "YAPConnectionError"
        assert result.http_status is None
        assert "refused" in result.message.lower() or "connection" in result.message.lower()

    @patch("hebrew_profiler.yap_adapter.requests.get")
    def test_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = parse_syntax("test", "http://localhost:8000/yap/heb/joint")

        assert isinstance(result, YAPError)
        assert result.error_type == "YAPHTTPError"
        assert result.http_status == 500

    @patch("hebrew_profiler.yap_adapter.requests.get")
    def test_multi_sentence(self, mock_get):
        """Pre-splitting produces one YAP call per sentence."""
        single_sentence_conll = (
            "1\tהילד\tילד\tNN\tNN\tgen=M|num=S\t2\tsubj\n"
            "2\tרץ\tרוץ\tVB\tVB\t_\t0\tROOT\n"
        )
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "md_lattice": "",
            "dep_tree": single_sentence_conll,
        }
        mock_get.return_value = mock_resp

        result = parse_syntax("sentence one. sentence two.", "http://localhost:8000/yap/heb/joint")

        assert isinstance(result, YAPResult)
        assert len(result.sentences) == 2
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from hebrew_profiler.models import DepTreeNode, SentenceTree


# -- Custom strategies for CoNLL generation --

# Alphabet safe for CoNLL fields: printable, no tabs/newlines/whitespace-only chars.
# Uses letters, digits, Hebrew chars, and safe punctuation to avoid strip() edge cases.
_SAFE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "אבגדהוזחטיכלמנסעפצקרשת"
    "._-:;!?()[]{}/@#$%^&*~"
)

_safe_text = st.text(alphabet=_SAFE_ALPHABET, min_size=1, max_size=30)

# Strategy for a pipe-separated feature string (e.g. "gen=M|num=S") or "_"
_feat_key_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
_feat_val_alphabet = _feat_key_alphabet + ".:"
_feature_value = st.text(alphabet=_feat_key_alphabet, min_size=1, max_size=10)
_feature_val = st.text(alphabet=_feat_val_alphabet, min_size=1, max_size=10)
_feature_pair = st.tuples(_feature_value, _feature_val).map(lambda kv: f"{kv[0]}={kv[1]}")
_feature_string = st.one_of(
    st.just("_"),
    st.lists(_feature_pair, min_size=1, max_size=5).map("|".join),
)


def _conll_line_strategy():
    """Generate a single well-formed CoNLL dependency-tree line with 8 tab-separated fields."""
    return st.tuples(
        st.integers(min_value=0, max_value=9999),   # token ID
        _safe_text,                                   # surface form
        _safe_text,                                   # lemma
        _safe_text,                                   # coarse POS
        _safe_text,                                   # fine POS
        _feature_string,                              # morphological features
        st.integers(min_value=0, max_value=9999),    # head index
        _safe_text,                                   # deprel
    ).map(lambda t: f"{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{t[4]}\t{t[5]}\t{t[6]}\t{t[7]}")


def _conll_block_strategy(min_lines=1, max_lines=5):
    """Generate a block of CoNLL lines (one sentence)."""
    return st.lists(
        _conll_line_strategy(),
        min_size=min_lines,
        max_size=max_lines,
    )


# ---------------------------------------------------------------------------
# Property 7: CoNLL dependency tree parsing completeness
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

class TestProperty7ConllParsingCompleteness:
    """**Validates: Requirements 5.2**

    For any well-formed CoNLL-format line containing tab-separated fields,
    the YAP_Adapter parser SHALL extract all required fields: token ID,
    surface form, lemma, coarse POS, fine POS, morphological features dict,
    head index, and dependency relation label.
    """

    @given(
        token_id=st.integers(min_value=0, max_value=9999),
        surface=_safe_text,
        lemma=_safe_text,
        cpostag=_safe_text,
        postag=_safe_text,
        feat_str=_feature_string,
        head=st.integers(min_value=0, max_value=9999),
        deprel=_safe_text,
    )
    @settings(max_examples=150)
    def test_parse_dep_tree_extracts_all_fields(
        self, token_id, surface, lemma, cpostag, postag, feat_str, head, deprel
    ):
        """Property 7: every field of a well-formed CoNLL line is extracted."""
        line = f"{token_id}\t{surface}\t{lemma}\t{cpostag}\t{postag}\t{feat_str}\t{head}\t{deprel}"
        nodes = _parse_dep_tree(line)

        assert len(nodes) == 1
        node = nodes[0]

        # Verify all required fields
        assert isinstance(node, DepTreeNode)
        assert node.id == token_id
        assert node.form == surface
        assert node.lemma == lemma
        assert node.cpostag == cpostag
        assert node.postag == postag
        assert isinstance(node.features, dict)
        assert node.head == head
        assert node.deprel == deprel

        # Verify features dict matches the input feature string
        expected_features = _parse_features(feat_str)
        assert node.features == expected_features


# ---------------------------------------------------------------------------
# Property 8: Multi-sentence CoNLL segmentation
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

class TestProperty8MultiSentenceSegmentation:
    """**Validates: Requirements 5.3**

    For any CoNLL output containing K blank-line-separated blocks (K >= 1),
    the YAP_Adapter SHALL produce exactly K sentence dependency trees.
    """

    @given(
        blocks=st.lists(
            _conll_block_strategy(min_lines=1, max_lines=4),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=150)
    def test_segment_sentences_produces_k_trees(self, blocks):
        """Property 8: K CoNLL blocks yield exactly K SentenceTree objects."""
        k = len(blocks)

        # Join blocks with blank-line separators (standard CoNLL format)
        raw = "\n\n".join("\n".join(lines) for lines in blocks)

        sentences = _segment_sentences(raw)

        assert len(sentences) == k
        for sent in sentences:
            assert isinstance(sent, SentenceTree)
            assert len(sent.nodes) >= 1
