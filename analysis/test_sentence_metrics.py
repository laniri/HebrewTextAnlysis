# Feature: probabilistic-analysis-layer, Property 8: SentenceMetrics count matches IR sentences
"""Property-based tests for sentence_metrics.py.

**Validates: Requirements 16.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from hebrew_profiler.models import (
    DepTreeNode,
    IntermediateRepresentation,
    IRSentence,
    IRToken,
    MorphAnalysis,
    SentenceTree,
)
from analysis.sentence_metrics import SentenceMetrics, extract_sentence_metrics


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def morph_analysis_strategy():
    return st.builds(
        MorphAnalysis,
        surface=st.text(min_size=1, max_size=10),
        lemma=st.text(min_size=1, max_size=10),
        pos=st.just("NN"),
        gender=st.none(),
        number=st.none(),
        prefixes=st.just([]),
        suffix=st.none(),
        binyan=st.none(),
        tense=st.none(),
        ambiguity_count=st.just(1),
        top_k_analyses=st.just([]),
    )


def ir_token_strategy():
    return st.builds(
        IRToken,
        surface=st.text(min_size=1, max_size=10),
        offset=st.just((0, 1)),
        morph=st.one_of(st.none(), morph_analysis_strategy()),
        dep_node=st.none(),
        prefixes=st.just([]),
        suffix=st.none(),
    )


def ir_sentence_strategy():
    return st.builds(
        IRSentence,
        tokens=st.lists(ir_token_strategy(), min_size=0, max_size=10),
        dep_tree=st.none(),  # keep it simple; tree_depth tested separately
    )


def ir_strategy():
    return st.builds(
        IntermediateRepresentation,
        original_text=st.just(""),
        normalized_text=st.just(""),
        sentences=st.lists(ir_sentence_strategy(), min_size=0, max_size=20),
        missing_layers=st.just([]),
    )


# ---------------------------------------------------------------------------
# Property 8: SentenceMetrics count matches IR sentences
# ---------------------------------------------------------------------------

@given(ir=ir_strategy())
@settings(max_examples=100)
def test_sentence_metrics_count_matches_ir_sentences(ir: IntermediateRepresentation):
    """For any IR with N sentences, extract_sentence_metrics SHALL produce exactly N
    SentenceMetrics objects, each with index equal to its position (0 through N-1).

    **Validates: Requirements 16.2**
    """
    metrics = extract_sentence_metrics(ir)

    # Count must match
    assert len(metrics) == len(ir.sentences)

    # Each index must match its position
    for i, sm in enumerate(metrics):
        assert sm.index == i
