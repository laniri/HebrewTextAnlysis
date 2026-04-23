"""Tests for the feature extractor module."""

from __future__ import annotations

import pytest

from hebrew_profiler.feature_extractor import (
    _agreement_error_rate,
    _all_tokens,
    _binyan_entropy,
    _build_cache,
    _compute_tree_depth,
    _CONNECTIVES,
    _connective_ratio,
    _construct_ratio,
    _extract_discourse,
    _extract_lexical,
    _extract_morphological,
    _extract_structural,
    _extract_syntactic,
    _FeatureCache,
    _pronoun_to_noun_ratio,
    _sentence_overlap,
    extract_features,
)
from hebrew_profiler.models import (
    DepTreeNode,
    DiscourseFeatures,
    Features,
    IntermediateRepresentation,
    IRSentence,
    IRToken,
    LexicalFeatures,
    MorphAnalysis,
    MorphFeatures,
    SentenceTree,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)


# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

def _make_morph(
    surface: str = "tok",
    lemma: str = "tok",
    pos: str = "NOUN",
    binyan: str | None = None,
    ambiguity_count: int = 1,
    suffix: str | None = None,
    gender: str | None = None,
    number: str | None = None,
) -> MorphAnalysis:
    return MorphAnalysis(
        surface=surface,
        lemma=lemma,
        pos=pos,
        gender=gender,
        number=number,
        prefixes=[],
        suffix=suffix,
        binyan=binyan,
        tense=None,
        ambiguity_count=ambiguity_count,
        top_k_analyses=[],
    )


def _make_dep_node(
    id: int = 1,
    form: str = "tok",
    head: int = 0,
    deprel: str = "nsubj",
) -> DepTreeNode:
    return DepTreeNode(
        id=id, form=form, lemma=form, cpostag="NN",
        postag="NN", features={}, head=head, deprel=deprel,
    )


def _make_ir_token(
    surface: str = "tok",
    morph: MorphAnalysis | None = None,
    dep_node: DepTreeNode | None = None,
    prefixes: list[str] | None = None,
    suffix: str | None = None,
) -> IRToken:
    return IRToken(
        surface=surface,
        offset=(0, len(surface)),
        morph=morph,
        dep_node=dep_node,
        prefixes=prefixes or [],
        suffix=suffix,
    )


def _make_ir(
    sentences: list[IRSentence] | None = None,
    missing_layers: list[str] | None = None,
) -> IntermediateRepresentation:
    return IntermediateRepresentation(
        original_text="test",
        normalized_text="test",
        sentences=sentences or [],
        missing_layers=missing_layers or [],
    )


# ---------------------------------------------------------------------------
# 9.1 Morphological feature extraction
# ---------------------------------------------------------------------------

class TestMorphologicalFeatures:
    def test_zero_tokens_returns_zeros(self):
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.verb_ratio == 0.0
        assert result.binyan_distribution == {}
        assert result.prefix_density == 0.0
        assert result.suffix_pronoun_ratio == 0.0
        assert result.morphological_ambiguity == 0.0

    def test_verb_ratio(self):
        tokens = [
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="ADJ")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.verb_ratio == pytest.approx(0.5)

    def test_binyan_distribution(self):
        tokens = [
            _make_ir_token(morph=_make_morph(pos="VERB", binyan="PAAL")),
            _make_ir_token(morph=_make_morph(pos="VERB", binyan="PIEL")),
            _make_ir_token(morph=_make_morph(pos="VERB", binyan="PAAL")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),  # no binyan
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.binyan_distribution == {"PAAL": 2, "PIEL": 1}

    def test_prefix_density(self):
        tokens = [
            _make_ir_token(prefixes=["ו", "ב"]),
            _make_ir_token(prefixes=[]),
            _make_ir_token(prefixes=["ל"]),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.prefix_density == pytest.approx(1.0)  # 3 prefixes / 3 tokens

    def test_suffix_pronoun_ratio(self):
        tokens = [
            _make_ir_token(suffix="ו"),
            _make_ir_token(suffix=None),
            _make_ir_token(suffix="ה"),
            _make_ir_token(suffix=None),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.suffix_pronoun_ratio == pytest.approx(0.5)

    def test_morphological_ambiguity(self):
        tokens = [
            _make_ir_token(morph=_make_morph(ambiguity_count=2)),
            _make_ir_token(morph=_make_morph(ambiguity_count=4)),
            _make_ir_token(morph=_make_morph(ambiguity_count=6)),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.morphological_ambiguity == pytest.approx(4.0)

    def test_ambiguity_skips_tokens_without_morph(self):
        tokens = [
            _make_ir_token(morph=_make_morph(ambiguity_count=3)),
            _make_ir_token(morph=None),  # no morph data
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.morphological_ambiguity == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 5.2 Morphological control features (agreement, entropy, construct)
# ---------------------------------------------------------------------------

class TestMorphologicalControlFeatures:
    """Tests for agreement_error_rate, binyan_entropy, and construct_ratio."""

    # --- agreement_error_rate ---

    def test_agreement_error_rate_with_gender_mismatch(self):
        """nsubj pair with gender mismatch is counted."""
        # dep token: nsubj, gender=Masc; head token: gender=Fem → mismatch
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep_node = _make_dep_node(id=2, head=1, deprel="nsubj")
        head_morph = _make_morph(pos="VERB", gender="Fem", number="Sing")
        dep_morph = _make_morph(pos="NOUN", gender="Masc", number="Sing")
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep_morph, dep_node=dep_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        # 1 pair, 1 mismatch → 1.0
        assert result == pytest.approx(1.0)

    def test_agreement_error_rate_with_number_mismatch(self):
        """amod pair with number mismatch (same gender) is counted."""
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep_node = _make_dep_node(id=2, head=1, deprel="amod")
        head_morph = _make_morph(pos="NOUN", gender="Masc", number="Plur")
        dep_morph = _make_morph(pos="ADJ", gender="Masc", number="Sing")
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep_morph, dep_node=dep_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        assert result == pytest.approx(1.0)

    def test_agreement_error_rate_no_mismatch(self):
        """Matching gender and number → 0.0 error rate."""
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep_node = _make_dep_node(id=2, head=1, deprel="nsubj")
        head_morph = _make_morph(pos="VERB", gender="Masc", number="Sing")
        dep_morph = _make_morph(pos="NOUN", gender="Masc", number="Sing")
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep_morph, dep_node=dep_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        assert result == pytest.approx(0.0)

    def test_agreement_error_rate_no_pairs(self):
        """No nsubj/amod pairs → 0.0."""
        node = _make_dep_node(id=1, head=0, deprel="root")
        tokens = [_make_ir_token(morph=_make_morph(), dep_node=node)]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        assert result == pytest.approx(0.0)

    def test_agreement_error_rate_none_gender_no_mismatch(self):
        """None gender on dependent → not counted as mismatch."""
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep_node = _make_dep_node(id=2, head=1, deprel="nsubj")
        head_morph = _make_morph(pos="VERB", gender="Masc", number="Sing")
        dep_morph = _make_morph(pos="NOUN", gender=None, number="Sing")
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep_morph, dep_node=dep_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        # gender is None → skip gender check; numbers match → no mismatch
        assert result == pytest.approx(0.0)

    def test_agreement_error_rate_none_number_no_mismatch(self):
        """None number on head → not counted as mismatch."""
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep_node = _make_dep_node(id=2, head=1, deprel="amod")
        head_morph = _make_morph(pos="NOUN", gender="Fem", number=None)
        dep_morph = _make_morph(pos="ADJ", gender="Fem", number="Plur")
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep_morph, dep_node=dep_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        # genders match; number None on head → skip number check → no mismatch
        assert result == pytest.approx(0.0)

    def test_agreement_error_rate_mixed_pairs(self):
        """Two pairs: one mismatch, one match → 0.5."""
        head_node = _make_dep_node(id=1, head=0, deprel="root")
        dep1_node = _make_dep_node(id=2, head=1, deprel="nsubj")
        dep2_node = _make_dep_node(id=3, head=1, deprel="amod")
        head_morph = _make_morph(pos="VERB", gender="Masc", number="Sing")
        dep1_morph = _make_morph(pos="NOUN", gender="Fem", number="Sing")  # gender mismatch
        dep2_morph = _make_morph(pos="ADJ", gender="Masc", number="Sing")  # match
        tokens = [
            _make_ir_token(morph=head_morph, dep_node=head_node),
            _make_ir_token(morph=dep1_morph, dep_node=dep1_node),
            _make_ir_token(morph=dep2_morph, dep_node=dep2_node),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _agreement_error_rate(sentences)
        assert result == pytest.approx(0.5)

    # --- binyan_entropy ---

    def test_binyan_entropy_known_distribution(self):
        """Entropy of uniform distribution over 2 binyans = ln(2)."""
        import math
        dist = {"PAAL": 5, "PIEL": 5}
        result = _binyan_entropy(dist)
        assert result == pytest.approx(math.log(2))

    def test_binyan_entropy_single_binyan(self):
        """Single binyan → entropy = 0.0."""
        dist = {"PAAL": 10}
        result = _binyan_entropy(dist)
        assert result == pytest.approx(0.0)

    def test_binyan_entropy_empty_distribution(self):
        """Empty distribution → 0.0."""
        assert _binyan_entropy({}) == 0.0

    def test_binyan_entropy_none_distribution(self):
        """None distribution → 0.0."""
        assert _binyan_entropy(None) == 0.0

    # --- construct_ratio ---

    def test_construct_ratio_adjacent_nouns(self):
        """Two adjacent nouns in 4 tokens → 1/4."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="ADJ")),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _construct_ratio(sentences, 4)
        assert result == pytest.approx(0.25)

    def test_construct_ratio_no_adjacent_nouns(self):
        """No adjacent nouns → 0.0."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),
        ]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _construct_ratio(sentences, 3)
        assert result == pytest.approx(0.0)

    def test_construct_ratio_zero_tokens(self):
        """Zero total tokens → 0.0."""
        result = _construct_ratio([], 0)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9.3 Syntactic feature extraction
# ---------------------------------------------------------------------------

class TestSyntacticFeatures:
    def test_zero_sentences_returns_zeros(self):
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.avg_sentence_length == 0.0
        assert result.avg_tree_depth == 0.0
        assert result.max_tree_depth == 0.0
        assert result.avg_dependency_distance == 0.0
        assert result.clauses_per_sentence == 0.0

    def test_avg_sentence_length(self):
        s1 = IRSentence(tokens=[_make_ir_token()] * 4, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token()] * 6, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.avg_sentence_length == pytest.approx(5.0)

    def test_tree_depth_simple_chain(self):
        """Tree: 0 -> 1 -> 2 -> 3  (depth = 3)"""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=1, deprel="nsubj"),
            _make_dep_node(id=3, head=2, deprel="det"),
        ]
        tree = SentenceTree(nodes=nodes)
        tokens = [
            _make_ir_token(dep_node=nodes[0]),
            _make_ir_token(dep_node=nodes[1]),
            _make_ir_token(dep_node=nodes[2]),
        ]
        s = IRSentence(tokens=tokens, dep_tree=tree)
        depth = _compute_tree_depth(s)
        assert depth == 3.0

    def test_tree_depth_flat(self):
        """Tree: 0 -> 1, 0 -> 2, 0 -> 3  (depth = 1)"""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=0, deprel="nsubj"),
            _make_dep_node(id=3, head=0, deprel="obj"),
        ]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=[], dep_tree=tree)
        depth = _compute_tree_depth(s)
        assert depth == 1.0

    def test_avg_dependency_distance(self):
        nodes = [
            _make_dep_node(id=1, head=0),   # |1-0| = 1
            _make_dep_node(id=2, head=1),   # |2-1| = 1
            _make_dep_node(id=3, head=1),   # |3-1| = 2
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        s = IRSentence(tokens=tokens, dep_tree=SentenceTree(nodes=nodes))
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # mean([1, 1, 2]) = 4/3
        assert result.avg_dependency_distance == pytest.approx(4.0 / 3.0)

    def test_clauses_per_sentence(self):
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=1, deprel="ccomp"),
            _make_dep_node(id=3, head=1, deprel="nsubj"),
            _make_dep_node(id=4, head=2, deprel="advcl"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # 2 subordinate rels (ccomp, advcl) / 1 sentence
        assert result.clauses_per_sentence == pytest.approx(2.0)

    def test_no_dep_nodes_still_computes(self):
        """Tokens without dep_node should not crash dependency distance."""
        tokens = [_make_ir_token(dep_node=None), _make_ir_token(dep_node=None)]
        s = IRSentence(tokens=tokens, dep_tree=None)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.avg_dependency_distance == 0.0

    # --- Syntactic complexity features (Task 6) ---

    def test_subordinate_clause_ratio_known_deps(self):
        """Known mix of clause rels: 2 subordinate out of 3 clause rels."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=1, deprel="ccomp"),
            _make_dep_node(id=3, head=1, deprel="advcl"),
            _make_dep_node(id=4, head=2, deprel="nsubj"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # clause rels: root, ccomp, advcl → 3 total; subordinate: ccomp, advcl → 2
        assert result.subordinate_clause_ratio == pytest.approx(2.0 / 3.0)

    def test_subordinate_clause_ratio_no_clause_rels(self):
        """No clause rels at all → 0.0."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="nsubj"),
            _make_dep_node(id=2, head=1, deprel="det"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.subordinate_clause_ratio == pytest.approx(0.0)

    def test_subordinate_clause_ratio_only_root(self):
        """Only root clause rel, no subordinate → 0.0."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=1, deprel="nsubj"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # 1 clause rel (root), 0 subordinate → 0/1 = 0.0
        assert result.subordinate_clause_ratio == pytest.approx(0.0)

    def test_right_branching_ratio_known_positions(self):
        """Known token/head positions: 2 right-branching out of 3 non-root."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),   # root, excluded
            _make_dep_node(id=2, head=1, deprel="nsubj"),  # 2 > 1 → right
            _make_dep_node(id=3, head=1, deprel="obj"),    # 3 > 1 → right
            _make_dep_node(id=4, head=3, deprel="det"),    # 4 > 3 → right
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # non-root: (2,1), (3,1), (4,3) → all right-branching → 3/3 = 1.0
        assert result.right_branching_ratio == pytest.approx(1.0)

    def test_right_branching_ratio_mixed(self):
        """Mix of left and right branching."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),   # root, excluded
            _make_dep_node(id=2, head=3, deprel="det"),    # 2 < 3 → left
            _make_dep_node(id=3, head=1, deprel="nsubj"),  # 3 > 1 → right
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # non-root: (2,3) left, (3,1) right → 1/2 = 0.5
        assert result.right_branching_ratio == pytest.approx(0.5)

    def test_right_branching_ratio_no_non_root(self):
        """Only root deps → 0.0."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.right_branching_ratio == pytest.approx(0.0)

    def test_dependency_distance_variance_known_distances(self):
        """Known distances: |2-1|=1, |3-1|=2, |5-2|=3 → variance([1,2,3]) = 1.0."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),   # root, excluded
            _make_dep_node(id=2, head=1, deprel="nsubj"),  # |2-1| = 1
            _make_dep_node(id=3, head=1, deprel="obj"),    # |3-1| = 2
            _make_dep_node(id=5, head=2, deprel="det"),    # |5-2| = 3
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        # sample variance of [1, 2, 3] = 1.0
        assert result.dependency_distance_variance == pytest.approx(1.0)

    def test_dependency_distance_variance_single_non_root(self):
        """Single non-root dep → 0.0 (need ≥ 2 for sample variance)."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
            _make_dep_node(id=2, head=1, deprel="nsubj"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.dependency_distance_variance == pytest.approx(0.0)

    def test_dependency_distance_variance_zero_non_root(self):
        """Zero non-root deps → 0.0."""
        nodes = [
            _make_dep_node(id=1, head=0, deprel="root"),
        ]
        tokens = [_make_ir_token(dep_node=n) for n in nodes]
        tree = SentenceTree(nodes=nodes)
        s = IRSentence(tokens=tokens, dep_tree=tree)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.dependency_distance_variance == pytest.approx(0.0)

    def test_syntactic_complexity_none_when_yap_missing(self):
        """All three new features are None when yap is in missing_layers."""
        morph = _make_morph(pos="NOUN")
        token = _make_ir_token(surface="test", morph=morph, dep_node=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["yap"])
        result = extract_features(ir)
        assert result.syntax.subordinate_clause_ratio is None
        assert result.syntax.right_branching_ratio is None
        assert result.syntax.dependency_distance_variance is None

    def test_syntactic_complexity_zero_sentences(self):
        """Zero sentences → 0.0 for all three new features."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.subordinate_clause_ratio == 0.0
        assert result.right_branching_ratio == 0.0
        assert result.dependency_distance_variance == 0.0


# ---------------------------------------------------------------------------
# 9.5 Lexical feature extraction
# ---------------------------------------------------------------------------

class TestLexicalFeatures:
    def test_zero_tokens_returns_zeros(self):
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        assert result.type_token_ratio == 0.0
        assert result.hapax_ratio == 0.0
        assert result.avg_token_length == 0.0
        assert result.lemma_diversity == 0.0

    def test_type_token_ratio(self):
        tokens = [
            _make_ir_token(surface="hello"),
            _make_ir_token(surface="world"),
            _make_ir_token(surface="hello"),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        assert result.type_token_ratio == pytest.approx(2.0 / 3.0)

    def test_hapax_ratio(self):
        tokens = [
            _make_ir_token(surface="a"),
            _make_ir_token(surface="b"),
            _make_ir_token(surface="a"),
            _make_ir_token(surface="c"),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        # hapax: "b" and "c" (freq=1), total=4
        assert result.hapax_ratio == pytest.approx(0.5)

    def test_avg_token_length(self):
        tokens = [
            _make_ir_token(surface="ab"),    # len 2
            _make_ir_token(surface="cdef"),  # len 4
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        assert result.avg_token_length == pytest.approx(3.0)

    def test_lemma_diversity_with_morph(self):
        tokens = [
            _make_ir_token(surface="ran", morph=_make_morph(lemma="run")),
            _make_ir_token(surface="runs", morph=_make_morph(lemma="run")),
            _make_ir_token(surface="jumped", morph=_make_morph(lemma="jump")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        # unique lemmas: {"run", "jump"} = 2, total = 3
        assert result.lemma_diversity == pytest.approx(2.0 / 3.0)

    def test_lemma_diversity_falls_back_to_surface(self):
        tokens = [
            _make_ir_token(surface="abc", morph=None),
            _make_ir_token(surface="def", morph=None),
            _make_ir_token(surface="abc", morph=None),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        # unique surfaces as lemmas: {"abc", "def"} = 2, total = 3
        assert result.lemma_diversity == pytest.approx(2.0 / 3.0)

    # --- Lexical sophistication features (Task 4) ---

    def test_rare_word_ratio_none_when_no_freq_dict(self):
        """rare_word_ratio is None when freq_dict is None."""
        tokens = [_make_ir_token(surface="word", morph=_make_morph(lemma="word"))]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache, freq_dict=None)
        assert result.rare_word_ratio is None

    def test_rare_word_ratio_zero_tokens_with_freq_dict(self):
        """rare_word_ratio is 0.0 when total tokens is zero and freq_dict provided."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache, freq_dict={"word": 100})
        assert result.rare_word_ratio == 0.0

    def test_rare_word_ratio_with_known_frequencies(self):
        """rare_word_ratio counts tokens whose lemma freq < 5."""
        tokens = [
            _make_ir_token(surface="common", morph=_make_morph(lemma="common")),
            _make_ir_token(surface="rare1", morph=_make_morph(lemma="rare1")),
            _make_ir_token(surface="rare2", morph=_make_morph(lemma="rare2")),
            _make_ir_token(surface="common2", morph=_make_morph(lemma="common2")),
        ]
        freq_dict = {"common": 100, "rare1": 2, "rare2": 0, "common2": 50}
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache, freq_dict=freq_dict)
        # rare1 (2 < 5) and rare2 (0 < 5) are rare → 2/4
        assert result.rare_word_ratio == pytest.approx(0.5)

    def test_rare_word_ratio_lemma_fallback_to_surface(self):
        """When morph is None, surface form is used as lemma for rare word lookup."""
        tokens = [
            _make_ir_token(surface="known", morph=None),
            _make_ir_token(surface="unknown", morph=None),
        ]
        freq_dict = {"known": 100, "unknown": 1}
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache, freq_dict=freq_dict)
        # "known" freq=100 (not rare), "unknown" freq=1 (rare) → 1/2
        assert result.rare_word_ratio == pytest.approx(0.5)

    def test_content_word_ratio_with_known_pos(self):
        """content_word_ratio counts NOUN/VERB/ADJ/ADV tokens."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="ADJ")),
            _make_ir_token(morph=_make_morph(pos="DET")),
            _make_ir_token(morph=_make_morph(pos="PUNCT")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        # 3 content words (NOUN, VERB, ADJ) / 5 total
        assert result.content_word_ratio == pytest.approx(3.0 / 5.0)

    def test_content_word_ratio_zero_tokens(self):
        """content_word_ratio is 0.0 when total tokens is zero."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        assert result.content_word_ratio == 0.0


# ---------------------------------------------------------------------------
# 9.7 Structural feature extraction
# ---------------------------------------------------------------------------

class TestStructuralFeatures:
    def test_zero_sentences_returns_zeros(self):
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.sentence_length_variance == 0.0
        assert result.long_sentence_ratio == 0.0

    def test_single_sentence_variance_is_zero(self):
        s = IRSentence(tokens=[_make_ir_token()] * 5, dep_tree=None)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.sentence_length_variance == 0.0

    def test_variance_two_sentences(self):
        s1 = IRSentence(tokens=[_make_ir_token()] * 4, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token()] * 6, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # variance([4, 6]) = 2.0 (sample variance)
        assert result.sentence_length_variance == pytest.approx(2.0)

    def test_long_sentence_ratio(self):
        s1 = IRSentence(tokens=[_make_ir_token()] * 10, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token()] * 25, dep_tree=None)
        s3 = IRSentence(tokens=[_make_ir_token()] * 5, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2, s3])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache, long_sentence_threshold=20)
        # 1 sentence > 20 out of 3
        assert result.long_sentence_ratio == pytest.approx(1.0 / 3.0)

    def test_long_sentence_ratio_custom_threshold(self):
        s1 = IRSentence(tokens=[_make_ir_token()] * 3, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token()] * 4, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache, long_sentence_threshold=3)
        # 1 sentence > 3 out of 2
        assert result.long_sentence_ratio == pytest.approx(0.5)

    # --- Structural fluency features (Task 3) ---

    def test_punctuation_ratio_zero_tokens(self):
        """Zero tokens → 0.0 for punctuation_ratio."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.punctuation_ratio == 0.0

    def test_punctuation_ratio_normal(self):
        """Known PUNCT tokens produce correct ratio."""
        tokens = [
            _make_ir_token(surface=".", morph=_make_morph(pos="PUNCT")),
            _make_ir_token(surface="word", morph=_make_morph(pos="NOUN")),
            _make_ir_token(surface="!", morph=_make_morph(pos="PUNCT")),
            _make_ir_token(surface="other", morph=_make_morph(pos="VERB")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # 2 PUNCT / 4 total = 0.5
        assert result.punctuation_ratio == pytest.approx(0.5)

    def test_punctuation_ratio_no_morph(self):
        """Tokens without morph are not counted as PUNCT."""
        tokens = [
            _make_ir_token(surface=".", morph=None),
            _make_ir_token(surface="word", morph=_make_morph(pos="NOUN")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # 0 PUNCT / 2 total = 0.0
        assert result.punctuation_ratio == pytest.approx(0.0)

    def test_short_sentence_ratio_zero_sentences(self):
        """Zero sentences → 0.0 for short_sentence_ratio."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.short_sentence_ratio == 0.0

    def test_short_sentence_ratio_normal(self):
        """Sentences with < 3 tokens are counted as short."""
        s1 = IRSentence(tokens=[_make_ir_token()] * 2, dep_tree=None)  # short
        s2 = IRSentence(tokens=[_make_ir_token()] * 5, dep_tree=None)  # not short
        s3 = IRSentence(tokens=[_make_ir_token()] * 1, dep_tree=None)  # short
        ir = _make_ir(sentences=[s1, s2, s3])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # 2 short / 3 total
        assert result.short_sentence_ratio == pytest.approx(2.0 / 3.0)

    def test_missing_terminal_punctuation_ratio_zero_sentences(self):
        """Zero sentences → 0.0 for missing_terminal_punctuation_ratio."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.missing_terminal_punctuation_ratio == 0.0

    def test_missing_terminal_punctuation_empty_sentence(self):
        """Empty sentence counts as missing terminal punctuation."""
        s_empty = IRSentence(tokens=[], dep_tree=None)
        s_ok = IRSentence(
            tokens=[_make_ir_token(surface="word"), _make_ir_token(surface=".")],
            dep_tree=None,
        )
        ir = _make_ir(sentences=[s_empty, s_ok])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # 1 missing (empty) / 2 total
        assert result.missing_terminal_punctuation_ratio == pytest.approx(0.5)

    def test_missing_terminal_punctuation_normal(self):
        """Sentences ending with terminal punct vs not."""
        s1 = IRSentence(
            tokens=[_make_ir_token(surface="hello"), _make_ir_token(surface=".")],
            dep_tree=None,
        )
        s2 = IRSentence(
            tokens=[_make_ir_token(surface="world"), _make_ir_token(surface="!")],
            dep_tree=None,
        )
        s3 = IRSentence(
            tokens=[_make_ir_token(surface="no"), _make_ir_token(surface="punct")],
            dep_tree=None,
        )
        s4 = IRSentence(
            tokens=[_make_ir_token(surface="ellipsis"), _make_ir_token(surface="…")],
            dep_tree=None,
        )
        ir = _make_ir(sentences=[s1, s2, s3, s4])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        # s3 is missing terminal punct → 1/4
        assert result.missing_terminal_punctuation_ratio == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 9.9 Top-level extract_features with missing layers
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_all_layers_present(self):
        morph = _make_morph(pos="VERB", binyan="PAAL")
        node = _make_dep_node(id=1, head=0, deprel="root")
        token = _make_ir_token(
            surface="test", morph=morph, dep_node=node,
        )
        tree = SentenceTree(nodes=[node])
        s = IRSentence(tokens=[token], dep_tree=tree)
        ir = _make_ir(sentences=[s], missing_layers=[])
        result = extract_features(ir)

        assert isinstance(result, Features)
        assert result.morphology.verb_ratio is not None
        assert result.syntax.avg_sentence_length is not None
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None
        assert isinstance(result.discourse, DiscourseFeatures)
        assert isinstance(result.style, StyleFeatures)

    def test_stanza_missing_sets_morph_to_none(self):
        token = _make_ir_token(surface="test", morph=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["stanza"])
        result = extract_features(ir)

        assert result.morphology.verb_ratio is None
        assert result.morphology.binyan_distribution is None
        assert result.morphology.prefix_density is None
        assert result.morphology.suffix_pronoun_ratio is None
        assert result.morphology.morphological_ambiguity is None
        assert result.morphology.agreement_error_rate is None
        assert result.morphology.binyan_entropy is None
        assert result.morphology.construct_ratio is None
        # Lexical and structural should still be computed
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None

    def test_yap_missing_sets_syntax_to_none(self):
        morph = _make_morph(pos="NOUN")
        token = _make_ir_token(surface="test", morph=morph, dep_node=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["yap"])
        result = extract_features(ir)

        assert result.syntax.avg_sentence_length is None
        assert result.syntax.avg_tree_depth is None
        assert result.syntax.max_tree_depth is None
        assert result.syntax.avg_dependency_distance is None
        assert result.syntax.clauses_per_sentence is None
        assert result.syntax.subordinate_clause_ratio is None
        assert result.syntax.right_branching_ratio is None
        assert result.syntax.dependency_distance_variance is None
        # Morphological should still be computed
        assert result.morphology.verb_ratio is not None

    def test_both_missing(self):
        token = _make_ir_token(surface="test", morph=None, dep_node=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["stanza", "yap"])
        result = extract_features(ir)

        assert result.morphology.verb_ratio is None
        assert result.syntax.avg_sentence_length is None
        # Lexical and structural still computed
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None

    def test_empty_ir(self):
        ir = _make_ir(sentences=[], missing_layers=[])
        result = extract_features(ir)

        assert result.morphology.verb_ratio == 0.0
        assert result.syntax.avg_sentence_length == 0.0
        assert result.lexicon.type_token_ratio == 0.0
        assert result.structure.sentence_length_variance == 0.0

    def test_long_sentence_threshold_passed_through(self):
        tokens = [_make_ir_token()] * 15
        s = IRSentence(tokens=tokens, dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=[])

        result_low = extract_features(ir, long_sentence_threshold=10)
        result_high = extract_features(ir, long_sentence_threshold=20)

        assert result_low.structure.long_sentence_ratio == pytest.approx(1.0)
        assert result_high.structure.long_sentence_ratio == pytest.approx(0.0)

    def test_freq_dict_parameter(self):
        token = _make_ir_token(surface="test", morph=_make_morph(pos="NOUN"))
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=[])

        result_no_dict = extract_features(ir)
        assert result_no_dict.lexicon.rare_word_ratio is None

        result_with_dict = extract_features(ir, freq_dict={"test": 100})
        assert result_with_dict.lexicon.rare_word_ratio is not None


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================

from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st
from collections import Counter
from statistics import mean, variance

# ---------------------------------------------------------------------------
# Hypothesis strategies for generating random IR data
# ---------------------------------------------------------------------------

_POS_TAGS = ["VERB", "NOUN", "ADJ", "ADV", "ADP", "DET", "PRON", "CONJ", "NUM", "PUNCT"]
_BINYAN_VALUES = [None, "PAAL", "PIEL", "HIFIL", "HITPAEL", "NIFAL", "PUAL", "HUFAL"]
_PREFIX_CHARS = ["ו", "ב", "ל", "כ", "ה", "מ", "ש"]
_SUFFIX_VALUES = [None, "ו", "ה", "ם", "ן", "י", "נו", "כם", "כן", "הם", "הן"]
_DEPRELS = ["root", "nsubj", "obj", "det", "amod", "advmod", "case", "rcmod", "ccomp", "xcomp", "advcl", "complm"]
_SUBORDINATE_RELS = frozenset({"rcmod", "ccomp", "xcomp", "advcl", "complm"})


@st.composite
def st_morph(draw):
    """Generate a random MorphAnalysis."""
    pos = draw(st.sampled_from(_POS_TAGS))
    binyan = draw(st.sampled_from(_BINYAN_VALUES))
    ambiguity = draw(st.integers(min_value=1, max_value=10))
    suffix = draw(st.sampled_from(_SUFFIX_VALUES))
    surface = draw(st.text(min_size=1, max_size=8, alphabet=st.characters(whitelist_categories=("L",))))
    lemma = draw(st.text(min_size=1, max_size=8, alphabet=st.characters(whitelist_categories=("L",))))
    return _make_morph(
        surface=surface,
        lemma=lemma,
        pos=pos,
        binyan=binyan,
        ambiguity_count=ambiguity,
        suffix=suffix,
    )


@st.composite
def st_ir_token_with_morph(draw):
    """Generate a random IRToken with morph data and optional prefixes/suffix."""
    morph = draw(st_morph())
    prefixes = draw(st.lists(st.sampled_from(_PREFIX_CHARS), min_size=0, max_size=3))
    suffix = draw(st.sampled_from(_SUFFIX_VALUES))
    surface = morph.surface
    return _make_ir_token(
        surface=surface,
        morph=morph,
        prefixes=prefixes,
        suffix=suffix,
    )


@st.composite
def st_ir_sentence_with_dep_tree(draw, min_tokens=1, max_tokens=10):
    """Generate a random IRSentence with dep_tree nodes (valid head indices)."""
    n = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
    tokens = []
    nodes = []
    for i in range(1, n + 1):
        morph = draw(st_morph())
        deprel = draw(st.sampled_from(_DEPRELS))
        # head must be in [0, n], and not self-referencing
        head = draw(st.integers(min_value=0, max_value=n).filter(lambda h, _i=i: h != _i))
        node = _make_dep_node(id=i, form=morph.surface, head=head, deprel=deprel)
        prefixes = draw(st.lists(st.sampled_from(_PREFIX_CHARS), min_size=0, max_size=3))
        suffix = draw(st.sampled_from(_SUFFIX_VALUES))
        token = _make_ir_token(
            surface=morph.surface,
            morph=morph,
            dep_node=node,
            prefixes=prefixes,
            suffix=suffix,
        )
        tokens.append(token)
        nodes.append(node)
    tree = SentenceTree(nodes=nodes)
    return IRSentence(tokens=tokens, dep_tree=tree)


@st.composite
def st_ir_nonempty(draw, min_sentences=1, max_sentences=5):
    """Generate a random IR with at least one sentence."""
    n_sent = draw(st.integers(min_value=min_sentences, max_value=max_sentences))
    sentences = [draw(st_ir_sentence_with_dep_tree()) for _ in range(n_sent)]
    return _make_ir(sentences=sentences, missing_layers=[])


# ---------------------------------------------------------------------------
# Property 12: Morphological feature formulas
# Feature: hebrew-linguistic-profiling-engine, Property 12
# ---------------------------------------------------------------------------

class TestProperty12MorphologicalFeatureFormulas:
    """**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_verb_ratio_formula(self, ir):
        """verb_ratio = count(POS=VERB) / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        expected = sum(1 for t in tokens if t.morph is not None and t.morph.pos == "VERB") / total
        assert result.verb_ratio == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_binyan_distribution_formula(self, ir):
        """binyan_distribution matches actual binyan values."""
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        tokens = _all_tokens(ir)
        expected: dict[str, int] = {}
        for t in tokens:
            if t.morph is not None and t.morph.binyan is not None:
                expected[t.morph.binyan] = expected.get(t.morph.binyan, 0) + 1
        assert result.binyan_distribution == expected

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_prefix_density_formula(self, ir):
        """prefix_density = total_prefixes / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        expected = sum(len(t.prefixes) for t in tokens) / total
        assert result.prefix_density == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_suffix_pronoun_ratio_formula(self, ir):
        """suffix_pronoun_ratio = count(tokens_with_suffix) / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        expected = sum(1 for t in tokens if t.suffix is not None) / total
        assert result.suffix_pronoun_ratio == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_morphological_ambiguity_formula(self, ir):
        """morphological_ambiguity = mean(ambiguity_counts) for tokens with morph."""
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        tokens = _all_tokens(ir)
        ambiguity_values = [t.morph.ambiguity_count for t in tokens if t.morph is not None]
        expected = mean(ambiguity_values) if ambiguity_values else 0.0
        assert result.morphological_ambiguity == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_zero_tokens_all_zero(self, data):
        """For zero tokens, all morphological features are 0.0."""
        ir = _make_ir(sentences=[], missing_layers=[])
        cache = _build_cache(ir)
        result = _extract_morphological(ir, cache)
        assert result.verb_ratio == 0.0
        assert result.binyan_distribution == {}
        assert result.prefix_density == 0.0
        assert result.suffix_pronoun_ratio == 0.0
        assert result.morphological_ambiguity == 0.0


# ---------------------------------------------------------------------------
# Property 13: Syntactic feature formulas
# Feature: hebrew-linguistic-profiling-engine, Property 13
# ---------------------------------------------------------------------------

class TestProperty13SyntacticFeatureFormulas:
    """**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_avg_sentence_length_formula(self, ir):
        """avg_sentence_length = mean(token_counts)."""
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        expected = mean([len(s.tokens) for s in ir.sentences])
        assert result.avg_sentence_length == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_avg_tree_depth_formula(self, ir):
        """avg_tree_depth = mean(max_depths)."""
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        depths = [_compute_tree_depth(s) for s in ir.sentences]
        expected = mean(depths)
        assert result.avg_tree_depth == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_max_tree_depth_formula(self, ir):
        """max_tree_depth = max(max_depths)."""
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        depths = [_compute_tree_depth(s) for s in ir.sentences]
        expected = max(depths)
        assert result.max_tree_depth == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_avg_dependency_distance_formula(self, ir):
        """avg_dependency_distance = mean(|pos−head|) for tokens with dep_node."""
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        distances = []
        for s in ir.sentences:
            for t in s.tokens:
                if t.dep_node is not None:
                    distances.append(abs(t.dep_node.id - t.dep_node.head))
        expected = mean(distances) if distances else 0.0
        assert result.avg_dependency_distance == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_clauses_per_sentence_formula(self, ir):
        """clauses_per_sentence = count(subordinate_rels) / num_sentences."""
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        sub_count = 0
        for s in ir.sentences:
            for t in s.tokens:
                if t.dep_node is not None and t.dep_node.deprel in _SUBORDINATE_RELS:
                    sub_count += 1
        expected = sub_count / len(ir.sentences)
        assert result.clauses_per_sentence == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_zero_sentences_all_zero(self, data):
        """For zero sentences, all syntactic features are 0.0."""
        ir = _make_ir(sentences=[], missing_layers=[])
        cache = _build_cache(ir)
        result = _extract_syntactic(ir, cache)
        assert result.avg_sentence_length == 0.0
        assert result.avg_tree_depth == 0.0
        assert result.max_tree_depth == 0.0
        assert result.avg_dependency_distance == 0.0
        assert result.clauses_per_sentence == 0.0


# ---------------------------------------------------------------------------
# Property 14: Lexical feature formulas
# Feature: hebrew-linguistic-profiling-engine, Property 14
# ---------------------------------------------------------------------------

class TestProperty14LexicalFeatureFormulas:
    """**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_type_token_ratio_formula(self, ir):
        """type_token_ratio = unique_forms / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        surfaces = [t.surface for t in tokens]
        expected = len(set(surfaces)) / total
        assert result.type_token_ratio == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_hapax_ratio_formula(self, ir):
        """hapax_ratio = count(freq=1) / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        surfaces = [t.surface for t in tokens]
        freq = Counter(surfaces)
        hapax_count = sum(1 for c in freq.values() if c == 1)
        expected = hapax_count / total
        assert result.hapax_ratio == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_avg_token_length_formula(self, ir):
        """avg_token_length = mean(char_counts)."""
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        tokens = _all_tokens(ir)
        surfaces = [t.surface for t in tokens]
        expected = mean(len(s) for s in surfaces)
        assert result.avg_token_length == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_lemma_diversity_formula(self, ir):
        """lemma_diversity = unique_lemmas / total_tokens."""
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        tokens = _all_tokens(ir)
        total = len(tokens)
        lemmas = [t.morph.lemma if t.morph is not None else t.surface for t in tokens]
        expected = len(set(lemmas)) / total
        assert result.lemma_diversity == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_zero_tokens_all_zero(self, data):
        """For zero tokens, all lexical features are 0.0."""
        ir = _make_ir(sentences=[], missing_layers=[])
        cache = _build_cache(ir)
        result = _extract_lexical(ir, cache)
        assert result.type_token_ratio == 0.0
        assert result.hapax_ratio == 0.0
        assert result.avg_token_length == 0.0
        assert result.lemma_diversity == 0.0


# ---------------------------------------------------------------------------
# Property 15: Structural feature formulas
# Feature: hebrew-linguistic-profiling-engine, Property 15
# ---------------------------------------------------------------------------

class TestProperty15StructuralFeatureFormulas:
    """**Validates: Requirements 10.1, 10.2, 10.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty(min_sentences=2, max_sentences=5))
    def test_sentence_length_variance_formula(self, ir):
        """sentence_length_variance = variance(token_counts) for ≥2 sentences."""
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        lengths = [len(s.tokens) for s in ir.sentences]
        expected = variance(lengths)
        assert result.sentence_length_variance == pytest.approx(expected)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_variance_zero_for_fewer_than_two_sentences(self, data):
        """sentence_length_variance = 0.0 if fewer than 2 sentences."""
        n = data.draw(st.integers(min_value=0, max_value=1))
        if n == 0:
            ir = _make_ir(sentences=[], missing_layers=[])
        else:
            sent = data.draw(st_ir_sentence_with_dep_tree())
            ir = _make_ir(sentences=[sent], missing_layers=[])
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache)
        assert result.sentence_length_variance == 0.0

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty(), threshold=st.integers(min_value=1, max_value=20))
    def test_long_sentence_ratio_formula(self, ir, threshold):
        """long_sentence_ratio = count(sentences > threshold) / num_sentences."""
        cache = _build_cache(ir)
        result = _extract_structural(ir, cache, long_sentence_threshold=threshold)
        lengths = [len(s.tokens) for s in ir.sentences]
        long_count = sum(1 for l in lengths if l > threshold)
        expected = long_count / len(ir.sentences)
        assert result.long_sentence_ratio == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Property 24: Feature extraction with missing IR layers
# Feature: hebrew-linguistic-profiling-engine, Property 24
# ---------------------------------------------------------------------------

class TestProperty24MissingIRLayers:
    """**Validates: Requirements 16.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_stanza_missing_sets_morph_none(self, ir):
        """When stanza is missing, morphological features are None, others computed."""
        ir.missing_layers = ["stanza"]
        result = extract_features(ir)
        assert result.morphology.verb_ratio is None
        assert result.morphology.binyan_distribution is None
        assert result.morphology.prefix_density is None
        assert result.morphology.suffix_pronoun_ratio is None
        assert result.morphology.morphological_ambiguity is None
        assert result.morphology.agreement_error_rate is None
        assert result.morphology.binyan_entropy is None
        assert result.morphology.construct_ratio is None
        # Lexical and structural should still be computed
        assert result.lexicon is not None
        assert result.lexicon.type_token_ratio is not None
        assert result.structure is not None
        assert result.structure.sentence_length_variance is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_yap_missing_sets_syntax_none(self, ir):
        """When yap is missing, syntactic features are None, others computed."""
        ir.missing_layers = ["yap"]
        result = extract_features(ir)
        assert result.syntax.avg_sentence_length is None
        assert result.syntax.avg_tree_depth is None
        assert result.syntax.max_tree_depth is None
        assert result.syntax.avg_dependency_distance is None
        assert result.syntax.clauses_per_sentence is None
        assert result.syntax.subordinate_clause_ratio is None
        assert result.syntax.right_branching_ratio is None
        assert result.syntax.dependency_distance_variance is None
        # Morphological should still be computed
        assert result.morphology is not None
        assert result.morphology.verb_ratio is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_both_missing_sets_morph_and_syntax_none(self, ir):
        """When both stanza and yap are missing, morph and syntax are None."""
        ir.missing_layers = ["stanza", "yap"]
        result = extract_features(ir)
        assert result.morphology.verb_ratio is None
        assert result.syntax.avg_sentence_length is None
        # Lexical and structural still computed
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_no_missing_layers_all_computed(self, ir):
        """When no layers are missing, all features are computed (not None)."""
        ir.missing_layers = []
        result = extract_features(ir)
        assert result.morphology.verb_ratio is not None
        assert result.syntax.avg_sentence_length is not None
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_available_layers_always_computed(self, ir, missing):
        """Features for available layers are always computed regardless of missing layers."""
        ir.missing_layers = missing
        result = extract_features(ir)
        # Lexical and structural are always computed (never depend on stanza/yap)
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None
        # Morphology depends on stanza
        if "stanza" in missing:
            assert result.morphology.verb_ratio is None
        else:
            assert result.morphology.verb_ratio is not None
        # Syntax depends on yap
        if "yap" in missing:
            assert result.syntax.avg_sentence_length is None
        else:
            assert result.syntax.avg_sentence_length is not None


# ---------------------------------------------------------------------------
# Property 11: Cache consistency
# Feature: feature-extraction-enhancements, Property 11
# ---------------------------------------------------------------------------

class TestProperty11CacheConsistency:
    """**Validates: Requirements 1.2, 1.3, 1.4**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_total_tokens_equals_sum_of_sentence_token_counts(self, ir):
        """cache.total_tokens == sum(len(s.tokens) for s in ir.sentences)."""
        cache = _build_cache(ir)
        expected = sum(len(s.tokens) for s in ir.sentences)
        assert cache.total_tokens == expected

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_sentence_lengths_count_matches_sentences(self, ir):
        """len(cache.sentence_lengths) == len(ir.sentences)."""
        cache = _build_cache(ir)
        assert len(cache.sentence_lengths) == len(ir.sentences)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_lemma_sets_count_matches_sentences(self, ir):
        """len(cache.lemma_sets_per_sentence) == len(ir.sentences)."""
        cache = _build_cache(ir)
        assert len(cache.lemma_sets_per_sentence) == len(ir.sentences)

    def test_empty_ir_total_tokens_zero(self):
        """Empty IR produces cache with total_tokens == 0."""
        ir = _make_ir(sentences=[], missing_layers=[])
        cache = _build_cache(ir)
        assert cache.total_tokens == 0
        assert len(cache.sentence_lengths) == 0
        assert len(cache.lemma_sets_per_sentence) == 0

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_sentence_lengths_values_match(self, ir):
        """Each entry in cache.sentence_lengths matches the actual token count of that sentence."""
        cache = _build_cache(ir)
        for i, s in enumerate(ir.sentences):
            assert cache.sentence_lengths[i] == len(s.tokens)


# ---------------------------------------------------------------------------
# Property 10: Rare word fallback
# Feature: feature-extraction-enhancements, Property 10
# ---------------------------------------------------------------------------

class TestProperty10RareWordFallback:
    """**Validates: Requirements 4.2, 14.2**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_rare_word_ratio_none_iff_no_freq_dict(self, ir):
        """rare_word_ratio is None iff freq_dict is None."""
        result_no_dict = extract_features(ir, freq_dict=None)
        assert result_no_dict.lexicon.rare_word_ratio is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_rare_word_ratio_not_none_with_freq_dict(self, ir):
        """rare_word_ratio is not None when freq_dict is provided."""
        freq_dict = {"common": 100}
        result_with_dict = extract_features(ir, freq_dict=freq_dict)
        assert result_with_dict.lexicon.rare_word_ratio is not None


# ---------------------------------------------------------------------------
# 8.2 Discourse cohesion feature unit tests
# ---------------------------------------------------------------------------

class TestDiscourseCohesionFeatures:
    """Unit tests for discourse cohesion features (Task 8.2)."""

    # --- connective_ratio ---

    def test_connective_ratio_with_known_connectives(self):
        """Known connective tokens produce correct ratio per sentence."""
        tokens = [
            _make_ir_token(surface="אבל"),  # connective
            _make_ir_token(surface="word"),
            _make_ir_token(surface="כי"),   # connective
            _make_ir_token(surface="other"),
        ]
        s = IRSentence(tokens=tokens, dep_tree=None)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        # 2 connectives / 1 sentence = 2.0
        assert result.connective_ratio == pytest.approx(2.0)

    def test_connective_ratio_multiple_sentences(self):
        """Connective ratio is connective count / number of sentences."""
        s1 = IRSentence(tokens=[_make_ir_token(surface="אבל"), _make_ir_token(surface="word")], dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token(surface="hello")], dep_tree=None)
        ir = _make_ir(sentences=[s1, s2])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        # 1 connective / 2 sentences = 0.5
        assert result.connective_ratio == pytest.approx(0.5)

    def test_connective_ratio_zero_sentences(self):
        """Zero sentences → 0.0."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.connective_ratio == 0.0

    def test_connective_ratio_no_connectives(self):
        """No connective tokens → 0.0."""
        tokens = [_make_ir_token(surface="hello"), _make_ir_token(surface="world")]
        s = IRSentence(tokens=tokens, dep_tree=None)
        ir = _make_ir(sentences=[s])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.connective_ratio == 0.0

    # --- sentence_overlap ---

    def test_sentence_overlap_with_known_lemma_sets(self):
        """Adjacent sentences with overlapping lemmas produce correct Jaccard."""
        # Sentence 1 lemmas: {a, b, c}, Sentence 2 lemmas: {b, c, d}
        # Jaccard = |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        s1_tokens = [
            _make_ir_token(surface="a", morph=_make_morph(lemma="a")),
            _make_ir_token(surface="b", morph=_make_morph(lemma="b")),
            _make_ir_token(surface="c", morph=_make_morph(lemma="c")),
        ]
        s2_tokens = [
            _make_ir_token(surface="b", morph=_make_morph(lemma="b")),
            _make_ir_token(surface="c", morph=_make_morph(lemma="c")),
            _make_ir_token(surface="d", morph=_make_morph(lemma="d")),
        ]
        ir = _make_ir(sentences=[
            IRSentence(tokens=s1_tokens, dep_tree=None),
            IRSentence(tokens=s2_tokens, dep_tree=None),
        ])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.sentence_overlap == pytest.approx(0.5)

    def test_sentence_overlap_fewer_than_two_sentences(self):
        """Fewer than 2 sentences → 0.0."""
        tokens = [_make_ir_token(surface="a", morph=_make_morph(lemma="a"))]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.sentence_overlap == 0.0

    def test_sentence_overlap_zero_sentences(self):
        """Zero sentences → 0.0."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.sentence_overlap == 0.0

    def test_sentence_overlap_identical_lemma_sets(self):
        """Identical lemma sets → Jaccard = 1.0."""
        tokens1 = [_make_ir_token(surface="x", morph=_make_morph(lemma="x"))]
        tokens2 = [_make_ir_token(surface="x", morph=_make_morph(lemma="x"))]
        ir = _make_ir(sentences=[
            IRSentence(tokens=tokens1, dep_tree=None),
            IRSentence(tokens=tokens2, dep_tree=None),
        ])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.sentence_overlap == pytest.approx(1.0)

    def test_sentence_overlap_disjoint_lemma_sets(self):
        """Disjoint lemma sets → Jaccard = 0.0."""
        tokens1 = [_make_ir_token(surface="a", morph=_make_morph(lemma="a"))]
        tokens2 = [_make_ir_token(surface="b", morph=_make_morph(lemma="b"))]
        ir = _make_ir(sentences=[
            IRSentence(tokens=tokens1, dep_tree=None),
            IRSentence(tokens=tokens2, dep_tree=None),
        ])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.sentence_overlap == pytest.approx(0.0)

    # --- pronoun_to_noun_ratio ---

    def test_pronoun_to_noun_ratio_with_known_pos(self):
        """Known PRON and NOUN counts produce correct ratio."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="PRON")),
            _make_ir_token(morph=_make_morph(pos="PRON")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        # pron_ratio = 2/4 = 0.5, noun_ratio = 1/4 + 1e-10 = 0.25 + 1e-10
        expected = 0.5 / (0.25 + 1e-10)
        assert result.pronoun_to_noun_ratio == pytest.approx(expected)

    def test_pronoun_to_noun_ratio_zero_tokens(self):
        """Zero tokens → 0.0."""
        ir = _make_ir(sentences=[])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.pronoun_to_noun_ratio == 0.0

    def test_pronoun_to_noun_ratio_no_pronouns(self):
        """No PRON tokens → 0.0."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        assert result.pronoun_to_noun_ratio == pytest.approx(0.0)

    def test_pronoun_to_noun_ratio_no_nouns(self):
        """No NOUN tokens → large ratio (epsilon prevents division by zero)."""
        tokens = [
            _make_ir_token(morph=_make_morph(pos="PRON")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        ir = _make_ir(sentences=[IRSentence(tokens=tokens, dep_tree=None)])
        cache = _build_cache(ir)
        result = _extract_discourse(ir, cache)
        # pron_ratio = 0.5, noun_ratio = 0 + 1e-10
        expected = 0.5 / 1e-10
        assert result.pronoun_to_noun_ratio == pytest.approx(expected)

    # --- discourse features with missing layers ---

    def test_discourse_computed_with_stanza_missing(self):
        """Discourse features are computed even when stanza is missing."""
        token = _make_ir_token(surface="אבל", morph=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["stanza"])
        result = extract_features(ir)
        assert result.discourse is not None
        assert isinstance(result.discourse, DiscourseFeatures)
        assert result.discourse.connective_ratio is not None
        assert result.discourse.sentence_overlap is not None
        assert result.discourse.pronoun_to_noun_ratio is not None

    def test_discourse_computed_with_yap_missing(self):
        """Discourse features are computed even when yap is missing."""
        token = _make_ir_token(surface="כי", morph=_make_morph(pos="NOUN"))
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["yap"])
        result = extract_features(ir)
        assert result.discourse is not None
        assert isinstance(result.discourse, DiscourseFeatures)

    def test_discourse_computed_with_both_missing(self):
        """Discourse features are computed even when both layers are missing."""
        token = _make_ir_token(surface="test", morph=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["stanza", "yap"])
        result = extract_features(ir)
        assert result.discourse is not None
        assert isinstance(result.discourse, DiscourseFeatures)


# ---------------------------------------------------------------------------
# Property 7: Discourse always computed
# Feature: feature-extraction-enhancements, Property 7
# ---------------------------------------------------------------------------

class TestProperty7DiscourseAlwaysComputed:
    """**Validates: Requirements 6.8**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_discourse_never_none(self, ir, missing):
        """features.discourse is never None for any IR regardless of missing_layers."""
        ir.missing_layers = missing
        result = extract_features(ir)
        assert result.discourse is not None
        assert isinstance(result.discourse, DiscourseFeatures)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_discourse_fields_are_float_when_tokens_exist(self, ir):
        """All DiscourseFeatures fields are float (not None) when tokens exist."""
        ir.missing_layers = []
        result = extract_features(ir)
        assert isinstance(result.discourse.connective_ratio, float)
        assert isinstance(result.discourse.sentence_overlap, float)
        assert isinstance(result.discourse.pronoun_to_noun_ratio, float)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_discourse_fields_are_float_with_missing_layers(self, ir, missing):
        """All DiscourseFeatures fields are float even with missing layers."""
        ir.missing_layers = missing
        result = extract_features(ir)
        assert isinstance(result.discourse.connective_ratio, float)
        assert isinstance(result.discourse.sentence_overlap, float)
        assert isinstance(result.discourse.pronoun_to_noun_ratio, float)


# ---------------------------------------------------------------------------
# 9.2 Stylistic consistency feature unit tests
# ---------------------------------------------------------------------------

from hebrew_profiler.feature_extractor import (
    _extract_style,
    _sentence_length_trend,
    _pos_distribution_variance,
)


class TestStylisticConsistencyFeatures:
    """Unit tests for stylistic consistency features (Task 9.2)."""

    # --- sentence_length_trend ---

    def test_sentence_length_trend_increasing(self):
        """Increasing sentence lengths → positive slope."""
        # Lengths: [2, 4, 6] → slope should be 2.0
        # x_mean = 1.0, y_mean = 4.0
        # num = (0-1)(2-4) + (1-1)(4-4) + (2-1)(6-4) = 2 + 0 + 2 = 4
        # den = (0-1)^2 + (1-1)^2 + (2-1)^2 = 1 + 0 + 1 = 2
        # slope = 4/2 = 2.0
        result = _sentence_length_trend([2, 4, 6])
        assert result == pytest.approx(2.0)

    def test_sentence_length_trend_decreasing(self):
        """Decreasing sentence lengths → negative slope."""
        # Lengths: [6, 4, 2] → slope should be -2.0
        result = _sentence_length_trend([6, 4, 2])
        assert result == pytest.approx(-2.0)

    def test_sentence_length_trend_constant(self):
        """Constant sentence lengths → slope = 0.0."""
        result = _sentence_length_trend([5, 5, 5])
        assert result == pytest.approx(0.0)

    def test_sentence_length_trend_single_sentence(self):
        """Single sentence → 0.0."""
        result = _sentence_length_trend([10])
        assert result == 0.0

    def test_sentence_length_trend_zero_sentences(self):
        """Zero sentences → 0.0."""
        result = _sentence_length_trend([])
        assert result == 0.0

    def test_sentence_length_trend_two_sentences(self):
        """Two sentences: [3, 7] → slope = 4.0."""
        # x_mean = 0.5, y_mean = 5.0
        # num = (0-0.5)(3-5) + (1-0.5)(7-5) = 1 + 1 = 2
        # den = (0-0.5)^2 + (1-0.5)^2 = 0.25 + 0.25 = 0.5
        # slope = 2/0.5 = 4.0
        result = _sentence_length_trend([3, 7])
        assert result == pytest.approx(4.0)

    # --- pos_distribution_variance ---

    def test_pos_distribution_variance_known_distributions(self):
        """Two sentences with different POS distributions produce known variance."""
        # Sentence 1: 2 NOUN, 1 VERB → normalized: NOUN=2/3, VERB=1/3
        # Sentence 2: 1 NOUN, 2 VERB → normalized: NOUN=1/3, VERB=2/3
        # NOUN values: [2/3, 1/3] → sample variance = ((2/3-1/2)^2 + (1/3-1/2)^2) / 1 = 2*(1/6)^2 = 2/36 = 1/18
        # VERB values: [1/3, 2/3] → same variance = 1/18
        # mean = 1/18
        s1_tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        s2_tokens = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        sentences = [
            IRSentence(tokens=s1_tokens, dep_tree=None),
            IRSentence(tokens=s2_tokens, dep_tree=None),
        ]
        result = _pos_distribution_variance(sentences)
        assert result == pytest.approx(1.0 / 18.0)

    def test_pos_distribution_variance_identical_distributions(self):
        """Identical POS distributions → variance = 0.0."""
        tokens1 = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        tokens2 = [
            _make_ir_token(morph=_make_morph(pos="NOUN")),
            _make_ir_token(morph=_make_morph(pos="VERB")),
        ]
        sentences = [
            IRSentence(tokens=tokens1, dep_tree=None),
            IRSentence(tokens=tokens2, dep_tree=None),
        ]
        result = _pos_distribution_variance(sentences)
        assert result == pytest.approx(0.0)

    def test_pos_distribution_variance_single_sentence(self):
        """Single sentence → 0.0."""
        tokens = [_make_ir_token(morph=_make_morph(pos="NOUN"))]
        sentences = [IRSentence(tokens=tokens, dep_tree=None)]
        result = _pos_distribution_variance(sentences)
        assert result == 0.0

    def test_pos_distribution_variance_zero_sentences(self):
        """Zero sentences → 0.0."""
        result = _pos_distribution_variance([])
        assert result == 0.0

    def test_pos_distribution_variance_no_morph_tokens(self):
        """Sentences with no morph data → 0.0 (no POS tags found)."""
        tokens1 = [_make_ir_token(surface="a", morph=None)]
        tokens2 = [_make_ir_token(surface="b", morph=None)]
        sentences = [
            IRSentence(tokens=tokens1, dep_tree=None),
            IRSentence(tokens=tokens2, dep_tree=None),
        ]
        result = _pos_distribution_variance(sentences)
        assert result == 0.0

    # --- style features via extract_features with missing layers ---

    def test_style_computed_with_stanza_missing(self):
        """Style features are computed even when stanza is missing."""
        s1 = IRSentence(tokens=[_make_ir_token(surface="a")] * 3, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token(surface="b")] * 5, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2], missing_layers=["stanza"])
        result = extract_features(ir)
        assert result.style is not None
        assert isinstance(result.style, StyleFeatures)
        assert isinstance(result.style.sentence_length_trend, float)
        assert isinstance(result.style.pos_distribution_variance, float)

    def test_style_computed_with_yap_missing(self):
        """Style features are computed even when yap is missing."""
        s1 = IRSentence(tokens=[_make_ir_token(surface="a", morph=_make_morph(pos="NOUN"))] * 2, dep_tree=None)
        s2 = IRSentence(tokens=[_make_ir_token(surface="b", morph=_make_morph(pos="VERB"))] * 4, dep_tree=None)
        ir = _make_ir(sentences=[s1, s2], missing_layers=["yap"])
        result = extract_features(ir)
        assert result.style is not None
        assert isinstance(result.style, StyleFeatures)
        assert isinstance(result.style.sentence_length_trend, float)
        assert isinstance(result.style.pos_distribution_variance, float)

    def test_style_computed_with_both_missing(self):
        """Style features are computed even when both layers are missing."""
        token = _make_ir_token(surface="test", morph=None)
        s = IRSentence(tokens=[token], dep_tree=None)
        ir = _make_ir(sentences=[s], missing_layers=["stanza", "yap"])
        result = extract_features(ir)
        assert result.style is not None
        assert isinstance(result.style, StyleFeatures)


# ---------------------------------------------------------------------------
# Property 8: Style always computed
# Feature: feature-extraction-enhancements, Property 8
# ---------------------------------------------------------------------------

class TestProperty8StyleAlwaysComputed:
    """**Validates: Requirements 7.6, 8.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_style_never_none(self, ir, missing):
        """features.style is never None for any IR regardless of missing_layers."""
        ir.missing_layers = missing
        result = extract_features(ir)
        assert result.style is not None
        assert isinstance(result.style, StyleFeatures)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_style_fields_are_float_when_tokens_exist(self, ir):
        """All StyleFeatures fields are float (not None) when tokens exist."""
        ir.missing_layers = []
        result = extract_features(ir)
        assert isinstance(result.style.sentence_length_trend, float)
        assert isinstance(result.style.pos_distribution_variance, float)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_style_fields_are_float_with_missing_layers(self, ir, missing):
        """All StyleFeatures fields are float even with missing layers."""
        ir.missing_layers = missing
        result = extract_features(ir)
        assert isinstance(result.style.sentence_length_trend, float)
        assert isinstance(result.style.pos_distribution_variance, float)


# ---------------------------------------------------------------------------
# Property 2: Missing layer propagation (Stanza)
# Feature: feature-extraction-enhancements, Property 2
# ---------------------------------------------------------------------------

class TestProperty2StanzaMissing:
    """**Validates: Requirements 5.8, 8.1**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_all_morph_features_none_when_stanza_missing(self, ir):
        """All morph-dependent features are None when stanza is in missing_layers."""
        ir.missing_layers = ["stanza"]
        result = extract_features(ir)
        assert result.morphology.verb_ratio is None
        assert result.morphology.binyan_distribution is None
        assert result.morphology.prefix_density is None
        assert result.morphology.suffix_pronoun_ratio is None
        assert result.morphology.morphological_ambiguity is None
        assert result.morphology.agreement_error_rate is None
        assert result.morphology.binyan_entropy is None
        assert result.morphology.construct_ratio is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_non_morph_features_still_computed_when_stanza_missing(self, ir):
        """Lexical, structural, discourse, and style features are still computed."""
        ir.missing_layers = ["stanza"]
        result = extract_features(ir)
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None
        assert result.discourse.connective_ratio is not None
        assert result.style.sentence_length_trend is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_stanza_missing_with_yap_present_syntax_computed(self, ir):
        """When only stanza is missing, syntax features are still computed."""
        ir.missing_layers = ["stanza"]
        result = extract_features(ir)
        assert result.syntax.avg_sentence_length is not None


# ---------------------------------------------------------------------------
# Property 3: Missing layer propagation (YAP)
# Feature: feature-extraction-enhancements, Property 3
# ---------------------------------------------------------------------------

class TestProperty3YapMissing:
    """**Validates: Requirements 3.7, 8.2**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_all_dep_features_none_when_yap_missing(self, ir):
        """All dep-dependent features are None when yap is in missing_layers."""
        ir.missing_layers = ["yap"]
        result = extract_features(ir)
        assert result.syntax.avg_sentence_length is None
        assert result.syntax.avg_tree_depth is None
        assert result.syntax.max_tree_depth is None
        assert result.syntax.avg_dependency_distance is None
        assert result.syntax.clauses_per_sentence is None
        assert result.syntax.subordinate_clause_ratio is None
        assert result.syntax.right_branching_ratio is None
        assert result.syntax.dependency_distance_variance is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_non_dep_features_still_computed_when_yap_missing(self, ir):
        """Lexical, structural, discourse, and style features are still computed."""
        ir.missing_layers = ["yap"]
        result = extract_features(ir)
        assert result.lexicon.type_token_ratio is not None
        assert result.structure.sentence_length_variance is not None
        assert result.discourse.connective_ratio is not None
        assert result.style.sentence_length_trend is not None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_yap_missing_with_stanza_present_morph_computed(self, ir):
        """When only yap is missing, morphological features are still computed."""
        ir.missing_layers = ["yap"]
        result = extract_features(ir)
        assert result.morphology.verb_ratio is not None


# ---------------------------------------------------------------------------
# Property 4: Zero-token safety
# Feature: feature-extraction-enhancements, Property 4
# ---------------------------------------------------------------------------

class TestProperty4ZeroTokenSafety:
    """**Validates: Requirements 9.1, 9.2**"""

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_empty_sentences_list_returns_zero(self, data):
        """IR with empty sentences list returns 0.0 for all ratio/variance features."""
        ir = _make_ir(sentences=[], missing_layers=[])
        result = extract_features(ir)
        # Morphological
        assert result.morphology.verb_ratio == 0.0
        assert result.morphology.prefix_density == 0.0
        assert result.morphology.suffix_pronoun_ratio == 0.0
        assert result.morphology.morphological_ambiguity == 0.0
        assert result.morphology.agreement_error_rate == 0.0
        assert result.morphology.binyan_entropy == 0.0
        assert result.morphology.construct_ratio == 0.0
        assert result.morphology.binyan_distribution == {}
        # Syntactic
        assert result.syntax.avg_sentence_length == 0.0
        assert result.syntax.avg_tree_depth == 0.0
        assert result.syntax.max_tree_depth == 0.0
        assert result.syntax.avg_dependency_distance == 0.0
        assert result.syntax.clauses_per_sentence == 0.0
        assert result.syntax.subordinate_clause_ratio == 0.0
        assert result.syntax.right_branching_ratio == 0.0
        assert result.syntax.dependency_distance_variance == 0.0
        # Lexical
        assert result.lexicon.type_token_ratio == 0.0
        assert result.lexicon.hapax_ratio == 0.0
        assert result.lexicon.avg_token_length == 0.0
        assert result.lexicon.lemma_diversity == 0.0
        assert result.lexicon.content_word_ratio == 0.0
        # Structural
        assert result.structure.sentence_length_variance == 0.0
        assert result.structure.long_sentence_ratio == 0.0
        assert result.structure.punctuation_ratio == 0.0
        assert result.structure.short_sentence_ratio == 0.0
        assert result.structure.missing_terminal_punctuation_ratio == 0.0
        # Discourse
        assert result.discourse.connective_ratio == 0.0
        assert result.discourse.sentence_overlap == 0.0
        assert result.discourse.pronoun_to_noun_ratio == 0.0
        # Style
        assert result.style.sentence_length_trend == 0.0
        assert result.style.pos_distribution_variance == 0.0

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_sentences=st.integers(min_value=1, max_value=5),
    )
    def test_sentences_with_zero_tokens_returns_zero(self, n_sentences):
        """IR with sentences that have zero tokens returns 0.0 for ratio/variance features."""
        empty_sentences = [IRSentence(tokens=[], dep_tree=None) for _ in range(n_sentences)]
        ir = _make_ir(sentences=empty_sentences, missing_layers=[])
        result = extract_features(ir)
        # Morphological
        assert result.morphology.verb_ratio == 0.0
        assert result.morphology.prefix_density == 0.0
        assert result.morphology.suffix_pronoun_ratio == 0.0
        assert result.morphology.morphological_ambiguity == 0.0
        assert result.morphology.agreement_error_rate == 0.0
        assert result.morphology.binyan_entropy == 0.0
        assert result.morphology.construct_ratio == 0.0
        # Lexical
        assert result.lexicon.type_token_ratio == 0.0
        assert result.lexicon.hapax_ratio == 0.0
        assert result.lexicon.avg_token_length == 0.0
        assert result.lexicon.lemma_diversity == 0.0
        assert result.lexicon.content_word_ratio == 0.0
        # Structural
        assert result.structure.punctuation_ratio == 0.0
        # Discourse
        assert result.discourse.pronoun_to_noun_ratio == 0.0


# ---------------------------------------------------------------------------
# Property 5: No division by zero
# Feature: feature-extraction-enhancements, Property 5
# ---------------------------------------------------------------------------

class TestProperty5NoDivisionByZero:
    """**Validates: Requirement 9.4**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_no_division_by_zero_nonempty_ir(self, ir):
        """No ZeroDivisionError for any non-empty IR."""
        extract_features(ir)  # should not raise

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_no_division_by_zero_empty_ir(self, data):
        """No ZeroDivisionError for empty IR."""
        ir = _make_ir(sentences=[], missing_layers=[])
        extract_features(ir)  # should not raise

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_no_division_by_zero_with_missing_layers(self, ir, missing):
        """No ZeroDivisionError for any IR with various missing_layers combinations."""
        ir.missing_layers = missing
        extract_features(ir)  # should not raise

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(data=st.data())
    def test_no_division_by_zero_single_sentence(self, data):
        """No ZeroDivisionError for single-sentence IR."""
        sent = data.draw(st_ir_sentence_with_dep_tree())
        ir = _make_ir(sentences=[sent], missing_layers=[])
        extract_features(ir)  # should not raise

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_sentences=st.integers(min_value=1, max_value=3),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_no_division_by_zero_empty_token_sentences(self, n_sentences, missing):
        """No ZeroDivisionError for IR with empty-token sentences and missing layers."""
        empty_sentences = [IRSentence(tokens=[], dep_tree=None) for _ in range(n_sentences)]
        ir = _make_ir(sentences=empty_sentences, missing_layers=missing)
        extract_features(ir)  # should not raise


# ---------------------------------------------------------------------------
# Property 6: Ratio bounds
# Feature: feature-extraction-enhancements, Property 6
# ---------------------------------------------------------------------------

class TestProperty6RatioBounds:
    """**Validates: Requirements 10.2, 10.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_bounded_ratios_in_unit_interval(self, ir):
        """Bounded ratio features are in [0.0, 1.0] when not None."""
        ir.missing_layers = []
        result = extract_features(ir)

        bounded_values = [
            result.structure.punctuation_ratio,
            result.structure.short_sentence_ratio,
            result.structure.missing_terminal_punctuation_ratio,
            result.syntax.subordinate_clause_ratio,
            result.syntax.right_branching_ratio,
            result.lexicon.content_word_ratio,
            result.morphology.agreement_error_rate,
            result.discourse.sentence_overlap,
        ]
        for val in bounded_values:
            if val is not None:
                assert 0.0 <= val <= 1.0, f"Expected [0,1] but got {val}"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_rare_word_ratio_bounded_when_present(self, ir):
        """rare_word_ratio is in [0.0, 1.0] when freq_dict is provided."""
        ir.missing_layers = []
        freq_dict = {"common": 100}
        result = extract_features(ir, freq_dict=freq_dict)
        if result.lexicon.rare_word_ratio is not None:
            assert 0.0 <= result.lexicon.rare_word_ratio <= 1.0

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_unbounded_ratios_non_negative(self, ir):
        """connective_ratio and pronoun_to_noun_ratio are >= 0.0 (may exceed 1.0)."""
        ir.missing_layers = []
        result = extract_features(ir)
        assert result.discourse.connective_ratio >= 0.0
        assert result.discourse.pronoun_to_noun_ratio >= 0.0


# ---------------------------------------------------------------------------
# Property 1: Return type invariant
# Feature: feature-extraction-enhancements, Property 1
# ---------------------------------------------------------------------------

class TestProperty1ReturnTypeInvariant:
    """**Validates: Requirements 10.1**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_all_numeric_fields_are_float_or_none(self, ir, missing):
        """Every numeric feature field is a real number (int|float) or None (except binyan_distribution which is dict or None).

        Note: Python's statistics.mean() can return int when given a list of ints,
        so we accept both int and float as valid numeric types.
        """
        ir.missing_layers = missing
        result = extract_features(ir)

        _numeric = (int, float)

        # Morphology
        for val in [
            result.morphology.verb_ratio,
            result.morphology.prefix_density,
            result.morphology.suffix_pronoun_ratio,
            result.morphology.morphological_ambiguity,
            result.morphology.agreement_error_rate,
            result.morphology.binyan_entropy,
            result.morphology.construct_ratio,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"

        # binyan_distribution is dict or None
        assert result.morphology.binyan_distribution is None or isinstance(result.morphology.binyan_distribution, dict)

        # Syntax
        for val in [
            result.syntax.avg_sentence_length,
            result.syntax.avg_tree_depth,
            result.syntax.max_tree_depth,
            result.syntax.avg_dependency_distance,
            result.syntax.clauses_per_sentence,
            result.syntax.subordinate_clause_ratio,
            result.syntax.right_branching_ratio,
            result.syntax.dependency_distance_variance,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"

        # Lexicon
        for val in [
            result.lexicon.type_token_ratio,
            result.lexicon.hapax_ratio,
            result.lexicon.avg_token_length,
            result.lexicon.lemma_diversity,
            result.lexicon.rare_word_ratio,
            result.lexicon.content_word_ratio,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"

        # Structure
        for val in [
            result.structure.sentence_length_variance,
            result.structure.long_sentence_ratio,
            result.structure.punctuation_ratio,
            result.structure.short_sentence_ratio,
            result.structure.missing_terminal_punctuation_ratio,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"

        # Discourse
        for val in [
            result.discourse.connective_ratio,
            result.discourse.sentence_overlap,
            result.discourse.pronoun_to_noun_ratio,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"

        # Style
        for val in [
            result.style.sentence_length_trend,
            result.style.pos_distribution_variance,
        ]:
            assert val is None or isinstance(val, _numeric), f"Expected numeric|None, got {type(val)}: {val}"


# ---------------------------------------------------------------------------
# Property 9: Non-negativity of variance and entropy
# Feature: feature-extraction-enhancements, Property 9
# ---------------------------------------------------------------------------

class TestProperty9NonNegativity:
    """**Validates: Requirement 10.4**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        ir=st_ir_nonempty(),
        missing=st.lists(st.sampled_from(["stanza", "yap"]), min_size=0, max_size=2, unique=True),
    )
    def test_variance_and_entropy_non_negative(self, ir, missing):
        """dependency_distance_variance, pos_distribution_variance, sentence_length_variance, binyan_entropy >= 0.0 when not None."""
        ir.missing_layers = missing
        result = extract_features(ir)

        if result.syntax.dependency_distance_variance is not None:
            assert result.syntax.dependency_distance_variance >= 0.0
        if result.style.pos_distribution_variance is not None:
            assert result.style.pos_distribution_variance >= 0.0
        if result.structure.sentence_length_variance is not None:
            assert result.structure.sentence_length_variance >= 0.0
        if result.morphology.binyan_entropy is not None:
            assert result.morphology.binyan_entropy >= 0.0


# ---------------------------------------------------------------------------
# Property 12: Idempotency
# Feature: feature-extraction-enhancements, Property 12
# ---------------------------------------------------------------------------

class TestProperty12Idempotency:
    """**Validates: Requirement 11.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(ir=st_ir_nonempty())
    def test_extract_features_idempotent(self, ir):
        """Calling extract_features() twice with same IR produces identical results."""
        result1 = extract_features(ir)
        result2 = extract_features(ir)

        # Morphology
        assert result1.morphology.verb_ratio == result2.morphology.verb_ratio
        assert result1.morphology.binyan_distribution == result2.morphology.binyan_distribution
        assert result1.morphology.prefix_density == result2.morphology.prefix_density
        assert result1.morphology.suffix_pronoun_ratio == result2.morphology.suffix_pronoun_ratio
        assert result1.morphology.morphological_ambiguity == result2.morphology.morphological_ambiguity
        assert result1.morphology.agreement_error_rate == result2.morphology.agreement_error_rate
        assert result1.morphology.binyan_entropy == result2.morphology.binyan_entropy
        assert result1.morphology.construct_ratio == result2.morphology.construct_ratio

        # Syntax
        assert result1.syntax.avg_sentence_length == result2.syntax.avg_sentence_length
        assert result1.syntax.avg_tree_depth == result2.syntax.avg_tree_depth
        assert result1.syntax.max_tree_depth == result2.syntax.max_tree_depth
        assert result1.syntax.avg_dependency_distance == result2.syntax.avg_dependency_distance
        assert result1.syntax.clauses_per_sentence == result2.syntax.clauses_per_sentence
        assert result1.syntax.subordinate_clause_ratio == result2.syntax.subordinate_clause_ratio
        assert result1.syntax.right_branching_ratio == result2.syntax.right_branching_ratio
        assert result1.syntax.dependency_distance_variance == result2.syntax.dependency_distance_variance

        # Lexicon
        assert result1.lexicon.type_token_ratio == result2.lexicon.type_token_ratio
        assert result1.lexicon.hapax_ratio == result2.lexicon.hapax_ratio
        assert result1.lexicon.avg_token_length == result2.lexicon.avg_token_length
        assert result1.lexicon.lemma_diversity == result2.lexicon.lemma_diversity
        assert result1.lexicon.rare_word_ratio == result2.lexicon.rare_word_ratio
        assert result1.lexicon.content_word_ratio == result2.lexicon.content_word_ratio

        # Structure
        assert result1.structure.sentence_length_variance == result2.structure.sentence_length_variance
        assert result1.structure.long_sentence_ratio == result2.structure.long_sentence_ratio
        assert result1.structure.punctuation_ratio == result2.structure.punctuation_ratio
        assert result1.structure.short_sentence_ratio == result2.structure.short_sentence_ratio
        assert result1.structure.missing_terminal_punctuation_ratio == result2.structure.missing_terminal_punctuation_ratio

        # Discourse
        assert result1.discourse.connective_ratio == result2.discourse.connective_ratio
        assert result1.discourse.sentence_overlap == result2.discourse.sentence_overlap
        assert result1.discourse.pronoun_to_noun_ratio == result2.discourse.pronoun_to_noun_ratio

        # Style
        assert result1.style.sentence_length_trend == result2.style.sentence_length_trend
        assert result1.style.pos_distribution_variance == result2.style.pos_distribution_variance


# ---------------------------------------------------------------------------
# Property 13: Single-sentence edge case
# Feature: feature-extraction-enhancements, Property 13
# ---------------------------------------------------------------------------

class TestProperty13SingleSentence:
    """**Validates: Requirement 9.3**"""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(sent=st_ir_sentence_with_dep_tree())
    def test_single_sentence_edge_case_features_zero(self, sent):
        """sentence_overlap, sentence_length_trend, sentence_length_variance, pos_distribution_variance all return 0.0 for single-sentence IR."""
        ir = _make_ir(sentences=[sent], missing_layers=[])
        result = extract_features(ir)
        assert result.discourse.sentence_overlap == 0.0
        assert result.style.sentence_length_trend == 0.0
        assert result.structure.sentence_length_variance == 0.0
        assert result.style.pos_distribution_variance == 0.0
