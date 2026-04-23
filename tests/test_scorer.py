"""Unit tests for hebrew_profiler.scorer."""

from __future__ import annotations

import pytest

from hebrew_profiler.models import (
    DifficultyWeights,
    Features,
    LexicalFeatures,
    MorphFeatures,
    NormalizationRanges,
    Scores,
    StructuralFeatures,
    StyleWeights,
    SyntaxFeatures,
)
from hebrew_profiler.scorer import _norm, compute_scores


# ── _norm helper ─────────────────────────────────────────────────────

class TestNorm:
    def test_midpoint(self):
        assert _norm(5.0, 0.0, 10.0) == 0.5

    def test_at_min(self):
        assert _norm(0.0, 0.0, 10.0) == 0.0

    def test_at_max(self):
        assert _norm(10.0, 0.0, 10.0) == 1.0

    def test_below_min_clamped(self):
        assert _norm(-5.0, 0.0, 10.0) == 0.0

    def test_above_max_clamped(self):
        assert _norm(15.0, 0.0, 10.0) == 1.0

    def test_degenerate_range(self):
        assert _norm(5.0, 5.0, 5.0) == 0.0


# ── helpers ──────────────────────────────────────────────────────────

def _full_features(
    avg_sentence_length=15.0,
    avg_tree_depth=5.0,
    hapax_ratio=0.4,
    morphological_ambiguity=3.0,
    suffix_pronoun_ratio=0.1,
    sentence_length_variance=10.0,
) -> Features:
    from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
    return Features(
        morphology=MorphFeatures(
            verb_ratio=0.2,
            binyan_distribution={"paal": 1},
            prefix_density=0.1,
            suffix_pronoun_ratio=suffix_pronoun_ratio,
            morphological_ambiguity=morphological_ambiguity,
            agreement_error_rate=0.0,
            binyan_entropy=0.0,
            construct_ratio=0.0,
        ),
        syntax=SyntaxFeatures(
            avg_sentence_length=avg_sentence_length,
            avg_tree_depth=avg_tree_depth,
            max_tree_depth=8.0,
            avg_dependency_distance=2.0,
            clauses_per_sentence=1.0,
            subordinate_clause_ratio=0.0,
            right_branching_ratio=0.0,
            dependency_distance_variance=0.0,
            clause_type_entropy=0.0,
        ),
        lexicon=LexicalFeatures(
            type_token_ratio=0.7,
            hapax_ratio=hapax_ratio,
            avg_token_length=4.0,
            lemma_diversity=0.6,
            rare_word_ratio=None,
            content_word_ratio=0.0,
        ),
        structure=StructuralFeatures(
            sentence_length_variance=sentence_length_variance,
            long_sentence_ratio=0.2,
            punctuation_ratio=0.0,
            short_sentence_ratio=0.0,
            missing_terminal_punctuation_ratio=0.0,
        ),
        discourse=DiscourseFeatures(
            connective_ratio=0.0,
            sentence_overlap=0.0,
            pronoun_to_noun_ratio=0.0,
        ),
        style=StyleFeatures(
            sentence_length_trend=0.0,
            pos_distribution_variance=0.0,
        ),
    )


DEFAULT_DW = DifficultyWeights()
DEFAULT_SW = StyleWeights()
DEFAULT_NR = NormalizationRanges()


# ── compute_scores – difficulty ──────────────────────────────────────

class TestDifficultyScore:
    def test_all_features_present(self):
        features = _full_features()
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        assert scores.difficulty is not None
        assert 0.0 <= scores.difficulty <= 1.0

    def test_difficulty_manual_calculation(self):
        """Verify the formula with known inputs."""
        features = _full_features(
            avg_sentence_length=25.0,  # norm → (25-10)/(40-10) = 0.5
            avg_tree_depth=9.5,        # norm → (9.5-4)/(15-4) = 0.5
            hapax_ratio=0.35,          # norm → (0.35-0.15)/(0.55-0.15) = 0.5
            morphological_ambiguity=7.0,  # norm → (7.0-4)/(10-4) = 0.5
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        # All normalized to 0.5, weights sum to 1.0 → difficulty = 0.5
        assert scores.difficulty == pytest.approx(0.5)

    def test_all_absent_returns_none(self):
        from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
        features = Features(
            morphology=MorphFeatures(
                verb_ratio=0.2,
                binyan_distribution={},
                prefix_density=0.1,
                suffix_pronoun_ratio=None,
                morphological_ambiguity=None,
                agreement_error_rate=None,
                binyan_entropy=None,
                construct_ratio=None,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=None,
                avg_tree_depth=None,
                max_tree_depth=None,
                avg_dependency_distance=None,
                clauses_per_sentence=None,
                subordinate_clause_ratio=None,
                right_branching_ratio=None,
                dependency_distance_variance=None,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=None,
                hapax_ratio=None,
                avg_token_length=None,
                lemma_diversity=None,
                rare_word_ratio=None,
                content_word_ratio=None,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=None,
                long_sentence_ratio=None,
                punctuation_ratio=None,
                short_sentence_ratio=None,
                missing_terminal_punctuation_ratio=None,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        assert scores.difficulty is None
        # Style is still computed from discourse/style features (sentence_length_trend,
        # pos_distribution_variance, pronoun_to_noun_ratio) even when morph/lexicon are absent
        assert scores.style is not None

    def test_partial_features_renormalize(self):
        """When some features are absent, remaining weights are re-normalized."""
        from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
        features = Features(
            morphology=MorphFeatures(
                verb_ratio=0.2,
                binyan_distribution={},
                prefix_density=0.1,
                suffix_pronoun_ratio=0.1,
                morphological_ambiguity=None,  # absent → w4 excluded
                agreement_error_rate=None,
                binyan_entropy=None,
                construct_ratio=None,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=25.0,  # norm → 0.5
                avg_tree_depth=9.5,        # norm → 0.5
                max_tree_depth=8.0,
                avg_dependency_distance=2.0,
                clauses_per_sentence=1.0,
                subordinate_clause_ratio=0.0,
                right_branching_ratio=0.0,
                dependency_distance_variance=0.0,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=0.7,
                hapax_ratio=0.35,          # norm → 0.5
                avg_token_length=4.0,
                lemma_diversity=0.6,
                rare_word_ratio=None,
                content_word_ratio=0.0,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=10.0,
                long_sentence_ratio=0.2,
                punctuation_ratio=0.0,
                short_sentence_ratio=0.0,
                missing_terminal_punctuation_ratio=0.0,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        # w1=0.3, w2=0.25, w3=0.25 → total=0.8
        # re-normalized: each * (1/0.8), all norms = 0.5 → difficulty = 0.5
        assert scores.difficulty == pytest.approx(0.5)


# ── compute_scores – style ───────────────────────────────────────────

class TestStyleScore:
    def test_all_features_present(self):
        features = _full_features(
            suffix_pronoun_ratio=0.275,   # norm → (0.275-0.05)/(0.50-0.05) = 0.5
            hapax_ratio=0.35,             # norm → (0.35-0.15)/(0.55-0.15) = 0.5
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        # style = 0.25*0.5 - 0.25*0.5 + 0.20*0 - 0.15*0 + 0.15*0 = 0.0
        assert scores.style == pytest.approx(0.0)

    def test_absent_pronoun_ratio(self):
        from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
        features = Features(
            morphology=MorphFeatures(
                verb_ratio=0.2,
                binyan_distribution={},
                prefix_density=0.1,
                suffix_pronoun_ratio=None,
                morphological_ambiguity=7.0,
                agreement_error_rate=0.0,
                binyan_entropy=0.0,
                construct_ratio=0.0,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=15.0,
                avg_tree_depth=5.0,
                max_tree_depth=8.0,
                avg_dependency_distance=2.0,
                clauses_per_sentence=1.0,
                subordinate_clause_ratio=0.0,
                right_branching_ratio=0.0,
                dependency_distance_variance=0.0,
                clause_type_entropy=0.0,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=0.7,
                hapax_ratio=0.35,          # norm → 0.5
                avg_token_length=4.0,
                lemma_diversity=0.6,
                rare_word_ratio=None,
                content_word_ratio=0.0,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=200.0,
                long_sentence_ratio=0.2,
                punctuation_ratio=0.0,
                short_sentence_ratio=0.0,
                missing_terminal_punctuation_ratio=0.0,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        # suffix_pronoun_ratio absent: style = -0.25*0.5 + 0 = -0.125
        assert scores.style == pytest.approx(-0.125)

    def test_all_style_features_absent(self):
        from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
        features = Features(
            morphology=MorphFeatures(
                verb_ratio=0.2,
                binyan_distribution={},
                prefix_density=0.1,
                suffix_pronoun_ratio=None,
                morphological_ambiguity=3.0,
                agreement_error_rate=0.0,
                binyan_entropy=0.0,
                construct_ratio=0.0,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=15.0,
                avg_tree_depth=5.0,
                max_tree_depth=8.0,
                avg_dependency_distance=2.0,
                clauses_per_sentence=1.0,
                subordinate_clause_ratio=0.0,
                right_branching_ratio=0.0,
                dependency_distance_variance=0.0,
                clause_type_entropy=0.0,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=0.7,
                hapax_ratio=None,
                avg_token_length=4.0,
                lemma_diversity=0.6,
                rare_word_ratio=None,
                content_word_ratio=0.0,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=None,
                long_sentence_ratio=0.2,
                punctuation_ratio=0.0,
                short_sentence_ratio=0.0,
                missing_terminal_punctuation_ratio=0.0,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        )
        scores = compute_scores(features, DEFAULT_DW, DEFAULT_SW, DEFAULT_NR)
        # All style inputs absent or 0: a4*0 - a5*0 + a6*0 = 0.0
        assert scores.style == pytest.approx(0.0)


# ── Property-based tests (Hypothesis) ────────────────────────────────

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from hebrew_profiler.scorer import _norm, compute_scores


# ── Strategies ───────────────────────────────────────────────────────

def _finite_floats(**kwargs):
    """Finite floats (no NaN / inf)."""
    return st.floats(allow_nan=False, allow_infinity=False, **kwargs)


@st.composite
def min_max_and_value(draw):
    """Draw (x, min_val, max_val) where min_val < max_val."""
    a = draw(_finite_floats(min_value=-1e6, max_value=1e6))
    b = draw(_finite_floats(min_value=-1e6, max_value=1e6))
    assume(a != b)
    min_val, max_val = sorted([a, b])
    x = draw(_finite_floats(min_value=-1e7, max_value=1e7))
    return x, min_val, max_val


@st.composite
def difficulty_weights_summing_to_one(draw):
    """Generate DifficultyWeights whose w1..w4 sum to 1.0."""
    raw = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(4)]
    total = sum(raw)
    normed = [r / total for r in raw]
    return DifficultyWeights(w1=normed[0], w2=normed[1], w3=normed[2], w4=normed[3])


@st.composite
def style_weights_strategy(draw):
    """Generate arbitrary positive StyleWeights."""
    a1 = draw(st.floats(min_value=0.0, max_value=10.0))
    a3 = draw(st.floats(min_value=0.0, max_value=10.0))
    a4 = draw(st.floats(min_value=0.0, max_value=10.0))
    a5 = draw(st.floats(min_value=0.0, max_value=10.0))
    a6 = draw(st.floats(min_value=0.0, max_value=10.0))
    return StyleWeights(a1=a1, a3=a3, a4=a4, a5=a5, a6=a6)


@st.composite
def full_features_strategy(draw):
    """Generate Features with ALL values present (no None)."""
    from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
    return Features(
        morphology=MorphFeatures(
            verb_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
            binyan_distribution={"paal": 1},
            prefix_density=draw(st.floats(min_value=0.0, max_value=1.0)),
            suffix_pronoun_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
            morphological_ambiguity=draw(st.floats(min_value=1.0, max_value=8.0)),
            agreement_error_rate=0.0,
            binyan_entropy=0.0,
            construct_ratio=0.0,
        ),
        syntax=SyntaxFeatures(
            avg_sentence_length=draw(st.floats(min_value=1.0, max_value=100.0)),
            avg_tree_depth=draw(st.floats(min_value=1.0, max_value=20.0)),
            max_tree_depth=draw(st.floats(min_value=1.0, max_value=20.0)),
            avg_dependency_distance=draw(st.floats(min_value=0.0, max_value=20.0)),
            clauses_per_sentence=draw(st.floats(min_value=0.0, max_value=10.0)),
            subordinate_clause_ratio=0.0,
            right_branching_ratio=0.0,
            dependency_distance_variance=0.0,
            clause_type_entropy=0.0,
        ),
        lexicon=LexicalFeatures(
            type_token_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
            hapax_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
            avg_token_length=draw(st.floats(min_value=1.0, max_value=20.0)),
            lemma_diversity=draw(st.floats(min_value=0.0, max_value=1.0)),
            rare_word_ratio=None,
            content_word_ratio=0.0,
        ),
        structure=StructuralFeatures(
            sentence_length_variance=draw(st.floats(min_value=0.0, max_value=500.0)),
            long_sentence_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
            punctuation_ratio=0.0,
            short_sentence_ratio=0.0,
            missing_terminal_punctuation_ratio=0.0,
        ),
        discourse=DiscourseFeatures(
            connective_ratio=0.0,
            sentence_overlap=0.0,
            pronoun_to_noun_ratio=0.0,
        ),
        style=StyleFeatures(
            sentence_length_trend=0.0,
            pos_distribution_variance=0.0,
        ),
    )


@st.composite
def partial_features_strategy(draw):
    """Generate Features where at least one difficulty field is present and at
    least one is None.  Similarly for style fields."""
    from hebrew_profiler.models import DiscourseFeatures, StyleFeatures
    # Difficulty-relevant: avg_sentence_length, avg_tree_depth, hapax_ratio, morphological_ambiguity
    diff_present = draw(st.lists(st.booleans(), min_size=4, max_size=4))
    # Ensure at least one True and at least one False
    assume(any(diff_present) and not all(diff_present))

    avg_sl = draw(st.floats(min_value=1.0, max_value=100.0)) if diff_present[0] else None
    avg_td = draw(st.floats(min_value=1.0, max_value=20.0)) if diff_present[1] else None
    hapax = draw(st.floats(min_value=0.0, max_value=1.0)) if diff_present[2] else None
    morph_amb = draw(st.floats(min_value=1.0, max_value=8.0)) if diff_present[3] else None

    # Style-relevant: suffix_pronoun_ratio, sentence_length_variance, hapax_ratio (reused)
    style_present = draw(st.lists(st.booleans(), min_size=3, max_size=3))
    assume(any(style_present) and not all(style_present))

    spr = draw(st.floats(min_value=0.0, max_value=1.0)) if style_present[0] else None
    slv = draw(st.floats(min_value=0.0, max_value=500.0)) if style_present[1] else None
    # hapax_ratio is shared between difficulty and style; use the difficulty value if present
    # for style, hapax_ratio drives rare_word_ratio
    style_hapax = hapax if style_present[2] else None
    # If difficulty wants hapax present but style doesn't (or vice-versa), we need to reconcile.
    # The scorer reads hapax from features.lexicon.hapax_ratio for both.
    # So we pick: present if either difficulty or style wants it.
    final_hapax = hapax if diff_present[2] else (draw(st.floats(min_value=0.0, max_value=1.0)) if style_present[2] else None)

    return Features(
        morphology=MorphFeatures(
            verb_ratio=0.2,
            binyan_distribution={"paal": 1},
            prefix_density=0.1,
            suffix_pronoun_ratio=spr,
            morphological_ambiguity=morph_amb,
            agreement_error_rate=0.0,
            binyan_entropy=0.0,
            construct_ratio=0.0,
        ),
        syntax=SyntaxFeatures(
            avg_sentence_length=avg_sl,
            avg_tree_depth=avg_td,
            max_tree_depth=8.0,
            avg_dependency_distance=2.0,
            clauses_per_sentence=1.0,
            subordinate_clause_ratio=0.0,
            right_branching_ratio=0.0,
            dependency_distance_variance=0.0,
            clause_type_entropy=0.0,
        ),
        lexicon=LexicalFeatures(
            type_token_ratio=0.7,
            hapax_ratio=final_hapax,
            avg_token_length=4.0,
            lemma_diversity=0.6,
            rare_word_ratio=None,
            content_word_ratio=0.0,
        ),
        structure=StructuralFeatures(
            sentence_length_variance=slv,
            long_sentence_ratio=0.2,
            punctuation_ratio=0.0,
            short_sentence_ratio=0.0,
            missing_terminal_punctuation_ratio=0.0,
        ),
        discourse=DiscourseFeatures(
            connective_ratio=0.0,
            sentence_overlap=0.0,
            pronoun_to_noun_ratio=0.0,
        ),
        style=StyleFeatures(
            sentence_length_trend=0.0,
            pos_distribution_variance=0.0,
        ),
    )


# ── Property 16: Min-max normalization range invariant ───────────────
# Feature: hebrew-linguistic-profiling-engine, Property 16
# **Validates: Requirements 11.2**

class TestProperty16NormRange:
    @given(data=min_max_and_value())
    @settings(max_examples=200)
    def test_norm_result_in_unit_interval(self, data):
        """For any numeric value and any valid (min, max) range where min < max,
        _norm SHALL produce a result in [0.0, 1.0]."""
        x, min_val, max_val = data
        result = _norm(x, min_val, max_val)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ── Property 17: Difficulty score formula and range ──────────────────
# Feature: hebrew-linguistic-profiling-engine, Property 17
# **Validates: Requirements 11.1, 11.3**

class TestProperty17DifficultyRange:
    @given(
        features=full_features_strategy(),
        dw=difficulty_weights_summing_to_one(),
    )
    @settings(max_examples=200)
    def test_difficulty_in_unit_interval(self, features, dw):
        """For any set of extracted features and valid difficulty weights
        (summing to 1.0), difficulty_score SHALL be in [0.0, 1.0]."""
        nr = NormalizationRanges()
        sw = StyleWeights()
        scores = compute_scores(features, dw, sw, nr)
        assert scores.difficulty is not None
        assert 0.0 <= scores.difficulty <= 1.0


# ── Property 18: Style score formula ─────────────────────────────────
# Feature: hebrew-linguistic-profiling-engine, Property 18
# **Validates: Requirements 12.1**

class TestProperty18StyleFormula:
    @given(
        features=full_features_strategy(),
        sw=style_weights_strategy(),
    )
    @settings(max_examples=200)
    def test_style_score_matches_formula(self, features, sw):
        """style_score SHALL equal weighted sum of 5 normalized inputs (no sentence_length_variance)."""
        dw = DifficultyWeights()
        nr = NormalizationRanges()
        scores = compute_scores(features, dw, sw, nr)

        spr_norm = _norm(
            features.morphology.suffix_pronoun_ratio,
            nr.suffix_pronoun_ratio[0],
            nr.suffix_pronoun_ratio[1],
        )
        hapax_norm = _norm(
            features.lexicon.hapax_ratio,
            nr.hapax_ratio[0],
            nr.hapax_ratio[1],
        )
        slt_norm = _norm(
            abs(features.style.sentence_length_trend),
            0.0,
            nr.sentence_length_trend[1],
        )
        pdv_norm = _norm(
            features.style.pos_distribution_variance,
            nr.pos_distribution_variance[0],
            nr.pos_distribution_variance[1],
        )
        pnr_norm = _norm(
            features.discourse.pronoun_to_noun_ratio,
            nr.pronoun_to_noun_ratio[0],
            nr.pronoun_to_noun_ratio[1],
        )
        expected = (
            sw.a1 * spr_norm
            - sw.a3 * hapax_norm
            + sw.a4 * slt_norm
            - sw.a5 * pdv_norm
            + sw.a6 * pnr_norm
        )
        assert scores.style == pytest.approx(expected)


# ── Property 19: Absent feature handling in scoring ──────────────────
# Feature: hebrew-linguistic-profiling-engine, Property 19
# **Validates: Requirements 11.5, 12.3**

class TestProperty19AbsentFeatures:
    @given(features=partial_features_strategy())
    @settings(max_examples=200)
    def test_absent_features_produce_valid_scores(self, features):
        """When one or more features are absent (None), the Scorer SHALL
        exclude them, re-normalize difficulty weights, and still produce
        valid numeric scores."""
        dw = DifficultyWeights()
        sw = StyleWeights()
        nr = NormalizationRanges()
        scores = compute_scores(features, dw, sw, nr)

        # --- Difficulty ---
        # At least one difficulty feature is present (by construction),
        # so difficulty must be a valid float in [0.0, 1.0].
        diff_present = [
            features.syntax.avg_sentence_length is not None,
            features.syntax.avg_tree_depth is not None,
            features.lexicon.hapax_ratio is not None,
            features.morphology.morphological_ambiguity is not None,
        ]
        if any(diff_present):
            assert scores.difficulty is not None
            assert 0.0 <= scores.difficulty <= 1.0

        # --- Style ---
        style_present = [
            features.morphology.suffix_pronoun_ratio is not None,
            features.structure.sentence_length_variance is not None,
            features.lexicon.hapax_ratio is not None,
        ]
        if any(style_present):
            assert scores.style is not None
            assert isinstance(scores.style, float)
        else:
            assert scores.style is None
