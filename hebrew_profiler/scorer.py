"""Difficulty and style scoring for the Hebrew Linguistic Profiling Engine.

Computes heuristic difficulty and style scores from extracted features
using weighted linear combinations with min-max normalization.
Also computes normalized feature values and composite scores for
fluency, cohesion, and complexity.
"""

from __future__ import annotations

from hebrew_profiler.models import (
    DifficultyWeights,
    Features,
    NormalizationRanges,
    Scores,
    StyleWeights,
)


def _norm(x: float, min_val: float, max_val: float) -> float:
    """Min-max normalize *x* into [0.0, 1.0], clamped.

    ``norm(x, min, max) = clamp((x - min) / (max - min), 0.0, 1.0)``

    If *min_val* == *max_val* the result is 0.0 (degenerate range).
    """
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (x - min_val) / (max_val - min_val)))


def compute_scores(
    features: Features,
    difficulty_weights: DifficultyWeights,
    style_weights: StyleWeights,
    normalization_ranges: NormalizationRanges,
) -> Scores:
    """Compute difficulty and style scores from extracted *features*.

    Parameters
    ----------
    features:
        The extracted linguistic features (some values may be ``None``).
    difficulty_weights:
        Weights ``w1..w4`` for the difficulty formula.
    style_weights:
        Weights ``a1..a3`` for the style formula.
    normalization_ranges:
        ``(min, max)`` tuples used by the min-max normalizer.

    Returns
    -------
    Scores
        ``difficulty`` and ``style`` fields (each may be ``None`` when
        all contributing features are absent).
    """

    # ------------------------------------------------------------------
    # Difficulty score
    # ------------------------------------------------------------------
    # Collect (weight, normalized_value) pairs for present features only.
    difficulty_pairs: list[tuple[float, float]] = []

    avg_sl = features.syntax.avg_sentence_length
    if avg_sl is not None:
        r = normalization_ranges.avg_sentence_length
        difficulty_pairs.append((difficulty_weights.w1, _norm(avg_sl, r[0], r[1])))

    avg_td = features.syntax.avg_tree_depth
    if avg_td is not None:
        r = normalization_ranges.avg_tree_depth
        difficulty_pairs.append((difficulty_weights.w2, _norm(avg_td, r[0], r[1])))

    hapax = features.lexicon.hapax_ratio
    if hapax is not None:
        r = normalization_ranges.hapax_ratio
        difficulty_pairs.append((difficulty_weights.w3, _norm(hapax, r[0], r[1])))

    morph_amb = features.morphology.morphological_ambiguity
    if morph_amb is not None:
        r = normalization_ranges.morphological_ambiguity
        difficulty_pairs.append((difficulty_weights.w4, _norm(morph_amb, r[0], r[1])))

    if difficulty_pairs:
        total_weight = sum(w for w, _ in difficulty_pairs)
        if total_weight > 0.0:
            difficulty = sum(
                (w / total_weight) * v for w, v in difficulty_pairs
            )
            difficulty = max(0.0, min(1.0, difficulty))
        else:
            difficulty = None
    else:
        difficulty = None

    # ------------------------------------------------------------------
    # Style score
    # ------------------------------------------------------------------
    # style = a1 × norm(suffix_pronoun_ratio)
    #       − a3 × norm(hapax_ratio)
    #       + a4 × norm(|sentence_length_trend|)
    #       − a5 × norm(pos_distribution_variance)
    #       + a6 × norm(pronoun_to_noun_ratio)
    # sentence_length_variance removed — belongs to fluency
    style_pairs: list[tuple[float, float]] = []

    pronoun_ratio = features.morphology.suffix_pronoun_ratio
    if pronoun_ratio is not None:
        r = normalization_ranges.suffix_pronoun_ratio
        style_pairs.append((style_weights.a1, _norm(pronoun_ratio, r[0], r[1])))

    hapax = features.lexicon.hapax_ratio
    if hapax is not None:
        r = normalization_ranges.hapax_ratio
        # Negative contribution: higher hapax ratio → lower style score
        style_pairs.append((-style_weights.a3, _norm(hapax, r[0], r[1])))

    slt = features.style.sentence_length_trend
    if slt is not None:
        r = normalization_ranges.sentence_length_trend
        # Use absolute value — any strong trend (up or down) is a style signal
        style_pairs.append((style_weights.a4, _norm(abs(slt), 0.0, r[1])))

    pdv = features.style.pos_distribution_variance
    if pdv is not None:
        r = normalization_ranges.pos_distribution_variance
        # Negative: higher POS variance = less consistent style
        style_pairs.append((-style_weights.a5, _norm(pdv, r[0], r[1])))

    pnr = features.discourse.pronoun_to_noun_ratio
    if pnr is not None:
        r = normalization_ranges.pronoun_to_noun_ratio
        style_pairs.append((style_weights.a6, _norm(pnr, r[0], r[1])))

    if style_pairs:
        style = sum(w * v for w, v in style_pairs)
    else:
        style = None

    return Scores(difficulty=difficulty, style=style)


def compute_normalized_features(
    features: Features,
    normalization_ranges: NormalizationRanges,
) -> dict[str, float | None]:
    """Normalize all features that have configured ranges to [0, 1].

    Features without a configured range but naturally bounded in [0, 1]
    are passed through as-is. Features that are None remain None.

    Returns a flat dict mapping feature names to normalized values.
    """
    nf: dict[str, float | None] = {}

    # --- Features with configured normalization ranges ---
    _range_map: list[tuple[str, float | None, tuple[float, float]]] = [
        ("avg_sentence_length", features.syntax.avg_sentence_length,
         normalization_ranges.avg_sentence_length),
        ("avg_tree_depth", features.syntax.avg_tree_depth,
         normalization_ranges.avg_tree_depth),
        ("hapax_ratio", features.lexicon.hapax_ratio,
         normalization_ranges.hapax_ratio),
        ("morphological_ambiguity", features.morphology.morphological_ambiguity,
         normalization_ranges.morphological_ambiguity),
        ("suffix_pronoun_ratio", features.morphology.suffix_pronoun_ratio,
         normalization_ranges.suffix_pronoun_ratio),
        ("sentence_length_variance", features.structure.sentence_length_variance,
         normalization_ranges.sentence_length_variance),
        ("sentence_length_trend", features.style.sentence_length_trend,
         normalization_ranges.sentence_length_trend),
        ("pos_distribution_variance", features.style.pos_distribution_variance,
         normalization_ranges.pos_distribution_variance),
        ("pronoun_to_noun_ratio", features.discourse.pronoun_to_noun_ratio,
         normalization_ranges.pronoun_to_noun_ratio),
        ("rare_word_ratio", features.lexicon.rare_word_ratio,
         normalization_ranges.rare_word_ratio),
        ("content_word_ratio", features.lexicon.content_word_ratio,
         normalization_ranges.content_word_ratio),
        ("connective_ratio", features.discourse.connective_ratio,
         normalization_ranges.connective_ratio),
        ("sentence_overlap", features.discourse.sentence_overlap,
         normalization_ranges.sentence_overlap),
        ("agreement_error_rate", features.morphology.agreement_error_rate,
         normalization_ranges.agreement_error_rate),
        ("dependency_distance_variance", features.syntax.dependency_distance_variance,
         normalization_ranges.dependency_distance_variance),
        ("clause_type_entropy", features.syntax.clause_type_entropy,
         normalization_ranges.clause_type_entropy),
    ]

    for name, val, (lo, hi) in _range_map:
        nf[name] = _norm(val, lo, hi) if val is not None else None

    # --- Features already in [0, 1] — pass through ---
    _passthrough: list[tuple[str, float | None]] = [
        ("verb_ratio", features.morphology.verb_ratio),
        ("prefix_density", features.morphology.prefix_density),
        ("binyan_entropy", features.morphology.binyan_entropy),
        ("construct_ratio", features.morphology.construct_ratio),
        ("avg_dependency_distance", features.syntax.avg_dependency_distance),
        ("clauses_per_sentence", features.syntax.clauses_per_sentence),
        ("subordinate_clause_ratio", features.syntax.subordinate_clause_ratio),
        ("right_branching_ratio", features.syntax.right_branching_ratio),
        ("type_token_ratio", features.lexicon.type_token_ratio),
        ("avg_token_length", features.lexicon.avg_token_length),
        ("lemma_diversity", features.lexicon.lemma_diversity),
        ("long_sentence_ratio", features.structure.long_sentence_ratio),
        ("punctuation_ratio", features.structure.punctuation_ratio),
        ("short_sentence_ratio", features.structure.short_sentence_ratio),
        ("missing_terminal_punctuation_ratio",
         features.structure.missing_terminal_punctuation_ratio),
    ]

    for name, val in _passthrough:
        nf[name] = val

    return nf


def _mean_of_present(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None if all are None."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def compute_composite_scores(
    normalized: dict[str, float | None],
) -> dict[str, float | None]:
    """Compute composite scores from normalized feature values.

    fluency:    Higher = more fluent writing. Based on absence of
                structural issues (inverted: fewer short sentences,
                less missing punctuation, moderate punctuation use).
    cohesion:   Higher = more cohesive text. Based on discourse
                connectives and sentence overlap.
    complexity: Higher = more syntactically/morphologically complex.
                Based on subordination, dependency distance, and
                agreement error rate.
    """
    # Fluency: structural regularity and consistency
    # Uses punctuation_ratio (moderate is good), sentence_length_variance
    # (lower = more consistent), and pos_distribution_variance (lower = more consistent)
    fluency_vals: list[float | None] = []
    # Punctuation ratio: moderate is good, so use raw value as positive signal
    fluency_vals.append(normalized.get("punctuation_ratio"))
    # Lower variance = more fluent, so invert
    slv = normalized.get("sentence_length_variance")
    fluency_vals.append(1.0 - slv if slv is not None else None)
    pdv = normalized.get("pos_distribution_variance")
    fluency_vals.append(1.0 - pdv if pdv is not None else None)
    fluency = _mean_of_present(fluency_vals)

    # Cohesion: weighted discourse signal
    # connective_ratio was r=0.91 with cohesion — too dominant.
    # Rebalanced: connectives (0.4) + sentence overlap (0.3) + referential clarity (0.3)
    # Inverted pronoun_to_noun_ratio: pronoun overload → poor referential clarity
    cohesion_vals: list[tuple[float, float | None]] = [
        (0.4, normalized.get("connective_ratio")),
        (0.3, normalized.get("sentence_overlap")),
    ]
    pnr = normalized.get("pronoun_to_noun_ratio")
    cohesion_vals.append((0.3, 1.0 - pnr if pnr is not None else None))

    present_cohesion = [(w, v) for w, v in cohesion_vals if v is not None]
    if present_cohesion:
        total_w = sum(w for w, _ in present_cohesion)
        cohesion = sum((w / total_w) * v for w, v in present_cohesion)
    else:
        cohesion = None

    # Complexity: structural diversity (not structural load)
    # Removed: subordinate_clause_ratio (broken — always 1.0 in YAP)
    # Removed: dependency_distance_variance (overlaps with difficulty)
    # Kept: binyan_entropy (Hebrew-specific verb diversity)
    # Kept: agreement_error_rate (complex constructions → more mismatches)
    # Added: pos_distribution_variance (syntactic diversity across sentences)
    # Added: clause_type_entropy (diversity of dependency relation types — NEW)
    # construct_ratio kept but downweighted (noisy heuristic)
    complexity_vals: list[float | None] = [
        normalized.get("binyan_entropy"),
        normalized.get("agreement_error_rate"),
        normalized.get("pos_distribution_variance"),
        normalized.get("clause_type_entropy"),
    ]
    # construct_ratio downweighted — noisy noun-noun heuristic
    cr = normalized.get("construct_ratio")
    if cr is not None:
        complexity_vals.append(cr * 0.5)
    complexity = _mean_of_present(complexity_vals)

    return {
        "fluency": max(0.0, min(1.0, fluency)) if fluency is not None else None,
        "cohesion": max(0.0, min(1.0, cohesion)) if cohesion is not None else None,
        "complexity": max(0.0, min(1.0, complexity)) if complexity is not None else None,
    }
