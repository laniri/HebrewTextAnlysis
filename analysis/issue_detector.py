"""Issue detector module for the probabilistic analysis layer.

Detects 13 linguistic issue types across 6 groups from raw features,
sentence metrics, and corpus-derived statistics.
"""

from __future__ import annotations

from typing import Dict, List

from analysis.issue_models import Issue
from analysis.normalization import soft_score
from analysis.sentence_metrics import SentenceMetrics
from analysis.statistics import FeatureStats

# Fallback stats when a key is missing from feature_stats
_FALLBACK_MEAN = 0.5
_FALLBACK_STD = 1.0


def _get_stats(feature_stats: Dict[str, FeatureStats], key: str) -> FeatureStats:
    """Return FeatureStats for key, or a fallback if missing."""
    if key in feature_stats:
        return feature_stats[key]
    return FeatureStats(
        mean=_FALLBACK_MEAN,
        std=_FALLBACK_STD,
        min=0.0,
        max=1.0,
        p10=0.0,
        p25=0.0,
        p50=0.5,
        p75=1.0,
        p90=1.0,
        valid_count=0,
        unstable=True,
        degenerate=False,
    )


def _feature_stability(stats_list: List[FeatureStats]) -> float:
    """Compute feature_stability from a list of FeatureStats.

    Returns 0.0 if any is degenerate, 0.5 if any is unstable, 1.0 otherwise.
    """
    for s in stats_list:
        if s.degenerate:
            return 0.0
    for s in stats_list:
        if s.unstable:
            return 0.5
    return 1.0


def _confidence(
    severity: float,
    feature_values: List[float | None],
    stats_list: List[FeatureStats],
) -> float:
    """Compute confidence = min(1.0, severity * feature_availability * feature_stability)."""
    total = len(feature_values)
    if total == 0:
        return 0.0
    non_none = sum(1 for v in feature_values if v is not None)
    availability = non_none / total
    stability = _feature_stability(stats_list)
    return min(1.0, severity * availability * stability)


# ---------------------------------------------------------------------------
# Task 6.1 — Morphology detectors
# ---------------------------------------------------------------------------

def _detect_morphology(
    raw_features: Dict[str, float | None],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # agreement_errors
    val = raw_features.get("agreement_error_rate")
    if val is not None:
        st = _get_stats(feature_stats, "agreement_error_rate")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="agreement_errors",
            group="morphology",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"agreement_error_rate": val},
        ))

    # morphological_ambiguity
    val = raw_features.get("morphological_ambiguity")
    if val is not None:
        st = _get_stats(feature_stats, "morphological_ambiguity")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="morphological_ambiguity",
            group="morphology",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"morphological_ambiguity": val},
        ))

    # low_morphological_diversity
    val = raw_features.get("binyan_entropy")
    if val is not None:
        st = _get_stats(feature_stats, "binyan_entropy")
        severity = 1.0 - soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="low_morphological_diversity",
            group="morphology",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"binyan_entropy": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.2 — Syntax detectors
# ---------------------------------------------------------------------------

def _detect_syntax(
    raw_features: Dict[str, float | None],
    sentence_metrics: List[SentenceMetrics],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # sentence_complexity — one issue per sentence
    st_len = _get_stats(feature_stats, "avg_sentence_length")
    st_depth = _get_stats(feature_stats, "avg_tree_depth")
    for sm in sentence_metrics:
        severity = (
            0.6 * soft_score(sm.token_count, st_len.mean, st_len.std)
            + 0.4 * soft_score(sm.tree_depth, st_depth.mean, st_depth.std)
        )
        conf = _confidence(severity, [float(sm.token_count), sm.tree_depth], [st_len, st_depth])
        issues.append(Issue(
            type="sentence_complexity",
            group="syntax",
            severity=severity,
            confidence=conf,
            span=(sm.index,),
            evidence={
                "token_count": float(sm.token_count),
                "tree_depth": sm.tree_depth,
            },
        ))

    # dependency_spread
    val = raw_features.get("dependency_distance_variance")
    if val is not None:
        st = _get_stats(feature_stats, "dependency_distance_variance")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="dependency_spread",
            group="syntax",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"dependency_distance_variance": val},
        ))

    # excessive_branching
    val = raw_features.get("right_branching_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "right_branching_ratio")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="excessive_branching",
            group="syntax",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"right_branching_ratio": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.3 — Lexicon detectors
# ---------------------------------------------------------------------------

def _detect_lexicon(
    raw_features: Dict[str, float | None],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # low_lexical_diversity — uses both lemma_diversity and type_token_ratio
    ld_val = raw_features.get("lemma_diversity")
    ttr_val = raw_features.get("type_token_ratio")
    if ld_val is not None or ttr_val is not None:
        st_ld = _get_stats(feature_stats, "lemma_diversity")
        st_ttr = _get_stats(feature_stats, "type_token_ratio")
        ld_score = (1.0 - soft_score(ld_val, st_ld.mean, st_ld.std)) if ld_val is not None else 0.5
        ttr_score = (1.0 - soft_score(ttr_val, st_ttr.mean, st_ttr.std)) if ttr_val is not None else 0.5
        severity = 0.6 * ld_score + 0.4 * ttr_score
        conf = _confidence(severity, [ld_val, ttr_val], [st_ld, st_ttr])
        evidence: dict = {}
        if ld_val is not None:
            evidence["lemma_diversity"] = ld_val
        if ttr_val is not None:
            evidence["type_token_ratio"] = ttr_val
        issues.append(Issue(
            type="low_lexical_diversity",
            group="lexicon",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence=evidence,
        ))

    # rare_word_overuse
    val = raw_features.get("rare_word_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "rare_word_ratio")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="rare_word_overuse",
            group="lexicon",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"rare_word_ratio": val},
        ))

    # low_content_density
    val = raw_features.get("content_word_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "content_word_ratio")
        severity = 1.0 - soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="low_content_density",
            group="lexicon",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"content_word_ratio": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.4 — Structure detectors
# ---------------------------------------------------------------------------

def _detect_structure(
    raw_features: Dict[str, float | None],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # sentence_length_variability
    val = raw_features.get("sentence_length_variance")
    if val is not None:
        st = _get_stats(feature_stats, "sentence_length_variance")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="sentence_length_variability",
            group="structure",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"sentence_length_variance": val},
        ))

    # punctuation_issues — uses both punctuation_ratio and missing_terminal_punctuation_ratio
    pr_val = raw_features.get("punctuation_ratio")
    mtp_val = raw_features.get("missing_terminal_punctuation_ratio")
    if pr_val is not None or mtp_val is not None:
        st_pr = _get_stats(feature_stats, "punctuation_ratio")
        st_mtp = _get_stats(feature_stats, "missing_terminal_punctuation_ratio")
        pr_score = (1.0 - soft_score(pr_val, st_pr.mean, st_pr.std)) if pr_val is not None else 0.5
        mtp_score = soft_score(mtp_val, st_mtp.mean, st_mtp.std) if mtp_val is not None else 0.5
        severity = 0.5 * pr_score + 0.5 * mtp_score
        conf = _confidence(severity, [pr_val, mtp_val], [st_pr, st_mtp])
        evidence_s: dict = {}
        if pr_val is not None:
            evidence_s["punctuation_ratio"] = pr_val
        if mtp_val is not None:
            evidence_s["missing_terminal_punctuation_ratio"] = mtp_val
        issues.append(Issue(
            type="punctuation_issues",
            group="structure",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence=evidence_s,
        ))

    # fragmentation
    val = raw_features.get("short_sentence_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "short_sentence_ratio")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="fragmentation",
            group="structure",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"short_sentence_ratio": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.5 — Discourse detectors
# ---------------------------------------------------------------------------

def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|, 0.0 if both empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _detect_discourse(
    raw_features: Dict[str, float | None],
    sentence_metrics: List[SentenceMetrics],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # weak_cohesion — one issue per adjacent pair
    # Use sentence_cosine_similarity stats when available (embedding path),
    # fall back to sentence_overlap stats (Jaccard path).
    use_embeddings = (
        len(sentence_metrics) >= 2
        and sentence_metrics[0].embedding is not None
        and "sentence_cosine_similarity" in feature_stats
    )
    if use_embeddings:
        st_cohesion = feature_stats["sentence_cosine_similarity"]
    else:
        st_cohesion = _get_stats(feature_stats, "sentence_overlap")

    for i in range(1, len(sentence_metrics)):
        prev = sentence_metrics[i - 1]
        curr = sentence_metrics[i]

        if use_embeddings and prev.embedding is not None and curr.embedding is not None:
            import numpy as np
            sim = float(np.dot(prev.embedding, curr.embedding))
            evidence_key = "cosine_similarity"
        else:
            sim = _jaccard(prev.lemma_set, curr.lemma_set)
            evidence_key = "jaccard"

        severity = 1.0 - soft_score(sim, st_cohesion.mean, st_cohesion.std)
        conf = _confidence(severity, [sim], [st_cohesion])
        issues.append(Issue(
            type="weak_cohesion",
            group="discourse",
            severity=severity,
            confidence=conf,
            span=(i - 1, i),
            evidence={evidence_key: sim},
        ))

    # missing_connectives
    val = raw_features.get("connective_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "connective_ratio")
        severity = 1.0 - soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="missing_connectives",
            group="discourse",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"connective_ratio": val},
        ))

    # pronoun_ambiguity
    val = raw_features.get("pronoun_to_noun_ratio")
    if val is not None:
        st = _get_stats(feature_stats, "pronoun_to_noun_ratio")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="pronoun_ambiguity",
            group="discourse",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"pronoun_to_noun_ratio": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.6 — Style detectors
# ---------------------------------------------------------------------------

def _detect_style(
    raw_features: Dict[str, float | None],
    feature_stats: Dict[str, FeatureStats],
    N: int,
) -> List[Issue]:
    issues: List[Issue] = []

    # structural_inconsistency
    val = raw_features.get("pos_distribution_variance")
    if val is not None:
        st = _get_stats(feature_stats, "pos_distribution_variance")
        severity = soft_score(val, st.mean, st.std)
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="structural_inconsistency",
            group="style",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"pos_distribution_variance": val},
        ))

    # sentence_progression_drift
    val = raw_features.get("sentence_length_trend")
    if val is not None:
        st = _get_stats(feature_stats, "sentence_length_trend")
        severity = abs(soft_score(val, st.mean, st.std) - 0.5) * 2
        conf = _confidence(severity, [val], [st])
        issues.append(Issue(
            type="sentence_progression_drift",
            group="style",
            severity=severity,
            confidence=conf,
            span=(0, N),
            evidence={"sentence_length_trend": val},
        ))

    return issues


# ---------------------------------------------------------------------------
# Task 6.8 — Entry point: wire all detectors
# ---------------------------------------------------------------------------

def detect_issues(
    raw_features: Dict[str, float | None],
    sentence_metrics: List[SentenceMetrics],
    feature_stats: Dict[str, FeatureStats],
) -> List[Issue]:
    """Detect all linguistic issues from raw features, sentence metrics, and corpus stats.

    Returns a flat list of Issue objects across all 6 groups.
    N = len(sentence_metrics) is used for document-level span (0, N).
    """
    N = len(sentence_metrics)

    issues: List[Issue] = []
    issues.extend(_detect_morphology(raw_features, feature_stats, N))
    issues.extend(_detect_syntax(raw_features, sentence_metrics, feature_stats, N))
    issues.extend(_detect_lexicon(raw_features, feature_stats, N))
    issues.extend(_detect_structure(raw_features, feature_stats, N))
    issues.extend(_detect_discourse(raw_features, sentence_metrics, feature_stats, N))
    issues.extend(_detect_style(raw_features, feature_stats, N))

    return issues
