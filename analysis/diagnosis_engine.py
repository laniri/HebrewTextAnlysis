"""Diagnosis engine — Layer 4 of the analysis pipeline.

Aggregates patterns of issues and composite scores into linguistically
meaningful diagnoses using weighted severity formulas and confidence-aware
activation thresholds.  Contains helper utilities and 8 diagnosis rules
consumed by ``run_diagnoses()``.

Requirements implemented: 3.1, 3.2, 3.3, 4–12, 21.1–21.6, 23.1, 23.3, 23.4.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from analysis.diagnosis_models import Diagnosis
from analysis.issue_models import Issue


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_issues(issues: List[Issue], issue_type: str) -> List[Issue]:
    """Return all issues whose ``type`` field matches *issue_type*."""
    return [i for i in issues if i.type == issue_type]


def _max_severity(issues: List[Issue], issue_type: str) -> float:
    """Return the maximum severity among issues matching *issue_type*.

    Returns 0.0 when no issues match.
    """
    matching = _get_issues(issues, issue_type)
    if not matching:
        return 0.0
    return max(i.severity for i in matching)


def _mean_severity(issues: List[Issue], issue_type: str) -> float:
    """Return the mean severity among issues matching *issue_type*.

    Returns 0.0 when no issues match.
    """
    matching = _get_issues(issues, issue_type)
    if not matching:
        return 0.0
    return sum(i.severity for i in matching) / len(matching)


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    """Compute ``sum(v * w) / sum(w)``.

    Returns 0.0 when *values* is empty.
    """
    if not values:
        return 0.0
    total = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    if weight_sum == 0.0:
        return 0.0
    return total / weight_sum


def _safe_score(
    scores: Dict[str, Optional[float]],
    key: str,
) -> tuple[float, dict]:
    """Look up a composite score, substituting 0.0 for ``None``.

    Returns ``(value, evidence_updates)`` where *evidence_updates* is an
    empty dict when the score is present, or ``{"{key}_missing": True}``
    when the score was ``None``.
    """
    value = scores.get(key)
    if value is None:
        return 0.0, {f"{key}_missing": True}
    return value, {}


# ---------------------------------------------------------------------------
# Diagnosis rules (each returns Diagnosis | None)
# ---------------------------------------------------------------------------

def _diagnose_low_lexical_diversity(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of low_lexical_diversity (0.7) + low_content_density (0.3).

    Activation threshold: > 0.6.
    """
    sev_lex = _max_severity(issues, "low_lexical_diversity")
    sev_den = _max_severity(issues, "low_content_density")

    severity = _weighted_mean([sev_lex, sev_den], [0.7, 0.3])
    if severity <= 0.6:
        return None

    supporting = (
        _get_issues(issues, "low_lexical_diversity")
        + _get_issues(issues, "low_content_density")
    )
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    return Diagnosis(
        type="low_lexical_diversity",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence={
            "max_low_lexical_diversity_severity": sev_lex,
            "max_low_content_density_severity": sev_den,
        },
    )


def _diagnose_pronoun_overuse(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of pronoun_ambiguity (0.8) + cohesion score (0.2).

    Activation threshold: > 0.6.
    """
    sev_pron = _max_severity(issues, "pronoun_ambiguity")
    cohesion_val, cohesion_ev = _safe_score(scores, "cohesion")

    severity = _weighted_mean([sev_pron, cohesion_val], [0.8, 0.2])
    if severity <= 0.6:
        return None

    supporting = _get_issues(issues, "pronoun_ambiguity")
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    evidence: dict = {
        "max_pronoun_ambiguity_severity": sev_pron,
        "cohesion_score": cohesion_val,
    }
    evidence.update(cohesion_ev)

    return Diagnosis(
        type="pronoun_overuse",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence=evidence,
    )


def _diagnose_low_cohesion(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of weak_cohesion (0.6) + missing_connectives (0.4).

    Activation threshold: > 0.6.
    """
    sev_coh = _max_severity(issues, "weak_cohesion")
    sev_con = _max_severity(issues, "missing_connectives")

    severity = _weighted_mean([sev_coh, sev_con], [0.6, 0.4])
    if severity <= 0.6:
        return None

    supporting = (
        _get_issues(issues, "weak_cohesion")
        + _get_issues(issues, "missing_connectives")
    )
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    return Diagnosis(
        type="low_cohesion",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence={
            "max_weak_cohesion_severity": sev_coh,
            "max_missing_connectives_severity": sev_con,
        },
    )


def _diagnose_sentence_over_complexity(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of sentence_complexity mean severity (0.7) + difficulty score (0.3).

    Uses mean severity (not max) so that a single complex sentence in an
    otherwise simple text does not inflate the diagnosis.

    Activation threshold: > 0.65.
    """
    sev_comp = _mean_severity(issues, "sentence_complexity")
    diff_val, diff_ev = _safe_score(scores, "difficulty")

    severity = _weighted_mean([sev_comp, diff_val], [0.7, 0.3])
    if severity <= 0.65:
        return None

    supporting = _get_issues(issues, "sentence_complexity")
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    evidence: dict = {
        "mean_sentence_complexity_severity": sev_comp,
        "difficulty_score": diff_val,
    }
    evidence.update(diff_ev)

    return Diagnosis(
        type="sentence_over_complexity",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence=evidence,
    )


def _diagnose_structural_inconsistency(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of structural_inconsistency (0.6) + fluency score (0.4).

    Activation threshold: > 0.6.
    """
    sev_struct = _max_severity(issues, "structural_inconsistency")
    flu_val, flu_ev = _safe_score(scores, "fluency")

    severity = _weighted_mean([sev_struct, flu_val], [0.6, 0.4])
    if severity <= 0.6:
        return None

    supporting = _get_issues(issues, "structural_inconsistency")
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    evidence: dict = {
        "max_structural_inconsistency_severity": sev_struct,
        "fluency_score": flu_val,
    }
    evidence.update(flu_ev)

    return Diagnosis(
        type="structural_inconsistency",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence=evidence,
    )


def _diagnose_low_morphological_richness(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Weighted mean of low_morphological_diversity (0.7) + complexity score (0.3).

    Activation threshold: > 0.6.
    """
    sev_morph = _max_severity(issues, "low_morphological_diversity")
    comp_val, comp_ev = _safe_score(scores, "complexity")

    severity = _weighted_mean([sev_morph, comp_val], [0.7, 0.3])
    if severity <= 0.6:
        return None

    supporting = _get_issues(issues, "low_morphological_diversity")
    confidence = min((i.confidence for i in supporting), default=0.0)
    supporting_issues = list(dict.fromkeys(i.type for i in supporting))
    supporting_spans = [i.span for i in supporting]

    evidence: dict = {
        "max_low_morphological_diversity_severity": sev_morph,
        "complexity_score": comp_val,
    }
    evidence.update(comp_ev)

    return Diagnosis(
        type="low_morphological_richness",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence=evidence,
    )


def _diagnose_fragmented_writing(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Direct max severity of fragmentation issues.

    Activation threshold: > 0.6.
    Confidence = confidence of the highest-severity fragmentation issue.
    """
    matching = _get_issues(issues, "fragmentation")
    severity = max((i.severity for i in matching), default=0.0)
    if severity <= 0.6:
        return None

    # Confidence of the highest-severity issue
    best = max(matching, key=lambda i: i.severity)
    confidence = best.confidence

    supporting_issues = list(dict.fromkeys(i.type for i in matching))
    supporting_spans = [i.span for i in matching]

    return Diagnosis(
        type="fragmented_writing",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence={"max_fragmentation_severity": severity},
    )


def _diagnose_punctuation_deficiency(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> Optional[Diagnosis]:
    """Direct max severity of punctuation_issues.

    Activation threshold: > 0.6.
    Confidence = confidence of the highest-severity punctuation_issues issue.
    """
    matching = _get_issues(issues, "punctuation_issues")
    severity = max((i.severity for i in matching), default=0.0)
    if severity <= 0.6:
        return None

    # Confidence of the highest-severity issue
    best = max(matching, key=lambda i: i.severity)
    confidence = best.confidence

    supporting_issues = list(dict.fromkeys(i.type for i in matching))
    supporting_spans = [i.span for i in matching]

    return Diagnosis(
        type="punctuation_deficiency",
        confidence=confidence,
        severity=severity,
        supporting_issues=supporting_issues,
        supporting_spans=supporting_spans,
        evidence={"max_punctuation_issues_severity": severity},
    )


# ---------------------------------------------------------------------------
# Public aggregation entry point
# ---------------------------------------------------------------------------

_ALL_DIAGNOSIS_RULES = [
    _diagnose_low_lexical_diversity,
    _diagnose_pronoun_overuse,
    _diagnose_low_cohesion,
    _diagnose_sentence_over_complexity,
    _diagnose_structural_inconsistency,
    _diagnose_low_morphological_richness,
    _diagnose_fragmented_writing,
    _diagnose_punctuation_deficiency,
]


def run_diagnoses(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> List[Diagnosis]:
    """Execute all 8 diagnosis rules, filter by threshold, sort by severity desc.

    Each rule returns ``Diagnosis | None``; ``None`` means the rule's
    activation threshold was not exceeded.  The returned list contains
    only activated diagnoses, sorted from highest to lowest severity.
    """
    results: List[Diagnosis] = []
    for rule in _ALL_DIAGNOSIS_RULES:
        diagnosis = rule(issues, scores)
        if diagnosis is not None:
            results.append(diagnosis)
    results.sort(key=lambda d: d.severity, reverse=True)
    return results
