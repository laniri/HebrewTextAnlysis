"""JSON serialization for analysis output."""

import json
from typing import List

from analysis.issue_models import Issue


def serialize_issues(issues: List[Issue]) -> str:
    """Serialize a list of Issue objects to a JSON string.

    Returns a JSON object with a top-level "issues" array where each element
    contains all 6 fields: type, group, severity, confidence, span, evidence.
    span is serialized as a JSON array of integers.
    evidence is serialized as a JSON object with string keys and float values.
    """
    serialized = []
    for issue in issues:
        serialized.append({
            "type": issue.type,
            "group": issue.group,
            "severity": issue.severity,
            "confidence": issue.confidence,
            "span": list(issue.span),
            "evidence": {k: float(v) for k, v in issue.evidence.items()},
        })
    return json.dumps({"issues": serialized})

def serialize_interpretation(output: dict) -> str:
    """Serialize InterpretationOutput to JSON string with ensure_ascii=False.

    Follows the same conventions as ``serialize_issues()``: tuples become
    JSON arrays of integers, evidence values become floats, and Hebrew
    characters are preserved via ``ensure_ascii=False``.

    Parameters
    ----------
    output:
        Dict with keys ``"diagnoses"`` (``List[Diagnosis]``) and
        ``"interventions"`` (``List[Intervention]``).

    Returns
    -------
    str
        JSON string with top-level keys ``"diagnoses"`` and
        ``"interventions"``.
    """
    serialized_diagnoses = []
    for diag in output["diagnoses"]:
        serialized_diagnoses.append({
            "type": diag.type,
            "confidence": float(diag.confidence),
            "severity": float(diag.severity),
            "supporting_issues": list(diag.supporting_issues),
            "supporting_spans": [list(s) for s in diag.supporting_spans],
            "evidence": {k: float(v) for k, v in diag.evidence.items()},
        })

    serialized_interventions = []
    for interv in output["interventions"]:
        serialized_interventions.append({
            "type": interv.type,
            "priority": float(interv.priority),
            "target_diagnosis": interv.target_diagnosis,
            "actions": list(interv.actions),
            "exercises": list(interv.exercises),
            "focus_features": list(interv.focus_features),
        })

    return json.dumps(
        {"diagnoses": serialized_diagnoses, "interventions": serialized_interventions},
        ensure_ascii=False,
    )
