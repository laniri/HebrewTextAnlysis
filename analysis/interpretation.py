"""Integration entry point — combines Layer 4 (Diagnosis) and Layer 5 (Intervention).

Provides ``run_interpretation()`` which orchestrates the diagnosis engine
and intervention mapper in sequence, returning the combined
InterpretationOutput dict.

Requirements implemented: 19.1, 19.2, 19.3, 19.4, 23.1, 23.3.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from analysis.diagnosis_engine import run_diagnoses
from analysis.intervention_mapper import map_interventions
from analysis.issue_models import Issue


def run_interpretation(
    issues: List[Issue],
    scores: Dict[str, Optional[float]],
) -> dict:
    """Run diagnosis + intervention mapping and return the combined output.

    Parameters
    ----------
    issues:
        List of ``Issue`` objects from the existing analysis pipeline.
    scores:
        Composite score dict with keys ``"difficulty"``, ``"style"``,
        ``"fluency"``, ``"cohesion"``, ``"complexity"``; values are
        floats in [0, 1] or ``None``.

    Returns
    -------
    dict
        ``{"diagnoses": List[Diagnosis], "interventions": List[Intervention]}``
        where diagnoses are sorted by severity descending and
        interventions are sorted by priority descending.
    """
    diagnoses = run_diagnoses(issues, scores)
    interventions = map_interventions(diagnoses)
    return {"diagnoses": diagnoses, "interventions": interventions}
