"""Diagnosis and Intervention data models for Layers 4–5.

Defines the two dataclasses produced by the diagnosis engine (Layer 4)
and the intervention mapper (Layer 5).  Follows the same @dataclass
pattern as Issue in analysis/issue_models.py.

Requirements implemented: 1.1–1.7, 2.1–2.7, 23.1, 23.2, 23.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Diagnosis:
    type: str                                          # one of 8 diagnosis type strings
    confidence: float                                  # [0, 1]
    severity: float                                    # [0, 1]
    supporting_issues: List[str] = field(default_factory=list)   # issue type strings
    supporting_spans: List[tuple] = field(default_factory=list)  # spans from supporting issues
    evidence: dict = field(default_factory=dict)                 # raw computation values


@dataclass
class Intervention:
    type: str                                          # one of 4 intervention type strings
    priority: float                                    # [0, 1], equals target diagnosis severity
    target_diagnosis: str                              # diagnosis type that triggered this
    actions: List[str] = field(default_factory=list)   # pedagogical actions
    exercises: List[str] = field(default_factory=list) # recommended exercises
    focus_features: List[str] = field(default_factory=list)  # linguistic feature names
