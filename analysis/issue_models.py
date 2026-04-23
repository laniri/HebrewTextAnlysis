from dataclasses import dataclass


@dataclass
class Issue:
    type: str        # e.g. "agreement_errors"
    group: str       # one of the 6 group names: morphology, syntax, lexicon, structure, discourse, style
    severity: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    span: tuple      # (i,) | (i-1, i) | (0, N)
    evidence: dict   # {feature_name: float}
