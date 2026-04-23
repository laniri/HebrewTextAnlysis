"""ML Distillation Layer (Layer 6) — public API.

This package implements a multi-task transformer student model that learns
to predict linguistic scores, issues, and diagnoses directly from raw
Hebrew text, using the existing deterministic pipeline (Layers 1–5) as a
teacher.

Modules
-------
model       – LinguisticModel: encoder + 3 prediction heads.
export      – Data export: pipeline JSONs → training JSONL.
dataset     – LinguisticDataset: JSONL → PyTorch tensors.
trainer     – Shared training logic (model construction, loop, eval, checkpointing).
inference   – Fast path, hybrid mode, intervention derivation.
disagreement – Model vs pipeline comparison, training set expansion.

Requirements implemented: 27.1, 27.3, 27.4.
"""

from __future__ import annotations
