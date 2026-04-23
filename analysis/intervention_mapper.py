"""Intervention mapper — Layer 5 of the analysis pipeline.

Maps each Diagnosis produced by the diagnosis engine (Layer 4) to a
pedagogical Intervention using a static mapping table.  Each of the 8
diagnosis types maps to exactly one of 4 intervention types, with
pre-defined actions, exercises, and focus features.

Requirements implemented: 13.1–13.6, 14.1–14.6, 15.1–15.6, 16.1–16.6,
17.1, 17.2, 17.3, 18.1, 18.2, 18.3, 23.1, 23.3, 23.5.
"""

from __future__ import annotations

from typing import Dict, List

from analysis.diagnosis_models import Diagnosis, Intervention


# ---------------------------------------------------------------------------
# Static intervention mapping table
# ---------------------------------------------------------------------------

INTERVENTION_MAP: Dict[str, dict] = {
    # --- vocabulary_expansion (Req 13) ---
    "low_lexical_diversity": {
        "type": "vocabulary_expansion",
        "actions": [
            "Introduce synonym substitution exercises",
            "Encourage use of varied word forms",
            "Practice paraphrasing sentences with new vocabulary",
        ],
        "exercises": [
            "Rewrite a paragraph replacing repeated words with synonyms",
            "Build a personal vocabulary list from reading texts",
            "Complete cloze passages using diverse word choices",
        ],
        "focus_features": [
            "lemma_diversity",
            "type_token_ratio",
            "content_word_ratio",
        ],
    },
    "low_morphological_richness": {
        "type": "vocabulary_expansion",
        "actions": [
            "Introduce varied verb patterns (binyanim)",
            "Practice using different morphological forms",
            "Encourage construct-state (smichut) usage",
        ],
        "exercises": [
            "Conjugate verbs across multiple binyanim for the same root",
            "Rewrite sentences using construct-state noun phrases",
            "Identify and vary morphological patterns in a text",
        ],
        "focus_features": [
            "binyan_entropy",
            "type_token_ratio",
            "lemma_diversity",
        ],
    },
    # --- pronoun_clarification (Req 14) ---
    "pronoun_overuse": {
        "type": "pronoun_clarification",
        "actions": [
            "Replace ambiguous pronouns with explicit noun phrases",
            "Reduce pronoun density in consecutive sentences",
            "Clarify referential chains across paragraphs",
        ],
        "exercises": [
            "Identify all pronouns in a paragraph and replace ambiguous ones",
            "Rewrite passages reducing pronoun-to-noun ratio",
            "Match pronouns to their antecedents in sample texts",
        ],
        "focus_features": [
            "pronoun_to_noun_ratio",
        ],
    },
    # --- sentence_simplification (Req 15) ---
    "sentence_over_complexity": {
        "type": "sentence_simplification",
        "actions": [
            "Break long sentences into shorter units",
            "Reduce subordinate clause nesting",
            "Simplify deeply embedded syntactic structures",
        ],
        "exercises": [
            "Rewrite sentences exceeding 30 tokens into two or more",
            "Identify and extract embedded clauses",
            "Reduce tree depth by restructuring complex sentences",
        ],
        "focus_features": [
            "avg_sentence_length",
            "avg_tree_depth",
            "clauses_per_sentence",
        ],
    },
    "structural_inconsistency": {
        "type": "sentence_simplification",
        "actions": [
            "Standardise sentence length across the text",
            "Reduce variation in syntactic patterns",
            "Align sentence structures for consistent readability",
        ],
        "exercises": [
            "Rewrite paragraphs to even out sentence lengths",
            "Identify outlier sentences and restructure them",
            "Practice writing with a target sentence-length range",
        ],
        "focus_features": [
            "avg_sentence_length",
            "avg_tree_depth",
            "short_sentence_ratio",
        ],
    },
    "fragmented_writing": {
        "type": "sentence_simplification",
        "actions": [
            "Combine short fragmented sentences into fuller ones",
            "Add connective phrases between fragments",
            "Develop sentence-expansion techniques",
        ],
        "exercises": [
            "Merge consecutive short sentences into compound sentences",
            "Expand sentence fragments with additional clauses",
            "Rewrite a paragraph to eliminate fragments",
        ],
        "focus_features": [
            "avg_sentence_length",
            "short_sentence_ratio",
            "avg_tree_depth",
        ],
    },
    # --- cohesion_improvement (Req 16) ---
    "low_cohesion": {
        "type": "cohesion_improvement",
        "actions": [
            "Add discourse connectives between sentences",
            "Increase lexical overlap across adjacent sentences",
            "Use transitional phrases to link ideas",
        ],
        "exercises": [
            "Insert appropriate connectives into a passage",
            "Rewrite paragraphs to improve sentence-to-sentence overlap",
            "Identify missing transitions and add them",
        ],
        "focus_features": [
            "connective_ratio",
            "sentence_overlap",
        ],
    },
    "punctuation_deficiency": {
        "type": "cohesion_improvement",
        "actions": [
            "Review and correct punctuation usage",
            "Add missing terminal punctuation marks",
            "Use commas and periods to improve text flow",
        ],
        "exercises": [
            "Punctuate an unpunctuated passage correctly",
            "Identify and fix punctuation errors in sample texts",
            "Practice using commas to separate clauses",
        ],
        "focus_features": [
            "punctuation_ratio",
            "connective_ratio",
            "sentence_overlap",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public mapping function
# ---------------------------------------------------------------------------

def map_interventions(diagnoses: List[Diagnosis]) -> List[Intervention]:
    """Map each diagnosis to an intervention via *INTERVENTION_MAP*.

    Diagnoses whose ``type`` is not present in the map are silently
    skipped.  The returned list is sorted by ``priority`` in descending
    order, where priority equals the source diagnosis's severity.
    """
    interventions: List[Intervention] = []
    for diag in diagnoses:
        entry = INTERVENTION_MAP.get(diag.type)
        if entry is None:
            continue
        interventions.append(
            Intervention(
                type=entry["type"],
                priority=diag.severity,
                target_diagnosis=diag.type,
                actions=list(entry["actions"]),
                exercises=list(entry["exercises"]),
                focus_features=list(entry["focus_features"]),
            )
        )
    interventions.sort(key=lambda iv: iv.priority, reverse=True)
    return interventions
