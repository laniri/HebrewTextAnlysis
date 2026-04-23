"""Pipeline orchestrator for the Hebrew Linguistic Profiling Engine.

Chains all processing stages in fixed order:
Normalizer → Tokenizer → Stanza_Adapter + YAP_Adapter → IR_Builder →
Feature_Extractor → Scorer.

Provides JSON serialization helpers that preserve Hebrew characters
and represent absent features as null.
"""

from __future__ import annotations

import json
from typing import Any

from hebrew_profiler.feature_extractor import extract_features
from hebrew_profiler.ir_builder import build_ir
from hebrew_profiler.models import (
    Features,
    PipelineConfig,
    PipelineOutput,
    Scores,
)
from hebrew_profiler.normalizer import normalize
from hebrew_profiler.scorer import compute_composite_scores, compute_normalized_features, compute_scores
from hebrew_profiler.stanza_adapter import analyze_morphology
from hebrew_profiler.stanza_setup import check_stanza_model
from hebrew_profiler.tokenizer import tokenize
from hebrew_profiler.yap_adapter import parse_syntax


def _load_freq_dict(path: str | None) -> dict[str, int] | None:
    """Load a JSON frequency dictionary from disk, or return None if no path given."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_document(
    text: str,
    config: PipelineConfig,
) -> PipelineOutput:
    """Execute the full pipeline on a single document.

    Stages (fixed order):
    1. Verify Stanza Hebrew model availability
    2. Normalize text
    3. Tokenize normalized text
    4. Morphological analysis via Stanza
    5. Syntactic parsing via YAP
    6. Build Intermediate Representation
    7. Extract features
    8. Compute scores

    Args:
        text: Raw Hebrew input text.
        config: Pipeline configuration (weights, thresholds, URLs).

    Returns:
        PipelineOutput with original text, features, and scores.
    """
    # 0. Verify Stanza model is available
    check_stanza_model(lang=config.stanza_lang)

    # 1. Normalize
    normalization = normalize(text)

    # 2. Tokenize
    tokenization = tokenize(normalization.normalized_text)

    # 3. Stanza morphological analysis
    stanza_result = analyze_morphology(normalization.normalized_text)

    # 4. YAP syntactic parsing
    yap_result = parse_syntax(normalization.normalized_text, config.yap_url)

    # 5. Build IR
    ir = build_ir(text, normalization, tokenization, stanza_result, yap_result)

    # 5b. Extract per-sentence metrics from the IR for downstream use
    sentence_metrics_raw: list[dict] = []
    for i, sentence in enumerate(ir.sentences):
        token_count = len(sentence.tokens)
        # Compute tree depth via BFS
        tree_depth = 0.0
        if sentence.dep_tree is not None and sentence.dep_tree.nodes:
            from collections import deque
            nodes = sentence.dep_tree.nodes
            children: dict[int, list[int]] = {}
            root_id = None
            for node in nodes:
                children.setdefault(node.id, [])
                if node.head == 0:
                    root_id = node.id
                else:
                    children.setdefault(node.head, [])
                    children[node.head].append(node.id)
            if root_id is not None:
                queue: deque[tuple[int, int]] = deque([(root_id, 0)])
                visited: set[int] = set()
                while queue:
                    nid, depth = queue.popleft()
                    if nid in visited:
                        continue
                    visited.add(nid)
                    if depth > tree_depth:
                        tree_depth = float(depth)
                    for cid in children.get(nid, []):
                        if cid not in visited:
                            queue.append((cid, depth + 1))
        # Extract lemmas (fall back to surface form when morph is unavailable)
        lemmas = [
            token.morph.lemma if token.morph is not None else token.surface
            for token in sentence.tokens
        ]
        sentence_metrics_raw.append({
            "index": i,
            "token_count": token_count,
            "tree_depth": tree_depth,
            "lemmas": lemmas,
        })

    # 6. Extract features
    freq_dict = _load_freq_dict(config.freq_dict_path)
    features = extract_features(ir, config.long_sentence_threshold, freq_dict=freq_dict)

    # 7. Compute scores
    scores = compute_scores(
        features,
        config.difficulty_weights,
        config.style_weights,
        config.normalization_ranges,
    )

    # 8. Compute normalized features and composite scores
    normalized = compute_normalized_features(features, config.normalization_ranges)
    composites = compute_composite_scores(normalized)
    scores.fluency = composites["fluency"]
    scores.cohesion = composites["cohesion"]
    scores.complexity = composites["complexity"]

    return PipelineOutput(
        text=text, features=features, scores=scores,
        normalized_features=normalized,
        sentence_metrics=sentence_metrics_raw,
    )


def _morph_features_to_dict(mf: Any) -> dict[str, Any]:
    """Convert MorphFeatures to a JSON-serializable dict."""
    return {
        "verb_ratio": mf.verb_ratio,
        "binyan_distribution": mf.binyan_distribution,
        "prefix_density": mf.prefix_density,
        "suffix_pronoun_ratio": mf.suffix_pronoun_ratio,
        "morphological_ambiguity": mf.morphological_ambiguity,
        "agreement_error_rate": mf.agreement_error_rate,
        "binyan_entropy": mf.binyan_entropy,
        "construct_ratio": mf.construct_ratio,
    }


def _syntax_features_to_dict(sf: Any) -> dict[str, Any]:
    """Convert SyntaxFeatures to a JSON-serializable dict."""
    return {
        "avg_sentence_length": sf.avg_sentence_length,
        "avg_tree_depth": sf.avg_tree_depth,
        "max_tree_depth": sf.max_tree_depth,
        "avg_dependency_distance": sf.avg_dependency_distance,
        "clauses_per_sentence": sf.clauses_per_sentence,
        "subordinate_clause_ratio": sf.subordinate_clause_ratio,
        "right_branching_ratio": sf.right_branching_ratio,
        "dependency_distance_variance": sf.dependency_distance_variance,
        "clause_type_entropy": sf.clause_type_entropy,
    }


def _lexical_features_to_dict(lf: Any) -> dict[str, Any]:
    """Convert LexicalFeatures to a JSON-serializable dict."""
    return {
        "type_token_ratio": lf.type_token_ratio,
        "hapax_ratio": lf.hapax_ratio,
        "avg_token_length": lf.avg_token_length,
        "lemma_diversity": lf.lemma_diversity,
        "rare_word_ratio": lf.rare_word_ratio,
        "content_word_ratio": lf.content_word_ratio,
    }


def _structural_features_to_dict(sf: Any) -> dict[str, Any]:
    """Convert StructuralFeatures to a JSON-serializable dict."""
    return {
        "sentence_length_variance": sf.sentence_length_variance,
        "long_sentence_ratio": sf.long_sentence_ratio,
        "punctuation_ratio": sf.punctuation_ratio,
        "short_sentence_ratio": sf.short_sentence_ratio,
        "missing_terminal_punctuation_ratio": sf.missing_terminal_punctuation_ratio,
    }


def _discourse_features_to_dict(df: Any) -> dict[str, Any]:
    """Convert DiscourseFeatures to a JSON-serializable dict."""
    return {
        "connective_ratio": df.connective_ratio,
        "sentence_overlap": df.sentence_overlap,
        "pronoun_to_noun_ratio": df.pronoun_to_noun_ratio,
    }


def _style_features_to_dict(sf: Any) -> dict[str, Any]:
    """Convert StyleFeatures to a JSON-serializable dict."""
    return {
        "sentence_length_trend": sf.sentence_length_trend,
        "pos_distribution_variance": sf.pos_distribution_variance,
    }


def pipeline_output_to_dict(output: PipelineOutput) -> dict:
    """Convert PipelineOutput to a JSON-serializable dict.

    Structure:
    {
        "text": "...",
        "features": { ... },
        "scores": { ... }
    }

    None feature values are preserved (serialized as null in JSON).
    """
    result = {
        "text": output.text,
        "features": {
            "morphology": _morph_features_to_dict(output.features.morphology),
            "syntax": _syntax_features_to_dict(output.features.syntax),
            "lexicon": _lexical_features_to_dict(output.features.lexicon),
            "structure": _structural_features_to_dict(output.features.structure),
            "discourse": _discourse_features_to_dict(output.features.discourse),
            "style": _style_features_to_dict(output.features.style),
        },
        "scores": {
            "difficulty": output.scores.difficulty,
            "style": output.scores.style,
            "fluency": output.scores.fluency,
            "cohesion": output.scores.cohesion,
            "complexity": output.scores.complexity,
        },
    }
    if output.sentence_metrics is not None:
        result["sentence_metrics"] = output.sentence_metrics
    return result


def pipeline_output_to_json(output: PipelineOutput, pretty: bool = False) -> str:
    """Serialize PipelineOutput to a JSON string.

    Uses ``ensure_ascii=False`` and UTF-8 compatible output so that
    Hebrew characters are preserved in their native Unicode form.

    Args:
        output: The pipeline result to serialize.
        pretty: When True, indent the JSON with 2 spaces.

    Returns:
        A JSON string with top-level keys ``"text"``, ``"features"``,
        and ``"scores"``.
    """
    d = pipeline_output_to_dict(output)
    indent = 2 if pretty else None
    return json.dumps(d, ensure_ascii=False, indent=indent)
