"""Analysis pipeline orchestration for the probabilistic analysis layer.

Provides:
- flatten_features(): flattens a Features object into a flat Dict[str, float | None]
- AnalysisInput: dataclass combining raw features, per-sentence metrics, sentences, and scores
- run_analysis_pipeline(): runs the existing pipeline once and returns AnalysisInput
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from hebrew_profiler.feature_extractor import extract_features
from hebrew_profiler.ir_builder import build_ir
from hebrew_profiler.models import Features, PipelineConfig, Scores
from hebrew_profiler.normalizer import normalize
from hebrew_profiler.scorer import compute_composite_scores, compute_normalized_features, compute_scores
from hebrew_profiler.stanza_adapter import analyze_morphology
from hebrew_profiler.stanza_setup import check_stanza_model
from hebrew_profiler.tokenizer import tokenize
from hebrew_profiler.yap_adapter import _SENTENCE_SPLIT_RE, parse_syntax

from analysis.sentence_metrics import SentenceMetrics, extract_sentence_metrics

if TYPE_CHECKING:
    from analysis.embedder import SentenceEmbedder


def flatten_features(features: Features) -> Dict[str, float | None]:
    """Flatten the nested Features object into a flat dict of raw values.

    Iterates over each feature group dataclass (morphology, syntax, lexicon,
    structure, discourse, style) and extracts scalar float|None fields.
    Uses field names without group prefixes:
      features.morphology.agreement_error_rate -> {"agreement_error_rate": 0.0625}

    Non-scalar fields (e.g., binyan_distribution: dict) are skipped.

    Requirements: 15.3, 15.4, 17.4
    """
    result: Dict[str, float | None] = {}

    groups = [
        features.morphology,
        features.syntax,
        features.lexicon,
        features.structure,
        features.discourse,
        features.style,
    ]

    for group in groups:
        for f in dataclasses.fields(group):
            value = getattr(group, f.name)
            if isinstance(value, (int, float)):
                result[f.name] = float(value)
            elif value is None and "float" in str(f.type):
                result[f.name] = None

    return result


@dataclass
class AnalysisInput:
    raw_features: Dict[str, float | None]   # flattened raw feature values from Features object
    sentence_metrics: List[SentenceMetrics]  # per-sentence data from IR
    sentence_count: int                      # len(ir.sentences)
    sentences: List[str]                     # original sentence texts (from normalized text split)
    scores: Dict[str, Optional[float]]       # difficulty, style, fluency, cohesion, complexity


def run_analysis_pipeline(
    text: str,
    config: PipelineConfig,
    embedder: "SentenceEmbedder | None" = None,
) -> AnalysisInput:
    """Run the existing pipeline once and extract both document-level features
    and per-sentence metrics.

    Steps:
    1. Run existing pipeline stages (normalize -> tokenize -> stanza -> yap -> build_ir)
    2. Extract document-level features via existing extract_features()
    3. Extract per-sentence metrics from the same IR via extract_sentence_metrics()
    4. Flatten features to raw dict (NOT min-max normalized)
    5. Return AnalysisInput

    Does NOT call process_document() per sentence.
    Does NOT use normalized_features from scorer.py.

    Requirements: 16.1, 17.1, 17.2, 17.3, 17.4, 17.5
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

    # 6. Load frequency dictionary if configured
    freq_dict: dict[str, int] | None = None
    if config.freq_dict_path is not None:
        with open(config.freq_dict_path, "r", encoding="utf-8") as f:
            freq_dict = json.load(f)

    # 7. Extract document-level features
    features = extract_features(ir, config.long_sentence_threshold, freq_dict=freq_dict)

    # 8. Compute scores (difficulty, style, fluency, cohesion, complexity)
    pipeline_scores = compute_scores(
        features,
        config.difficulty_weights,
        config.style_weights,
        config.normalization_ranges,
    )
    normalized = compute_normalized_features(features, config.normalization_ranges)
    composites = compute_composite_scores(normalized)
    pipeline_scores.fluency = composites["fluency"]
    pipeline_scores.cohesion = composites["cohesion"]
    pipeline_scores.complexity = composites["complexity"]

    scores = {
        "difficulty": pipeline_scores.difficulty,
        "style": pipeline_scores.style,
        "fluency": pipeline_scores.fluency,
        "cohesion": pipeline_scores.cohesion,
        "complexity": pipeline_scores.complexity,
    }

    # 11. Extract original sentence texts by splitting the normalized text
    #     (same regex YAP uses) — preserves original Hebrew without token splitting
    sentences = [
        s.strip()
        for s in _SENTENCE_SPLIT_RE.split(normalization.normalized_text.strip())
        if s.strip()
    ]

    # 9. Extract per-sentence metrics from the same IR, with optional embeddings
    sentence_metrics = extract_sentence_metrics(ir, sentences=sentences, embedder=embedder)

    # 10. Flatten features to raw dict
    raw_features = flatten_features(features)

    return AnalysisInput(
        raw_features=raw_features,
        sentence_metrics=sentence_metrics,
        sentence_count=len(ir.sentences),
        sentences=sentences,
        scores=scores,
    )
