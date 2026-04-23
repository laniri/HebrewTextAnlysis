"""Default configuration values for the Hebrew Linguistic Profiling Engine."""

from hebrew_profiler.models import (
    DifficultyWeights,
    NormalizationRanges,
    PipelineConfig,
    StyleWeights,
)

# --- External service defaults ---
DEFAULT_YAP_URL = "http://localhost:8000/yap/heb/joint"
DEFAULT_STANZA_LANG = "he"

# --- Thresholds ---
DEFAULT_LONG_SENTENCE_THRESHOLD = 20

# --- Scoring weights ---
DEFAULT_DIFFICULTY_WEIGHTS = DifficultyWeights(
    w1=0.30,  # avg_sentence_length
    w2=0.25,  # avg_tree_depth
    w3=0.25,  # hapax_ratio
    w4=0.20,  # morphological_ambiguity
)

DEFAULT_STYLE_WEIGHTS = StyleWeights(
    a1=0.25,  # suffix_pronoun_ratio
    a3=0.25,  # hapax_ratio (negative)
    a4=0.20,  # sentence_length_trend
    a5=0.15,  # pos_distribution_variance (negative)
    a6=0.15,  # pronoun_to_noun_ratio
)

# --- Normalization ranges for min-max scoring ---
DEFAULT_NORMALIZATION_RANGES = NormalizationRanges(
    avg_sentence_length=(10.0, 40.0),
    avg_tree_depth=(4.0, 15.0),
    hapax_ratio=(0.15, 0.55),
    morphological_ambiguity=(4.0, 10.0),
    suffix_pronoun_ratio=(0.05, 0.50),
    sentence_length_variance=(0.0, 400.0),
    sentence_length_trend=(-1.5, 1.5),
    pos_distribution_variance=(0.0, 0.008),
    pronoun_to_noun_ratio=(0.0, 0.45),
    rare_word_ratio=(0.0, 0.3),
    content_word_ratio=(0.1, 0.8),
    connective_ratio=(0.0, 1.2),
    sentence_overlap=(0.0, 0.4),
    agreement_error_rate=(0.0, 0.3),
    dependency_distance_variance=(0.0, 27.0),
    clause_type_entropy=(2.0, 3.0),
)

# --- Workers ---
DEFAULT_WORKERS = 4

# --- Convenience: fully-populated default config ---
DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    yap_url=DEFAULT_YAP_URL,
    stanza_lang=DEFAULT_STANZA_LANG,
    long_sentence_threshold=DEFAULT_LONG_SENTENCE_THRESHOLD,
    difficulty_weights=DEFAULT_DIFFICULTY_WEIGHTS,
    style_weights=DEFAULT_STYLE_WEIGHTS,
    normalization_ranges=DEFAULT_NORMALIZATION_RANGES,
    pretty_output=False,
    workers=DEFAULT_WORKERS,
)
