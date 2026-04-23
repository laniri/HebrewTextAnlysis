"""Data models for the Hebrew Linguistic Profiling Engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NormalizationResult:
    normalized_text: str


@dataclass
class TokenizationResult:
    tokens: list[str]
    character_offsets: list[tuple[int, int]]
    prefix_annotations: list[list[str]]   # per-token list of detected prefixes
    suffix_annotations: list[str | None]   # per-token detected suffix or None


@dataclass
class MorphAnalysis:
    surface: str
    lemma: str
    pos: str
    gender: str | None
    number: str | None
    prefixes: list[str]
    suffix: str | None
    binyan: str | None
    tense: str | None
    ambiguity_count: int
    top_k_analyses: list[dict]


@dataclass
class StanzaResult:
    analyses: list[MorphAnalysis]


@dataclass
class StanzaSetupError:
    error_type: str
    message: str
    install_instructions: str


@dataclass
class StanzaError:
    error_type: str
    message: str


@dataclass
class DepTreeNode:
    id: int
    form: str
    lemma: str
    cpostag: str
    postag: str
    features: dict[str, str]
    head: int
    deprel: str


@dataclass
class SentenceTree:
    nodes: list[DepTreeNode]


@dataclass
class YAPResult:
    morphological_disambiguation: list[dict]
    sentences: list[SentenceTree]
    ambiguity_counts: list[int]  # per-token count of MA lattice analyses


@dataclass
class YAPError:
    error_type: str
    http_status: int | None
    message: str


@dataclass
class IRToken:
    surface: str
    offset: tuple[int, int]
    morph: MorphAnalysis | None
    dep_node: DepTreeNode | None
    prefixes: list[str]
    suffix: str | None


@dataclass
class IRSentence:
    tokens: list[IRToken]
    dep_tree: SentenceTree | None


@dataclass
class IntermediateRepresentation:
    original_text: str
    normalized_text: str
    sentences: list[IRSentence]
    missing_layers: list[str]


@dataclass
class MorphFeatures:
    verb_ratio: float | None
    binyan_distribution: dict[str, int] | None
    prefix_density: float | None
    suffix_pronoun_ratio: float | None
    morphological_ambiguity: float | None
    # New morphological control features
    agreement_error_rate: float | None
    binyan_entropy: float | None
    construct_ratio: float | None


@dataclass
class SyntaxFeatures:
    avg_sentence_length: float | None
    avg_tree_depth: float | None
    max_tree_depth: float | None
    avg_dependency_distance: float | None
    clauses_per_sentence: float | None
    # New syntactic complexity features
    subordinate_clause_ratio: float | None
    right_branching_ratio: float | None
    dependency_distance_variance: float | None
    # Structural diversity
    clause_type_entropy: float | None = None


@dataclass
class LexicalFeatures:
    type_token_ratio: float | None
    hapax_ratio: float | None
    avg_token_length: float | None
    lemma_diversity: float | None
    # New lexical sophistication features
    rare_word_ratio: float | None
    content_word_ratio: float | None


@dataclass
class StructuralFeatures:
    sentence_length_variance: float | None
    long_sentence_ratio: float | None
    # New fluency & readability features
    punctuation_ratio: float | None
    short_sentence_ratio: float | None
    missing_terminal_punctuation_ratio: float | None


@dataclass
class DiscourseFeatures:
    """Discourse & cohesion features."""
    connective_ratio: float | None
    sentence_overlap: float | None
    pronoun_to_noun_ratio: float | None


@dataclass
class StyleFeatures:
    """Stylistic consistency features."""
    sentence_length_trend: float | None
    pos_distribution_variance: float | None


@dataclass
class Features:
    morphology: MorphFeatures
    syntax: SyntaxFeatures
    lexicon: LexicalFeatures
    structure: StructuralFeatures
    discourse: DiscourseFeatures
    style: StyleFeatures


@dataclass
class Scores:
    difficulty: float | None
    style: float | None
    fluency: float | None = None
    cohesion: float | None = None
    complexity: float | None = None


@dataclass
class PipelineOutput:
    text: str
    features: Features
    scores: Scores
    normalized_features: dict[str, float | None] | None = None
    sentence_metrics: list[dict] | None = None  # per-sentence token_count, tree_depth, lemmas


@dataclass
class DifficultyWeights:
    w1: float = 0.30  # avg_sentence_length
    w2: float = 0.25  # avg_tree_depth
    w3: float = 0.25  # hapax_ratio
    w4: float = 0.20  # morphological_ambiguity


@dataclass
class StyleWeights:
    a1: float = 0.25   # suffix_pronoun_ratio
    a3: float = 0.25   # hapax_ratio (negative)
    a4: float = 0.20   # sentence_length_trend
    a5: float = 0.15   # pos_distribution_variance (negative)
    a6: float = 0.15   # pronoun_to_noun_ratio
    # a2 (sentence_length_variance) removed — belongs to fluency


@dataclass
class NormalizationRanges:
    avg_sentence_length: tuple[float, float] = (10.0, 40.0)
    avg_tree_depth: tuple[float, float] = (4.0, 15.0)
    hapax_ratio: tuple[float, float] = (0.15, 0.55)
    morphological_ambiguity: tuple[float, float] = (4.0, 10.0)
    # Style normalization ranges
    suffix_pronoun_ratio: tuple[float, float] = (0.05, 0.50)
    sentence_length_variance: tuple[float, float] = (0.0, 400.0)
    sentence_length_trend: tuple[float, float] = (-1.5, 1.5)
    pos_distribution_variance: tuple[float, float] = (0.0, 0.008)
    pronoun_to_noun_ratio: tuple[float, float] = (0.0, 0.45)
    # New feature normalization ranges
    rare_word_ratio: tuple[float, float] = (0.0, 0.3)
    content_word_ratio: tuple[float, float] = (0.1, 0.8)
    connective_ratio: tuple[float, float] = (0.0, 1.2)
    sentence_overlap: tuple[float, float] = (0.0, 0.4)
    agreement_error_rate: tuple[float, float] = (0.0, 0.3)
    dependency_distance_variance: tuple[float, float] = (0.0, 27.0)
    clause_type_entropy: tuple[float, float] = (2.0, 3.0)


@dataclass
class PipelineConfig:
    yap_url: str = "http://localhost:8000/yap/heb/joint"
    stanza_lang: str = "he"
    long_sentence_threshold: int = 20
    difficulty_weights: DifficultyWeights = field(default_factory=DifficultyWeights)
    style_weights: StyleWeights = field(default_factory=StyleWeights)
    normalization_ranges: NormalizationRanges = field(default_factory=NormalizationRanges)
    pretty_output: bool = False
    workers: int = 4
    freq_dict_path: str | None = None  # path to JSON frequency dictionary for rare_word_ratio


@dataclass
class BatchResult:
    total_processed: int
    error_count: int
    errors: list[dict]
