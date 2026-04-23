"""Feature extraction for the Hebrew Linguistic Profiling Engine.

Computes morphological, syntactic, lexical, and structural features
from an IntermediateRepresentation. Handles missing IR layers gracefully
by setting affected feature categories to None.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, variance

from hebrew_profiler.models import (
    DiscourseFeatures,
    Features,
    IntermediateRepresentation,
    IRSentence,
    IRToken,
    LexicalFeatures,
    MorphFeatures,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)

# Subordinate dependency relations used for clauses_per_sentence
_SUBORDINATE_RELS = frozenset({"rcmod", "ccomp", "xcomp", "advcl", "complm"})

# Subordinate clause relations for subordinate_clause_ratio
_SUBORDINATE_CLAUSE_RELS = frozenset({"ccomp", "xcomp", "advcl", "acl", "relcl"})
_CLAUSE_RELS = _SUBORDINATE_CLAUSE_RELS | {"root"}

# Rare word frequency threshold for rare_word_ratio
_RARE_THRESHOLD = 5

# Content POS tags for content_word_ratio
_CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

# Terminal punctuation marks for missing_terminal_punctuation_ratio
# Includes colon (:) for structured/legal text that uses it to introduce lists
_TERMINAL_PUNCT = frozenset({".", "!", "?", "…", ":"})


# ---------------------------------------------------------------------------
# Internal Cache
# ---------------------------------------------------------------------------

@dataclass
class _FeatureCache:
    all_tokens: list[IRToken]
    total_tokens: int
    pos_counts: Counter[str]                    # POS tag -> count
    sentence_lengths: list[int]                 # tokens per sentence
    dependencies: list[tuple[int, int, str]]    # (token_id, head_id, deprel)
    lemma_sets_per_sentence: list[set[str]]     # lemma set per sentence
    sentences: list[IRSentence]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_tokens(ir: IntermediateRepresentation) -> list[IRToken]:
    """Flatten all tokens from all sentences in the IR."""
    tokens: list[IRToken] = []
    for sentence in ir.sentences:
        tokens.extend(sentence.tokens)
    return tokens


def _build_cache(ir: IntermediateRepresentation) -> _FeatureCache:
    """Precompute shared data structures from the IR.

    Built once at the start of extract_features() and passed to all
    extraction functions to avoid redundant iteration.
    """
    all_tokens = _all_tokens(ir)
    total_tokens = len(all_tokens)

    pos_counts: Counter[str] = Counter()
    for t in all_tokens:
        if t.morph is not None:
            pos_counts[t.morph.pos] += 1

    sentence_lengths = [len(s.tokens) for s in ir.sentences]

    dependencies: list[tuple[int, int, str]] = []
    for s in ir.sentences:
        for t in s.tokens:
            if t.dep_node is not None:
                dependencies.append((t.dep_node.id, t.dep_node.head, t.dep_node.deprel))

    lemma_sets: list[set[str]] = []
    for s in ir.sentences:
        lemmas = set()
        for t in s.tokens:
            if t.morph is not None:
                lemmas.add(t.morph.lemma)
            else:
                lemmas.add(t.surface)
        lemma_sets.append(lemmas)

    return _FeatureCache(
        all_tokens=all_tokens,
        total_tokens=total_tokens,
        pos_counts=pos_counts,
        sentence_lengths=sentence_lengths,
        dependencies=dependencies,
        lemma_sets_per_sentence=lemma_sets,
        sentences=ir.sentences,
    )


def _compute_tree_depth(sentence: IRSentence) -> float:
    """Compute the maximum depth of a dependency tree for a sentence.

    Builds an adjacency list from dep_node data and performs BFS from
    the root (head == 0) to find the maximum depth.
    Returns 0.0 if no dependency tree is available.
    """
    if sentence.dep_tree is None or not sentence.dep_tree.nodes:
        return 0.0

    # Build children map: head_id -> list of child node ids
    children: dict[int, list[int]] = defaultdict(list)
    for node in sentence.dep_tree.nodes:
        children[node.head].append(node.id)

    # BFS from virtual root (id=0)
    max_depth = 0
    queue: list[tuple[int, int]] = [(0, 0)]  # (node_id, depth)
    while queue:
        node_id, depth = queue.pop(0)
        if depth > max_depth:
            max_depth = depth
        for child_id in children.get(node_id, []):
            queue.append((child_id, depth + 1))

    return float(max_depth)


# ---------------------------------------------------------------------------
# Morphological control helpers
# ---------------------------------------------------------------------------

def _agreement_error_rate(sentences: list[IRSentence]) -> float:
    """Compute fraction of nsubj/amod pairs with gender/number mismatch.

    Returns 0.0 if no qualifying pairs are found.
    None morph values are treated as "no data" (not counted as mismatch).
    Gender mismatch is checked first; only one mismatch counted per pair.
    """
    total_pairs = 0
    mismatch_count = 0

    for s in sentences:
        for t in s.tokens:
            if t.dep_node is None or t.morph is None:
                continue
            if t.dep_node.deprel not in ("nsubj", "amod"):
                continue
            # Find head token by matching dep_node.id == head_id
            head_id = t.dep_node.head
            head_token = None
            for ht in s.tokens:
                if ht.dep_node is not None and ht.dep_node.id == head_id:
                    head_token = ht
                    break
            if head_token is None or head_token.morph is None:
                continue

            total_pairs += 1
            # Check gender mismatch first
            if (t.morph.gender is not None and head_token.morph.gender is not None
                    and t.morph.gender != head_token.morph.gender):
                mismatch_count += 1
            elif (t.morph.number is not None and head_token.morph.number is not None
                    and t.morph.number != head_token.morph.number):
                mismatch_count += 1

    if total_pairs == 0:
        return 0.0
    return mismatch_count / total_pairs


def _binyan_entropy(binyan_distribution: dict[str, int] | None) -> float:
    """Compute Shannon entropy (natural log) over binyan distribution.

    Returns 0.0 if distribution is None or empty.
    """
    if binyan_distribution is None or not binyan_distribution:
        return 0.0
    total = sum(binyan_distribution.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in binyan_distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def _construct_ratio(sentences: list[IRSentence], total: int) -> float:
    """Count adjacent noun-noun pairs / total tokens.

    Returns 0.0 if total is zero.
    """
    if total == 0:
        return 0.0
    adjacent_noun_pairs = 0
    for s in sentences:
        for i in range(len(s.tokens) - 1):
            t1, t2 = s.tokens[i], s.tokens[i + 1]
            if (t1.morph is not None and t2.morph is not None
                    and t1.morph.pos == "NOUN" and t2.morph.pos == "NOUN"):
                adjacent_noun_pairs += 1
    return adjacent_noun_pairs / total


# ---------------------------------------------------------------------------
# 9.1  Morphological feature extraction
# ---------------------------------------------------------------------------

def _extract_morphological(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
) -> MorphFeatures:
    """Compute morphological features from the IR.

    Requirements 7.1-7.6.
    """
    tokens = cache.all_tokens
    total = cache.total_tokens

    if total == 0:
        return MorphFeatures(
            verb_ratio=0.0,
            binyan_distribution={},
            prefix_density=0.0,
            suffix_pronoun_ratio=0.0,
            morphological_ambiguity=0.0,
            agreement_error_rate=0.0,
            binyan_entropy=0.0,
            construct_ratio=0.0,
        )

    # 7.1 verb_ratio: count(POS == "VERB") / total_tokens
    verb_count = sum(
        1 for t in tokens if t.morph is not None and t.morph.pos == "VERB"
    )
    verb_ratio = verb_count / total

    # 7.2 binyan_distribution: histogram {binyan_type: count}
    binyan_dist: dict[str, int] = {}
    for t in tokens:
        if t.morph is not None and t.morph.binyan is not None:
            binyan_dist[t.morph.binyan] = binyan_dist.get(t.morph.binyan, 0) + 1

    # 7.3 prefix_density: total_prefixes / total_tokens
    total_prefixes = sum(len(t.prefixes) for t in tokens)
    prefix_density = total_prefixes / total

    # 7.4 suffix_pronoun_ratio: count(tokens_with_suffix) / total_tokens
    suffix_count = sum(1 for t in tokens if t.suffix is not None)
    suffix_pronoun_ratio = suffix_count / total

    # 7.5 morphological_ambiguity: mean(ambiguity_count per token)
    ambiguity_values = [
        t.morph.ambiguity_count for t in tokens if t.morph is not None
    ]
    morphological_ambiguity = mean(ambiguity_values) if ambiguity_values else 0.0

    return MorphFeatures(
        verb_ratio=verb_ratio,
        binyan_distribution=binyan_dist,
        prefix_density=prefix_density,
        suffix_pronoun_ratio=suffix_pronoun_ratio,
        morphological_ambiguity=morphological_ambiguity,
        agreement_error_rate=_agreement_error_rate(cache.sentences),
        binyan_entropy=_binyan_entropy(binyan_dist),
        construct_ratio=_construct_ratio(cache.sentences, total),
    )


# ---------------------------------------------------------------------------
# 9.3  Syntactic feature extraction
# ---------------------------------------------------------------------------

def _extract_syntactic(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
) -> SyntaxFeatures:
    """Compute syntactic features from the IR.

    Requirements 8.1-8.6.
    """
    sentences = cache.sentences
    num_sentences = len(sentences)

    if num_sentences == 0:
        return SyntaxFeatures(
            avg_sentence_length=0.0,
            avg_tree_depth=0.0,
            max_tree_depth=0.0,
            avg_dependency_distance=0.0,
            clauses_per_sentence=0.0,
            subordinate_clause_ratio=0.0,
            right_branching_ratio=0.0,
            dependency_distance_variance=0.0,
            clause_type_entropy=0.0,
        )

    # 8.1 avg_sentence_length: mean(token_count per sentence)
    sentence_lengths = cache.sentence_lengths
    avg_sentence_length = mean(sentence_lengths) if sentence_lengths else 0.0

    # 8.2 avg_tree_depth: mean(max_depth per sentence dep tree)
    tree_depths = [_compute_tree_depth(s) for s in sentences]
    avg_tree_depth = mean(tree_depths) if tree_depths else 0.0

    # 8.3 max_tree_depth: max(max_depth across all sentence dep trees)
    max_tree_depth = max(tree_depths) if tree_depths else 0.0

    # 8.4 avg_dependency_distance: mean(|token_pos - head_pos|) for all tokens
    distances: list[float] = []
    for s in sentences:
        for t in s.tokens:
            if t.dep_node is not None:
                distances.append(abs(t.dep_node.id - t.dep_node.head))
    avg_dependency_distance = mean(distances) if distances else 0.0

    # 8.5 clauses_per_sentence: count(subordinate_dep_relations) / num_sentences
    subordinate_count = 0
    for s in sentences:
        for t in s.tokens:
            if t.dep_node is not None and t.dep_node.deprel in _SUBORDINATE_RELS:
                subordinate_count += 1
    clauses_per_sentence = subordinate_count / num_sentences

    # 3.1 subordinate_clause_ratio: subordinate clause deps / all clause deps
    deps = cache.dependencies
    sub_clause_count = sum(1 for _, _, rel in deps if rel in _SUBORDINATE_CLAUSE_RELS)
    total_clause_count = sum(1 for _, _, rel in deps if rel in _CLAUSE_RELS)
    subordinate_clause_ratio = (
        sub_clause_count / total_clause_count if total_clause_count > 0 else 0.0
    )

    # 3.2 right_branching_ratio: non-root deps where tid > hid / total non-root
    non_root = [(tid, hid) for tid, hid, _ in deps if hid != 0]
    if non_root:
        right_count = sum(1 for tid, hid in non_root if tid > hid)
        right_branching_ratio = right_count / len(non_root)
    else:
        right_branching_ratio = 0.0

    # 3.3 dependency_distance_variance: sample variance of |tid - hid| for non-root
    non_root_distances = [abs(tid - hid) for tid, hid, _ in deps if hid != 0]
    if len(non_root_distances) < 2:
        dependency_distance_variance = 0.0
    else:
        dependency_distance_variance = variance(non_root_distances)

    # clause_type_entropy: Shannon entropy over dependency relation type distribution
    # Captures diversity of syntactic constructions used in the document.
    # p_i = count(deprel_i) / total_deps; entropy = -Σ p_i * log(p_i)
    if deps:
        deprel_counts: Counter[str] = Counter(rel for _, _, rel in deps)
        total_deps = len(deps)
        clause_type_entropy = 0.0
        for count in deprel_counts.values():
            if count > 0:
                p = count / total_deps
                clause_type_entropy -= p * math.log(p)
    else:
        clause_type_entropy = 0.0

    return SyntaxFeatures(
        avg_sentence_length=avg_sentence_length,
        avg_tree_depth=avg_tree_depth,
        max_tree_depth=max_tree_depth,
        avg_dependency_distance=avg_dependency_distance,
        clauses_per_sentence=clauses_per_sentence,
        subordinate_clause_ratio=subordinate_clause_ratio,
        right_branching_ratio=right_branching_ratio,
        dependency_distance_variance=dependency_distance_variance,
        clause_type_entropy=clause_type_entropy,
    )


# ---------------------------------------------------------------------------
# 9.5  Lexical feature extraction
# ---------------------------------------------------------------------------

def _extract_lexical(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
    freq_dict: dict[str, int] | None = None,
) -> LexicalFeatures:
    """Compute lexical features from the IR.

    Requirements 9.1-9.5.
    """
    tokens = cache.all_tokens
    total = cache.total_tokens

    if total == 0:
        return LexicalFeatures(
            type_token_ratio=0.0,
            hapax_ratio=0.0,
            avg_token_length=0.0,
            lemma_diversity=0.0,
            rare_word_ratio=None if freq_dict is None else 0.0,
            content_word_ratio=0.0,
        )

    surfaces = [t.surface for t in tokens]

    # 9.1 type_token_ratio: unique_surface_forms / total_tokens
    type_token_ratio = len(set(surfaces)) / total

    # 9.2 hapax_ratio: count(frequency==1 tokens) / total_tokens
    freq = Counter(surfaces)
    hapax_count = sum(1 for count in freq.values() if count == 1)
    hapax_ratio = hapax_count / total

    # 9.3 avg_token_length: mean(len(surface_form))
    avg_token_length = mean(len(s) for s in surfaces)

    # 9.4 lemma_diversity: unique_lemmas / total_tokens
    # Fall back to surface form when morph data is unavailable
    lemmas = [
        t.morph.lemma if t.morph is not None else t.surface
        for t in tokens
    ]
    lemma_diversity = len(set(lemmas)) / total

    # 9.5a rare_word_ratio: tokens whose lemma freq < _RARE_THRESHOLD / total
    if freq_dict is None:
        rare_word_ratio: float | None = None
    else:
        rare_count = 0
        for t in tokens:
            lemma = t.morph.lemma if t.morph is not None else t.surface
            if freq_dict.get(lemma, 0) < _RARE_THRESHOLD:
                rare_count += 1
        rare_word_ratio = rare_count / total

    # 9.5b content_word_ratio: NOUN/VERB/ADJ/ADV tokens / total
    content_count = sum(cache.pos_counts.get(pos, 0) for pos in _CONTENT_POS)
    content_word_ratio = content_count / total

    return LexicalFeatures(
        type_token_ratio=type_token_ratio,
        hapax_ratio=hapax_ratio,
        avg_token_length=avg_token_length,
        lemma_diversity=lemma_diversity,
        rare_word_ratio=rare_word_ratio,
        content_word_ratio=content_word_ratio,
    )


# ---------------------------------------------------------------------------
# 9.7  Structural feature extraction
# ---------------------------------------------------------------------------

def _extract_structural(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
    long_sentence_threshold: int = 20,
) -> StructuralFeatures:
    """Compute structural features from the IR.

    Requirements 10.1-10.3.
    """
    sentences = cache.sentences
    num_sentences = len(sentences)

    if num_sentences == 0:
        return StructuralFeatures(
            sentence_length_variance=0.0,
            long_sentence_ratio=0.0,
            punctuation_ratio=0.0,
            short_sentence_ratio=0.0,
            missing_terminal_punctuation_ratio=0.0,
        )

    sentence_lengths = cache.sentence_lengths

    # 10.1 sentence_length_variance: variance(token_counts per sentence)
    # 10.3 0.0 if fewer than 2 sentences
    if num_sentences < 2:
        sent_var = 0.0
    else:
        sent_var = variance(sentence_lengths)

    # 10.2 long_sentence_ratio: count(sentences > threshold) / num_sentences
    long_count = sum(1 for length in sentence_lengths if length > long_sentence_threshold)
    long_sentence_ratio = long_count / num_sentences

    # 2.1 punctuation_ratio: PUNCT count / total tokens (0.0 if zero tokens)
    total = cache.total_tokens
    if total == 0:
        punct_ratio = 0.0
    else:
        punct_count = sum(
            1 for t in cache.all_tokens
            if t.morph is not None and t.morph.pos == "PUNCT"
        )
        punct_ratio = punct_count / total

    # 2.2 short_sentence_ratio: sentences with < 3 tokens / total sentences
    short_count = sum(1 for s in sentences if len(s.tokens) < 3)
    short_sent_ratio = short_count / num_sentences

    # 2.3 missing_terminal_punctuation_ratio: sentence has no terminal punctuation token
    # Checks any token in the sentence (not just last) since YAP may reorder tokens
    # relative to our sentence splitter. A sentence is considered punctuated if any
    # token's surface is in _TERMINAL_PUNCT.
    missing = 0
    for s in sentences:
        if not s.tokens:
            missing += 1
            continue
        has_terminal = any(t.surface in _TERMINAL_PUNCT for t in s.tokens)
        if not has_terminal:
            missing += 1
    missing_terminal_ratio = missing / num_sentences

    return StructuralFeatures(
        sentence_length_variance=sent_var,
        long_sentence_ratio=long_sentence_ratio,
        punctuation_ratio=punct_ratio,
        short_sentence_ratio=short_sent_ratio,
        missing_terminal_punctuation_ratio=missing_terminal_ratio,
    )


# ---------------------------------------------------------------------------
# Discourse feature extraction
# ---------------------------------------------------------------------------

_CONNECTIVES = frozenset({
    "אבל", "אולם", "אם", "אז", "גם", "או", "כי", "לכן",
    "משום", "בגלל", "למרות", "אלא", "עם זאת", "בנוסף",
    "לעומת", "כלומר", "כך", "אכן", "הרי", "שכן",
})

_EPSILON = 1e-10


def _connective_ratio(
    sentences: list[IRSentence],
    all_tokens: list[IRToken],
) -> float:
    """Count connective tokens / total sentences. 0.0 if zero sentences."""
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0.0
    conn_count = sum(1 for t in all_tokens if t.surface in _CONNECTIVES)
    return conn_count / num_sentences


def _sentence_overlap(lemma_sets: list[set[str]]) -> float:
    """Mean Jaccard similarity of adjacent sentence lemma sets.

    Returns 0.0 if fewer than 2 sentences. Empty set pairs yield 0.0.
    """
    if len(lemma_sets) < 2:
        return 0.0
    jaccard_scores: list[float] = []
    for i in range(len(lemma_sets) - 1):
        a, b = lemma_sets[i], lemma_sets[i + 1]
        union = a | b
        if not union:
            jaccard_scores.append(0.0)
        else:
            jaccard_scores.append(len(a & b) / len(union))
    return statistics.mean(jaccard_scores)


def _pronoun_to_noun_ratio(pos_counts: Counter[str], total: int) -> float:
    """(PRON/total) / (NOUN/total + epsilon). 0.0 if zero tokens."""
    if total == 0:
        return 0.0
    pron_ratio = pos_counts.get("PRON", 0) / total
    noun_ratio = pos_counts.get("NOUN", 0) / total + _EPSILON
    return pron_ratio / noun_ratio


def _extract_discourse(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
) -> DiscourseFeatures:
    """Compute discourse cohesion features from the IR.

    Requirements 6.1-6.8. Always computed regardless of missing_layers.
    """
    return DiscourseFeatures(
        connective_ratio=_connective_ratio(cache.sentences, cache.all_tokens),
        sentence_overlap=_sentence_overlap(cache.lemma_sets_per_sentence),
        pronoun_to_noun_ratio=_pronoun_to_noun_ratio(cache.pos_counts, cache.total_tokens),
    )


# ---------------------------------------------------------------------------
# Style feature extraction
# ---------------------------------------------------------------------------

def _sentence_length_trend(sentence_lengths: list[int]) -> float:
    """Compute linear regression slope over sentence lengths.

    slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    Returns 0.0 if fewer than 2 sentences or zero denominator.
    """
    n = len(sentence_lengths)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = mean(sentence_lengths)
    numerator = sum((i - x_mean) * (sentence_lengths[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _pos_distribution_variance(sentences: list[IRSentence]) -> float:
    """Compute mean of per-POS-tag sample variances of normalized histograms.

    Returns 0.0 if fewer than 2 sentences or no POS tags found.
    """
    if len(sentences) < 2:
        return 0.0

    all_pos_tags: set[str] = set()
    histograms: list[dict[str, float]] = []

    for s in sentences:
        counts: Counter[str] = Counter()
        for t in s.tokens:
            if t.morph is not None:
                counts[t.morph.pos] += 1
        total = sum(counts.values())
        if total == 0:
            histograms.append({})
            continue
        normalized = {pos: count / total for pos, count in counts.items()}
        histograms.append(normalized)
        all_pos_tags.update(normalized.keys())

    if not all_pos_tags:
        return 0.0

    variances: list[float] = []
    for pos in all_pos_tags:
        values = [h.get(pos, 0.0) for h in histograms]
        if len(values) >= 2:
            variances.append(variance(values))

    return mean(variances) if variances else 0.0


def _extract_style(
    ir: IntermediateRepresentation,
    cache: _FeatureCache,
) -> StyleFeatures:
    """Compute stylistic consistency features from the IR.

    Requirements 7.1-7.6. Always computed regardless of missing_layers.
    """
    return StyleFeatures(
        sentence_length_trend=_sentence_length_trend(cache.sentence_lengths),
        pos_distribution_variance=_pos_distribution_variance(cache.sentences),
    )


# ---------------------------------------------------------------------------
# 9.9  Top-level extract_features()
# ---------------------------------------------------------------------------

def extract_features(
    ir: IntermediateRepresentation,
    long_sentence_threshold: int = 20,
    freq_dict: dict[str, int] | None = None,
) -> Features:
    """Compute all feature categories from the IR, handling missing layers.

    When "stanza" is in missing_layers, morphological features are set to None.
    When "yap" is in missing_layers, syntactic features are set to None.
    Lexical, structural, discourse, and style features are always computed.

    Computation order: lexical+structural → morphological → syntactic → discourse → style.

    Requirements 7.1-10.3, 16.3.
    """
    missing = set(ir.missing_layers)
    cache = _build_cache(ir)

    # Step 1: Lexical + Structural (always computed)
    lexicon = _extract_lexical(ir, cache, freq_dict)
    structure = _extract_structural(ir, cache, long_sentence_threshold)

    # Step 2: Morphological features require Stanza data
    if "stanza" in missing:
        morphology = MorphFeatures(
            verb_ratio=None,
            binyan_distribution=None,
            prefix_density=None,
            suffix_pronoun_ratio=None,
            morphological_ambiguity=None,
            agreement_error_rate=None,
            binyan_entropy=None,
            construct_ratio=None,
        )
    else:
        morphology = _extract_morphological(ir, cache)

    # Step 3: Syntactic features require YAP dependency trees
    if "yap" in missing:
        syntax = SyntaxFeatures(
            avg_sentence_length=None,
            avg_tree_depth=None,
            max_tree_depth=None,
            avg_dependency_distance=None,
            clauses_per_sentence=None,
            subordinate_clause_ratio=None,
            right_branching_ratio=None,
            dependency_distance_variance=None,
            clause_type_entropy=None,
        )
    else:
        syntax = _extract_syntactic(ir, cache)

    # Step 4: Discourse (always computed)
    discourse = _extract_discourse(ir, cache)

    # Step 5: Style (always computed)
    style = _extract_style(ir, cache)

    return Features(
        morphology=morphology,
        syntax=syntax,
        lexicon=lexicon,
        structure=structure,
        discourse=discourse,
        style=style,
    )
