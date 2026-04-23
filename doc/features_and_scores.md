# Hebrew Linguistic Profiling Engine — Features, Scores & Analysis

## Features

The pipeline extracts 30+ linguistic features from Hebrew text, grouped into six categories.

### Morphology

| Feature | Description | Range |
|---|---|---|
| verb_ratio | Fraction of tokens tagged as VERB | [0, 1] |
| binyan_distribution | Histogram of Hebrew verb conjugation patterns (binyanim) | dict |
| prefix_density | Average number of prefixes per token (ו, ב, ל, etc.) | ≥ 0 |
| suffix_pronoun_ratio | Fraction of tokens with a pronominal suffix | [0, 1] |
| morphological_ambiguity | Mean number of morphological analyses per token | ≥ 1 |
| agreement_error_rate | Fraction of nsubj/amod pairs with gender or number mismatch | [0, 1] |
| binyan_entropy | Shannon entropy (natural log) over binyan distribution. Higher = more diverse verb usage | ≥ 0 |
| construct_ratio | Fraction of adjacent noun-noun pairs (smichut constructions) | ≥ 0 |

Requires: Stanza layer. Returns null when stanza is unavailable.

### Syntax

| Feature | Description | Range |
|---|---|---|
| avg_sentence_length | Mean token count per sentence | ≥ 0 |
| avg_tree_depth | Mean maximum depth of dependency trees | ≥ 0 |
| max_tree_depth | Maximum dependency tree depth across all sentences | ≥ 0 |
| avg_dependency_distance | Mean absolute distance between tokens and their heads | ≥ 0 |
| clauses_per_sentence | Count of subordinate dependency relations per sentence | ≥ 0 |
| subordinate_clause_ratio | Subordinate clause relations / all clause relations. **Note: nearly always 1.0 with YAP — kept for debugging only, not used in scoring** | [0, 1] |
| right_branching_ratio | Non-root dependencies where token follows head / total non-root | [0, 1] |
| dependency_distance_variance | Sample variance of dependency distances (non-root) | ≥ 0 |
| clause_type_entropy | Shannon entropy over all dependency relation types in the document. Higher = more diverse syntactic constructions | ≥ 0 |

Requires: YAP layer. Returns null when yap is unavailable.

### Lexicon

| Feature | Description | Range |
|---|---|---|
| type_token_ratio | Unique surface forms / total tokens | [0, 1] |
| hapax_ratio | Tokens appearing exactly once / total tokens | [0, 1] |
| avg_token_length | Mean character count per token | ≥ 0 |
| lemma_diversity | Unique lemmas / total tokens | [0, 1] |
| rare_word_ratio | Tokens with corpus frequency < 5 / total tokens. Null if no frequency dictionary provided | [0, 1] or null |
| content_word_ratio | NOUN/VERB/ADJ/ADV tokens / total tokens | [0, 1] |

Always computed. Falls back to surface forms when morphological data is unavailable.

### Structure

| Feature | Description | Range |
|---|---|---|
| sentence_length_variance | Sample variance of token counts across sentences | ≥ 0 |
| long_sentence_ratio | Sentences exceeding threshold (default 20) / total sentences | [0, 1] |
| punctuation_ratio | PUNCT-tagged tokens / total tokens | [0, 1] |
| short_sentence_ratio | Sentences with < 3 tokens / total sentences. **Note: typically 0.0 in Hebrew Wikipedia** | [0, 1] |
| missing_terminal_punctuation_ratio | Sentences not ending with . ! ? … / total sentences. **Note: nearly always 1.0 — YAP strips terminal punctuation** | [0, 1] |

Always computed.

### Discourse

| Feature | Description | Range |
|---|---|---|
| connective_ratio | Hebrew discourse connective tokens / total sentences. Can exceed 1.0 | ≥ 0 |
| sentence_overlap | Mean Jaccard similarity of lemma sets between adjacent sentences | [0, 1] |
| pronoun_to_noun_ratio | (PRON/total) / (NOUN/total + ε). Can exceed 1.0 | ≥ 0 |

Always computed regardless of missing layers.

### Style

| Feature | Description | Range |
|---|---|---|
| sentence_length_trend | Linear regression slope over sentence lengths by position. Positive = sentences getting longer | any float |
| pos_distribution_variance | Mean of per-POS-tag sample variances of normalized histograms across sentences. Higher = less consistent POS usage | ≥ 0 |

Always computed regardless of missing layers.

---

## Scores

Five composite scores are computed from the features above. Each is a weighted combination of normalized feature values. All scores are independent — no pair has |r| > 0.3 in the current corpus.

### Difficulty (0–1)

How hard the text is to read. Higher = more difficult.

Formula: weighted sum of normalized avg_sentence_length (0.30), avg_tree_depth (0.25), hapax_ratio (0.25), morphological_ambiguity (0.20). Absent features are excluded and remaining weights re-normalized.

### Style (typically −0.15 to 0.20)

Stylistic register and consistency. Higher = more stylistically marked.

Formula: suffix_pronoun_ratio (+0.25) − hapax_ratio (0.25) + |sentence_length_trend| (+0.20) − pos_distribution_variance (0.15) + pronoun_to_noun_ratio (+0.15). `sentence_length_variance` was removed — it belongs to fluency.

### Fluency (0–1)

Structural consistency and regularity. Higher = more fluent.

Formula: mean of punctuation_ratio, inverted sentence_length_variance, inverted pos_distribution_variance. Lower variance = higher fluency.

### Cohesion (0–1)

How well the text maintains coherence across sentences. Higher = more cohesive.

Formula: weighted combination — 0.4 × connective_ratio + 0.3 × sentence_overlap + 0.3 × (1 − pronoun_to_noun_ratio). The inverted pronoun_to_noun_ratio captures referential clarity: pronoun overload signals poor cohesion.

### Complexity (0–1)

Morpho-syntactic elaboration — diversity of grammatical constructions, independent of reading difficulty. Higher = more complex.

Formula: mean of binyan_entropy, agreement_error_rate, pos_distribution_variance, clause_type_entropy, and downweighted construct_ratio (×0.5). `subordinate_clause_ratio` and `dependency_distance_variance` were removed — the former is broken (always 1.0 in YAP), the latter overlapped with difficulty.

---

## Normalization Ranges

Features with configured normalization ranges are scaled to [0, 1] using min-max normalization with clamping: `norm(x) = clamp((x - min) / (max - min), 0, 1)`.

| Feature | Min | Max | Notes |
|---|---|---|---|
| avg_sentence_length | 10.0 | 40.0 | |
| avg_tree_depth | 4.0 | 15.0 | |
| hapax_ratio | 0.15 | 0.55 | |
| morphological_ambiguity | 4.0 | 10.0 | |
| suffix_pronoun_ratio | 0.05 | 0.50 | |
| sentence_length_variance | 0.0 | 400.0 | |
| sentence_length_trend | −1.5 | 1.5 | Style score uses absolute value |
| pos_distribution_variance | 0.0 | 0.008 | |
| pronoun_to_noun_ratio | 0.0 | 0.45 | |
| rare_word_ratio | 0.0 | 0.3 | |
| content_word_ratio | 0.1 | 0.8 | |
| connective_ratio | 0.0 | 1.2 | |
| sentence_overlap | 0.0 | 0.4 | |
| agreement_error_rate | 0.0 | 0.3 | |
| dependency_distance_variance | 0.0 | 27.0 | |
| clause_type_entropy | 2.0 | 3.0 | Tightened from (0, 3) — observed range 2.27–2.96 |

Ranges were calibrated against a 100-document Hebrew Wikipedia corpus. The `analyze_results.py` script checks whether ranges fit observed data and suggests adjustments when >20% of values are clamped.

---

## Correlation Analysis

The `analyze_results.py` script includes a score independence analysis that computes:

1. **Pairwise Pearson correlations** between all 5 scores — presented as a matrix. Pairs with |r| > 0.7 are flagged as potentially redundant (measuring the same thing).

2. **Top feature drivers per score** — for each score, lists the raw features most strongly correlated with it (|r| > 0.3), showing what actually drives each score in practice.

Interpretation guide:
- |r| < 0.3: independent — scores capture genuinely different text dimensions
- |r| 0.3–0.7: moderate overlap — some shared signal but still distinct
- |r| > 0.7: redundant — consider revising one of the score formulas

### Observed Correlation Matrix (100-document Hebrew Wikipedia corpus, after Phase 1+2 refactoring)

```
                difficulty    style   fluency  cohesion  complexity
  difficulty       1.000    0.216    -0.309    -0.067     0.202
  style            0.216    1.000    -0.077    -0.072    -0.265
  fluency         -0.309   -0.077     1.000    -0.056    -0.232
  cohesion        -0.067   -0.072    -0.056     1.000    -0.019
  complexity       0.202   -0.265    -0.232    -0.019     1.000
```

✅ No redundant pairs. All scores are independent (max |r| = 0.309).

### Top Feature Drivers Per Score

**Difficulty** — driven by sentence length and tree depth:
- avg_sentence_length (r=+0.82), avg_tree_depth (r=+0.77), long_sentence_ratio (r=+0.74), clauses_per_sentence (r=+0.59), clause_type_entropy (r=+0.55)

**Style** — driven by POS consistency and pronoun usage:
- pos_distribution_variance (r=−0.56), clauses_per_sentence (r=+0.53), avg_sentence_length (r=+0.51), pronoun_to_noun_ratio (r=+0.50)

**Fluency** — driven by structural consistency (inverted):
- sentence_length_variance (r=−0.82), dependency_distance_variance (r=−0.46), max_tree_depth (r=−0.45), pos_distribution_variance (r=−0.40)

**Cohesion** — driven by discourse connectives and referential clarity:
- connective_ratio (r=+0.68), pronoun_to_noun_ratio (r=−0.50), lemma_diversity (r=−0.45), hapax_ratio (r=−0.42)

**Complexity** — driven by morphological elaboration:
- binyan_entropy (r=+0.81), agreement_error_rate (r=+0.65), subordinate_clause_ratio (r=+0.62, raw feature only), sentence_overlap (r=−0.57)

---

## Analysis Layer — Probabilistic Issue Detection

The analysis layer sits downstream of the feature extraction pipeline. It takes the raw feature values and per-sentence metrics from a single pipeline run, compares them against corpus-derived statistical baselines, and produces a ranked list of linguistic issues — each localized to a specific sentence, sentence pair, or the full document.

### Design Principles

- **No hard thresholds.** Every feature value is converted to a continuous severity score in [0, 1] using a sigmoid function anchored to corpus statistics. There is no binary "good/bad" cutoff.
- **Corpus-anchored.** All scoring is relative to a reference corpus. A feature value is "problematic" only if it deviates significantly from the corpus mean. Different corpora produce different baselines.
- **Confidence-discounted.** Issues derived from sparse or statistically degenerate features are automatically downweighted before ranking, so the top results reflect data quality as well as severity.
- **Single pipeline run.** The analysis layer reads the IR and features from one pipeline execution. It does not re-run the pipeline per sentence.

### Soft Scoring

Every raw feature value is transformed into a severity score using a sigmoid of the z-score:

```
z = (value − corpus_mean) / corpus_std
soft_score = 1 / (1 + e^(−z))
```

This maps any feature value to [0, 1], where 0.5 means "exactly at the corpus mean". Values above the mean score > 0.5 (more problematic for "high is bad" features), values below score < 0.5.

For "low is bad" features (e.g., `binyan_entropy`, `content_word_ratio`), severity is `1 − soft_score`.

When `std = 0` (degenerate feature — all corpus values identical), `soft_score` returns 0.5 (neutral).

### Corpus Statistics

The reference baseline is a `FeatureStats` record per feature, computed from a corpus of pipeline output JSONs:

- `mean`, `std` — used by `soft_score` for z-score computation. Computed from values clipped to the [p5, p95] range to reduce the impact of extreme outliers in diverse corpora (e.g., web crawl data). This prevents a handful of anomalous documents from skewing the z-score baseline.
- `p10`–`p90` — percentiles computed from the original (unclipped) distribution
- `valid_count` — number of non-null values in the corpus
- `unstable` — true when `valid_count < 30` (insufficient data)
- `degenerate` — true when `std = 0` (no variance in the clipped distribution)

Statistics are computed from raw (un-normalized) feature values — the same values that `feature_extractor` produces. The min-max normalized values from the scorer are not used.

### Issue Model

Each detected issue is described by six fields:

| Field | Type | Description |
|---|---|---|
| type | string | Specific issue name (e.g., `sentence_complexity`, `weak_cohesion`) |
| group | string | One of: `morphology`, `syntax`, `lexicon`, `structure`, `discourse`, `style` |
| severity | float [0, 1] | Sigmoid-scored deviation from corpus baseline. Higher = more problematic |
| confidence | float [0, 1] | Severity discounted by data quality (see below) |
| span | tuple | Scope: `(i,)` sentence-level, `(i−1, i)` discourse pair, `(0, N)` document-level |
| evidence | dict | Raw feature values and ranking metadata |

### Confidence Calculation

```
confidence = min(1.0, severity × feature_availability × feature_stability)
```

Where:
- `feature_availability` = fraction of non-null contributing features (0 to 1)
- `feature_stability`:
  - `1.0` — all contributing features are stable and non-degenerate
  - `0.5` — at least one contributing feature is unstable (`valid_count < 30`)
  - `0.0` — at least one contributing feature is degenerate (`std = 0`)

This ensures that issues based on unreliable statistics are pushed to the bottom of the ranking.

### Issue Types

17 issue types across 6 groups. Each uses one or more raw features and corpus statistics.

#### Morphology (document-level, span `(0, N)`)

| Issue | Severity Formula | Input Features |
|---|---|---|
| `agreement_errors` | `soft_score(agreement_error_rate)` | agreement_error_rate |
| `morphological_ambiguity` | `soft_score(morphological_ambiguity)` | morphological_ambiguity |
| `low_morphological_diversity` | `1 − soft_score(binyan_entropy)` | binyan_entropy |

#### Syntax

| Issue | Severity Formula | Span | Input |
|---|---|---|---|
| `sentence_complexity` | `0.6·soft_score(token_count) + 0.4·soft_score(tree_depth)` | `(i,)` per sentence | SentenceMetrics.token_count, tree_depth |
| `dependency_spread` | `soft_score(dependency_distance_variance)` | `(0, N)` | dependency_distance_variance |
| `excessive_branching` | `soft_score(right_branching_ratio)` | `(0, N)` | right_branching_ratio |

`sentence_complexity` uses per-sentence data extracted directly from the IR (token count and dependency tree depth), not document-level averages. The corpus statistics for `avg_sentence_length` and `avg_tree_depth` serve as the reference baseline.

#### Lexicon (document-level, span `(0, N)`)

| Issue | Severity Formula | Input Features |
|---|---|---|
| `low_lexical_diversity` | `0.6·(1−soft_score(lemma_diversity)) + 0.4·(1−soft_score(type_token_ratio))` | lemma_diversity, type_token_ratio |
| `rare_word_overuse` | `soft_score(rare_word_ratio)` | rare_word_ratio |
| `low_content_density` | `1 − soft_score(content_word_ratio)` | content_word_ratio |

#### Structure (document-level, span `(0, N)`)

| Issue | Severity Formula | Input Features |
|---|---|---|
| `sentence_length_variability` | `soft_score(sentence_length_variance)` | sentence_length_variance |
| `punctuation_issues` | `0.5·(1−soft_score(punctuation_ratio)) + 0.5·soft_score(missing_terminal_punctuation_ratio)` | punctuation_ratio, missing_terminal_punctuation_ratio |
| `fragmentation` | `soft_score(short_sentence_ratio)` | short_sentence_ratio |

#### Discourse

| Issue | Severity Formula | Span | Input |
|---|---|---|---|
| `weak_cohesion` | `1 − soft_score(similarity)` | `(i−1, i)` per adjacent pair | cosine_similarity (embedding) or Jaccard (fallback) |
| `missing_connectives` | `1 − soft_score(connective_ratio)` | `(0, N)` | connective_ratio |
| `pronoun_ambiguity` | `soft_score(pronoun_to_noun_ratio)` | `(0, N)` | pronoun_to_noun_ratio |

`weak_cohesion` computes similarity between adjacent sentences. Two methods are available:

**Sentence embeddings (preferred):** Each sentence is encoded into a dense vector using a multilingual transformer model (`paraphrase-multilingual-mpnet-base-v2`). Cosine similarity between adjacent sentence vectors captures semantic relatedness — including pronoun coreference, synonymy, and general-to-specific relationships that surface-level metrics miss. The corpus baseline is `sentence_cosine_similarity` (mean ≈ 0.50, std ≈ 0.18 on a mixed Hebrew corpus).

**Jaccard similarity (fallback):** When embeddings are unavailable, similarity is computed as `|A ∩ B| / |A ∪ B|` over the lemma sets of adjacent sentences. This captures lexical overlap but cannot detect pronoun coreference or semantic paraphrase. The corpus baseline is `sentence_overlap` (mean ≈ 0.15, std ≈ 0.03 on Hebrew Wikipedia).

#### Style (document-level, span `(0, N)`)

| Issue | Severity Formula | Input Features |
|---|---|---|
| `structural_inconsistency` | `soft_score(pos_distribution_variance)` | pos_distribution_variance |
| `sentence_progression_drift` | `abs(soft_score(sentence_length_trend) − 0.5) × 2` | sentence_length_trend |

`sentence_progression_drift` uses a symmetric formula: both strong positive trends (sentences getting longer) and strong negative trends (sentences getting shorter) produce high severity. A flat trend (slope ≈ 0) produces severity ≈ 0.

### Ranking

Issues are ranked by a composite score that blends individual severity with the overall score of the issue's group:

```
group_score(g) = mean severity of all issues in group g
rank_score(issue) = 0.7 × severity + 0.3 × group_score
```

This boosts issues from broadly problematic groups. For example, if all syntax issues have high severity, each individual syntax issue gets an additional boost from the group score.

The top K issues (default 5) are returned, sorted by `rank_score` descending. No hard threshold is applied — all issues are candidates regardless of severity. If fewer than K issues are detected, all are returned.

Both `rank_score` and `group_score` are included in each issue's `evidence` dict for transparency.

---

## Known Limitations — YAP classifies most clause relations as subordinate. Kept as a raw feature for debugging but excluded from all score formulas.
- **missing_terminal_punctuation_ratio** is nearly always 1.0 — YAP strips terminal punctuation during tokenization.
- **short_sentence_ratio** is typically 0.0 — Hebrew Wikipedia sentences rarely have fewer than 3 tokens.
- **rare_word_ratio** requires a frequency dictionary passed via `--freq-dict`. Returns null otherwise.
- The frequency dictionary built by `split_corpus.py --build-freq-dict` counts raw surface tokens, not morphological lemmas. For more accurate rare word detection, a lemma-based dictionary is needed.
- **construct_ratio** is noisy — the adjacent noun-noun heuristic produces false positives in Hebrew. Downweighted in complexity (×0.5). Future improvement: refine using morphological definiteness markers.

- **subordinate_clause_ratio** is nearly always 1.0 — YAP classifies most clause relations as subordinate. Kept as a raw feature for debugging but excluded from all score formulas.
- **missing_terminal_punctuation_ratio** is nearly always 1.0 — YAP strips terminal punctuation during tokenization. The analysis layer checks for any punctuation token in the sentence (not just the last token) and accepts `:` as terminal punctuation for legal/structured text.
- **short_sentence_ratio** is typically 0.0 in Hebrew Wikipedia — the `fragmentation` issue will have confidence = 0 when the corpus baseline is degenerate (std = 0). Enriching the corpus with diverse text (e.g., HeDC4 web crawl) resolves this.
- **rare_word_ratio** requires a frequency dictionary passed via `--freq-dict`. Returns null otherwise.
- The frequency dictionary built by `split_corpus.py --build-freq-dict` counts raw surface tokens, not morphological lemmas. For more accurate rare word detection, a lemma-based dictionary is needed.
- **construct_ratio** is noisy — the adjacent noun-noun heuristic produces false positives in Hebrew. Downweighted in complexity (×0.5).
- **Jaccard-based cohesion** cannot detect pronoun coreference, synonymy, or general-to-specific relationships. Sentence embeddings (enabled with `--embed`) address this limitation.
- **Corpus baseline sensitivity** — all analysis layer scoring is relative to the reference corpus. A Wikipedia-only baseline produces skewed results for legal, news, or informal text. A diverse corpus (Wikipedia + HeDC4) is recommended.


---

## Layer 4 — Diagnosis Engine

The diagnosis engine aggregates patterns of issues and composite scores into 8 linguistically meaningful diagnoses. Each diagnosis rule combines one or more issue severities (and optionally a composite score) using a weighted mean, then applies a confidence-aware activation threshold. Only diagnoses whose severity exceeds the threshold are emitted.

### Input

- `issues: List[Issue]` — the existing issue list from the analysis layer (17 issue types).
- `scores: Dict[str, Optional[float]]` — the 5 composite scores (`difficulty`, `style`, `fluency`, `cohesion`, `complexity`). Values may be `None` when upstream layers are unavailable.

### Severity Computation

Each diagnosis rule computes severity as a **weighted mean** of contributing signals:

```
severity = sum(value_i × weight_i) / sum(weight_i)
```

For "direct" rules (fragmented_writing, punctuation_deficiency), severity equals the maximum severity of the matching issue type — no weighted combination is applied.

### Confidence Calculation

```
confidence = min(confidence_i for i in supporting_issues)
```

Where `supporting_issues` are all Issue objects that contributed to the diagnosis. When no supporting issues exist, confidence is 0.0. For direct rules, confidence equals the confidence of the highest-severity matching issue.

### Null Score Handling

When a composite score is `None`, the engine substitutes 0.0 and records the substitution in the evidence dict:

```python
evidence[f"{score_name}_missing"] = True
```

This is conservative — substituting 0.0 lowers the weighted mean, making activation less likely when data is missing.

### Activation Threshold

A diagnosis is emitted only when `severity > threshold`. The threshold is 0.6 for all diagnoses except `sentence_over_complexity`, which uses 0.65.

### Diagnosis Rules

| # | Diagnosis Type | Input Issues | Score Used | Weights | Threshold |
|---|---|---|---|---|---|
| 1 | `low_lexical_diversity` | `low_lexical_diversity`, `low_content_density` | — | 0.7, 0.3 | 0.6 |
| 2 | `pronoun_overuse` | `pronoun_ambiguity` | cohesion | 0.8, 0.2 | 0.6 |
| 3 | `low_cohesion` | `weak_cohesion`, `missing_connectives` | — | 0.6, 0.4 | 0.6 |
| 4 | `sentence_over_complexity` | `sentence_complexity` | difficulty | 0.7, 0.3 | 0.65 |
| 5 | `structural_inconsistency` | `structural_inconsistency` | fluency | 0.6, 0.4 | 0.6 |
| 6 | `low_morphological_richness` | `low_morphological_diversity` | complexity | 0.7, 0.3 | 0.6 |
| 7 | `fragmented_writing` | `fragmentation` | — | direct | 0.6 |
| 8 | `punctuation_deficiency` | `punctuation_issues` | — | direct | 0.6 |

**Weights column:** For weighted rules, the first weight applies to the max issue severity (or mean severity for `sentence_over_complexity`) and the second to the composite score (or second issue type). "direct" means severity equals the max severity of the matching issue type directly.

### Diagnosis Rule Details

1. **low_lexical_diversity** — `0.7 × max_severity("low_lexical_diversity") + 0.3 × max_severity("low_content_density")`. Combines vocabulary poverty with low content word density.

2. **pronoun_overuse** — `0.8 × max_severity("pronoun_ambiguity") + 0.2 × cohesion_score`. Pronoun ambiguity is the primary signal; low cohesion reinforces the diagnosis.

3. **low_cohesion** — `0.6 × max_severity("weak_cohesion") + 0.4 × max_severity("missing_connectives")`. Combines inter-sentence similarity weakness with missing discourse connectives.

4. **sentence_over_complexity** — `0.7 × mean_severity("sentence_complexity") + 0.3 × difficulty_score`. Uses mean severity (not max) so that a single complex sentence in an otherwise simple text does not inflate the diagnosis. The global difficulty score provides context. Uses a higher threshold (0.65) to reduce false positives.

5. **structural_inconsistency** — `0.6 × max_severity("structural_inconsistency") + 0.4 × fluency_score`. POS distribution variance is the primary signal; the fluency score captures broader structural regularity.

6. **low_morphological_richness** — `0.7 × max_severity("low_morphological_diversity") + 0.3 × complexity_score`. Low binyan entropy is the primary signal; the complexity score adds morpho-syntactic context.

7. **fragmented_writing** — `max_severity("fragmentation")` directly. Activates when short sentence ratio is high enough to produce a severity above 0.6.

8. **punctuation_deficiency** — `max_severity("punctuation_issues")` directly. Activates when punctuation problems are severe enough.

### Output

`run_diagnoses(issues, scores)` returns a `List[Diagnosis]` sorted by severity in descending order. Each `Diagnosis` contains:

| Field | Type | Description |
|---|---|---|
| `type` | `str` | One of the 8 diagnosis type strings |
| `confidence` | `float` | Min confidence of supporting issues [0, 1] |
| `severity` | `float` | Weighted severity from formula [0, 1] |
| `supporting_issues` | `List[str]` | Issue types that contributed |
| `supporting_spans` | `List[tuple]` | Spans from supporting issues |
| `evidence` | `dict` | Raw values used in computation (e.g., max/mean severities, score values, missing flags) |

---

## Layer 5 — Intervention Mapper

The intervention mapper takes the list of diagnoses from Layer 4 and maps each one to a pedagogical intervention. The mapping is static — each of the 8 diagnosis types maps to exactly one of 4 intervention types.

### Diagnosis-to-Intervention Mapping

| Diagnosis Type | Intervention Type |
|---|---|
| `low_lexical_diversity` | `vocabulary_expansion` |
| `low_morphological_richness` | `vocabulary_expansion` |
| `pronoun_overuse` | `pronoun_clarification` |
| `sentence_over_complexity` | `sentence_simplification` |
| `structural_inconsistency` | `sentence_simplification` |
| `fragmented_writing` | `sentence_simplification` |
| `low_cohesion` | `cohesion_improvement` |
| `punctuation_deficiency` | `cohesion_improvement` |

### Intervention Types

**vocabulary_expansion** — Targets lexical and morphological weakness. Actions include synonym substitution, varied word forms, and paraphrasing. Focus features: `lemma_diversity`, `type_token_ratio`, `content_word_ratio`, `binyan_entropy`.

**pronoun_clarification** — Targets referential ambiguity from pronoun overuse. Actions include replacing ambiguous pronouns with explicit noun phrases and reducing pronoun density. Focus features: `pronoun_to_noun_ratio`.

**sentence_simplification** — Targets overly complex, inconsistent, or fragmented sentence structures. Actions include breaking long sentences, reducing clause nesting, standardizing sentence length, and combining fragments. Focus features: `avg_sentence_length`, `avg_tree_depth`, `clauses_per_sentence`, `short_sentence_ratio`.

**cohesion_improvement** — Targets weak discourse connectivity and punctuation problems. Actions include adding discourse connectives, increasing lexical overlap, and correcting punctuation. Focus features: `connective_ratio`, `sentence_overlap`, `punctuation_ratio`.

### Intervention Fields

| Field | Type | Description |
|---|---|---|
| `type` | `str` | One of the 4 intervention type strings |
| `priority` | `float` | Equals the severity of the target diagnosis [0, 1] |
| `target_diagnosis` | `str` | The diagnosis type that triggered this intervention |
| `actions` | `List[str]` | Pedagogical actions (non-empty) |
| `exercises` | `List[str]` | Recommended exercises (non-empty) |
| `focus_features` | `List[str]` | Linguistic feature names to focus on |

### Output

`map_interventions(diagnoses)` returns a `List[Intervention]` sorted by priority in descending order. Diagnoses with types not present in the mapping table are silently skipped.

---

## Layer 6 — ML Distillation

Layer 6 is a multi-task transformer student model that learns to predict linguistic scores, issues, and diagnoses directly from raw Hebrew text, using the existing deterministic pipeline (Layers 1–5) as a teacher in a teacher–student distillation approach. The student replaces the full pipeline's heavy NLP passes (Stanza ~30s + YAP ~5s) with a single forward pass (~50ms GPU / ~200ms CPU).

### Prediction Heads

The student model uses a DictaBERT Hebrew encoder (`dicta-il/dictabert`) with the `[CLS]` token representation as the document-level input to three linear prediction heads, plus two sentence-level heads that operate on mean-pooled token embeddings:

| Head | Output Dimension | Activation | Description |
|------|-----------------|------------|-------------|
| Scores | 5 | sigmoid | Composite scores: difficulty, style, fluency, cohesion, complexity |
| Issues | 17 | sigmoid | Issue type severities (one per Issue_Type) |
| Diagnoses | 8 | sigmoid | Diagnosis type severities (one per Diagnosis_Type) |
| Sentence | 1 | sigmoid | Per-sentence complexity prediction (applied to each sentence embedding) |
| Pair | 1 | sigmoid | Per-adjacent-pair cohesion weakness prediction (applied to concatenated pair embeddings) |

All outputs are continuous values in [0, 1]. These are **soft labels** — the pipeline produces calibrated continuous severity values, and using them directly as training targets preserves the "how severe" signal rather than reducing to binary 0/1.

The document-level heads (Scores, Issues, Diagnoses) use the CLS token. The sentence-level heads use mean-pooled token embeddings per sentence boundary: for each sentence, the token embeddings within that sentence's span are averaged to produce a sentence representation. The Sentence head is applied independently to each sentence embedding. For the Pair head, adjacent sentence representations are concatenated into a `(2 × hidden_dim)` vector.

### Loss Functions

| Head | Loss Function | Description |
|------|--------------|-------------|
| Scores | MSE (Mean Squared Error) | Regression loss for continuous score values |
| Issues | BCE (Binary Cross-Entropy) | Soft multi-label loss with continuous severity targets |
| Diagnoses | BCE (Binary Cross-Entropy) | Soft multi-label loss with continuous severity targets |
| Sentence | BCE (Binary Cross-Entropy) | Per-sentence complexity severity (variable-length per document) |
| Pair | BCE (Binary Cross-Entropy) | Per-adjacent-pair cohesion weakness severity (variable-length per document) |

### Multi-Task Loss

The five head losses are combined into a single multi-task loss:

```
L = w1 × MSE(scores) + w2 × BCE(issues) + w3 × BCE(diagnoses) + w4 × BCE(sentence_complexity) + w5 × BCE(weak_cohesion)
```

Default weights: `w1 = 1.0`, `w2 = 1.5`, `w3 = 2.0`, `w4 = 1.5`, `w5 = 1.5`. The higher weight on diagnoses reflects their downstream importance for intervention derivation. The sentence-level terms (`w4`, `w5`) use BCE with variable-length targets — each batch item contributes a different number of sentences/pairs, and the loss is averaged across all items.

An alternative **uncertainty weighting** mode (Kendall et al. 2018) is available, where each head loss is scaled by a learned log-variance parameter:

```
L_i / (2 × exp(s_i)) + s_i / 2
```

### Intervention Derivation

Interventions are NOT predicted by the model. Instead, predicted diagnosis severities exceeding a threshold of 0.5 are converted to `Diagnosis` objects and passed through the existing `map_interventions()` function from `analysis/intervention_mapper.py`. This reuses the existing deterministic mapping (Layer 5) without duplication:

```python
active_diagnoses = [
    Diagnosis(type=t, severity=s, confidence=s)
    for t, s in predicted_diagnoses.items()
    if s > 0.5
]
interventions = map_interventions(active_diagnoses)
```

This ensures that the model's intervention output is always consistent with the pipeline's intervention logic.

### Sentence-Level Predictions

The model includes two sentence-level prediction heads that operate on sub-document representations derived from a single DictaBERT forward pass.

#### Architecture

Token embeddings from the encoder's last hidden state are mean-pooled per sentence boundary to produce sentence-level representations. Sentence boundaries are detected by mapping sentence character spans to token index spans using the tokenizer's offset mapping (see `ml/sentence_utils.py`).

- **Sentence head** — `Linear(hidden_dim, 1)` → sigmoid. Applied independently to each sentence embedding. Predicts per-sentence complexity severity in [0, 1].
- **Pair head** — `Linear(hidden_dim × 2, 1)` → sigmoid. Adjacent sentence embeddings are concatenated into a `(2 × hidden_dim)` vector. Predicts per-pair cohesion weakness severity in [0, 1].

The document-level heads (scores, issues, diagnoses) continue to use the CLS token and are unaffected by the sentence-level additions.

#### Training Data Format

Training records include two additional fields for sentence-level labels:

- `sentence_complexities`: list of float severity values, one per sentence. Each value is the `sentence_complexity` issue severity for that sentence from the pipeline's `detect_issues()` output. Sentences without a complexity issue get 0.0.
- `cohesion_pairs`: list of float severity values, one per adjacent sentence pair. Each value is the `weak_cohesion` issue severity for that pair. Pairs without a cohesion issue get 0.0.

Both lists have variable length depending on the document's sentence count. Old training records without these fields are backward compatible — sentence-level loss terms are skipped.

#### Loss Function

The 5-term multi-task loss:

```
L = w1 × MSE(scores) + w2 × BCE(issues) + w3 × BCE(diagnoses) + w4 × BCE(sentence_complexity) + w5 × BCE(weak_cohesion)
```

Default weights: `(1.0, 1.5, 2.0, 1.5, 1.5)`. The sentence-level BCE terms handle variable-length targets per batch item — each item's loss is computed independently and then averaged across the batch.

#### Inference Output

The `predict()` function returns two additional fields when sentence-level predictions are available:

- `sentence_complexity`: list of `{sentence: int, severity: float}` for sentences with severity > 0.3
- `weak_cohesion`: list of `{pair: [int, int], severity: float}` for adjacent pairs with severity > 0.3

The document-level `issues` dict is also updated with the max of per-sentence/per-pair predictions for `sentence_complexity` and `weak_cohesion`.
