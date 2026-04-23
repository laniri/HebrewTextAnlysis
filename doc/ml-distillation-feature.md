# Feature Description: ML Distillation Layer (Layer 6)

## Overview

Build a multi-task Hebrew NLP model that learns to predict linguistic scores, issues, diagnoses, and interventions directly from raw text — using the existing deterministic pipeline as a teacher. This is a classic teacher–student distillation approach: the rule-based pipeline generates structured training labels, and a transformer model learns to reproduce those labels from text alone, achieving 10–50x faster inference without requiring Stanza, YAP, or any external NLP services at runtime.

## Motivation

The current pipeline is accurate but slow:

```
Raw Text → Stanza (~30s) → YAP (~5s) → IR Builder → Feature Extractor → Scorer
         → Analysis Layer → Diagnosis Engine → Intervention Mapper → JSON
```

Each document requires two heavy NLP passes (Stanza morphological analysis, YAP syntactic parsing), corpus statistics lookup, and multiple rule layers. This is fine for batch processing but too slow for interactive use (e.g., a writing assistant giving real-time feedback).

A distilled transformer model replaces the entire pipeline with a single forward pass:

```
Raw Text → Transformer Encoder → [CLS] → Multi-task Heads → Scores + Issues + Diagnoses + Interventions
```

## What Already Exists (Teacher Pipeline)

### Pipeline Layers

| Layer | Module | Output | Speed |
|-------|--------|--------|-------|
| 1. Features | `hebrew_profiler/feature_extractor.py` | 30+ linguistic features (morphology, syntax, lexicon, structure, discourse, style) | ~35s per doc (Stanza + YAP) |
| 2. Scores | `hebrew_profiler/scorer.py` | 5 composite scores: difficulty, style, fluency, cohesion, complexity — each float in [0,1] or None | <1ms |
| 3. Issues | `analysis/issue_detector.py` | 17 issue types across 6 groups, each with severity [0,1], confidence [0,1], span, evidence | <100ms |
| 4. Diagnoses | `analysis/diagnosis_engine.py` | 8 diagnosis types with weighted severity formulas and activation thresholds | <1ms |
| 5. Interventions | `analysis/intervention_mapper.py` | 4 intervention types mapped from diagnoses, with actions, exercises, focus features | <1ms |

### Issue Types (17)

| Group | Issues |
|-------|--------|
| Morphology | `agreement_errors`, `morphological_ambiguity`, `low_morphological_diversity` |
| Syntax | `sentence_complexity`, `dependency_spread`, `excessive_branching` |
| Lexicon | `low_lexical_diversity`, `rare_word_overuse`, `low_content_density` |
| Structure | `sentence_length_variability`, `punctuation_issues`, `fragmentation` |
| Discourse | `weak_cohesion`, `missing_connectives`, `pronoun_ambiguity` |
| Style | `structural_inconsistency`, `sentence_progression_drift` |

### Diagnosis Types (8)

| Diagnosis | Formula | Threshold |
|-----------|---------|-----------|
| `low_lexical_diversity` | 0.7 × max_sev(low_lexical_diversity) + 0.3 × max_sev(low_content_density) | 0.6 |
| `pronoun_overuse` | 0.8 × max_sev(pronoun_ambiguity) + 0.2 × cohesion_score | 0.6 |
| `low_cohesion` | 0.6 × max_sev(weak_cohesion) + 0.4 × max_sev(missing_connectives) | 0.6 |
| `sentence_over_complexity` | 0.7 × mean_sev(sentence_complexity) + 0.3 × difficulty_score | 0.65 |
| `structural_inconsistency` | 0.6 × max_sev(structural_inconsistency) + 0.4 × fluency_score | 0.6 |
| `low_morphological_richness` | 0.7 × max_sev(low_morphological_diversity) + 0.3 × complexity_score | 0.6 |
| `fragmented_writing` | max_sev(fragmentation) | 0.6 |
| `punctuation_deficiency` | max_sev(punctuation_issues) | 0.6 |

### Intervention Types (4)

| Intervention | Triggered By |
|-------------|-------------|
| `vocabulary_expansion` | low_lexical_diversity, low_morphological_richness |
| `pronoun_clarification` | pronoun_overuse |
| `sentence_simplification` | sentence_over_complexity, structural_inconsistency, fragmented_writing |
| `cohesion_improvement` | low_cohesion, punctuation_deficiency |

### Existing Data Assets

- **Corpus directories**: `results_sample/`, `results_hedc4/` — contain pipeline output JSONs with full features and scores for each document
- **Merged statistics**: `feature_stats_merged.json` — corpus-derived baselines for the analysis layer
- **Batch infrastructure**: `run_pipeline.py batch` can process thousands of documents in parallel with `--build-stats`
- **Analysis CLI**: `run_pipeline.py single --analyze` runs the full pipeline including diagnosis and intervention layers

### Existing Output Format (per document)

```json
{
  "text": "...",
  "features": { "morphology": {...}, "syntax": {...}, ... },
  "scores": { "difficulty": 0.72, "style": 0.21, "fluency": 0.41, "cohesion": 0.33, "complexity": 0.59 },
  "issues": [
    { "type": "sentence_complexity", "severity": 0.99, "confidence": 0.99, "span": [4], ... },
    { "type": "weak_cohesion", "severity": 0.74, "confidence": 0.74, "span": [5, 6], ... }
  ],
  "diagnoses": [
    { "type": "sentence_over_complexity", "severity": 0.78, "confidence": 0.85, ... }
  ],
  "interventions": [
    { "type": "sentence_simplification", "priority": 0.78, "target_diagnosis": "sentence_over_complexity", ... }
  ]
}
```

## What to Build (Student Model)

### Architecture

```
                        ┌─────────────────────┐
                        │   Hebrew Encoder     │
                        │  (DictaBERT / 512)   │
                        └──────────┬──────────┘
                                   │
                              [CLS] token
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
              │  Scores   │ │  Issues   │ │ Diagnoses │
              │  Head     │ │  Head     │ │   Head    │
              │ (Linear→5)│ │(Linear→17)│ │(Linear→8) │
              └───────────┘ └───────────┘ └───────────┘
                regression   soft multi-   soft multi-
                  (MSE)      label (BCE)   label (BCE)
```

Note: The interventions head is omitted from the model. Interventions are deterministically derived from diagnoses via the existing `INTERVENTION_MAP` — no learning needed. At inference time, predicted diagnoses are passed through `map_interventions()` to produce interventions.

### Encoder Selection

**Primary candidate: `dicta-il/dictabert`** — trained on a larger and more diverse Hebrew corpus than AlephBERT, tends to outperform on downstream Hebrew tasks. Also consider `dicta-il/dictabert-large` if GPU memory allows.

**Fallback: `onlplab/alephbert-base`** — well-established Hebrew BERT, good baseline.

Benchmark both on a validation set before committing.

### Input Handling

- **Max sequence length**: 512 tokens (encoder limit)
- **Truncation strategy**: Truncate at 512 tokens for training. for infrence in production use sliding window . Most diagnostic signal is in the first few sentences. Document this as a known limitation.
- ** production**: Sliding window with aggregation, or hierarchical encoding (sentence-level → document-level). Replace truncation with sliding window + aggregation or a hierarchical encoder (sentence embeddings → document-level attention) to handle documents longer than 512 tokens.


### Output Heads

**Scores head (regression, 5 outputs)**:
- Predicts: difficulty, style, fluency, cohesion, complexity
- Each output in [0, 1] (apply sigmoid)
- Loss: MSE
- Target: pipeline scores dict values (substitute 0.0 for None)

**Issues head (soft multi-label, 17 outputs)**:
- Predicts: severity of each issue type
- Each output in [0, 1] (apply sigmoid)
- Loss: BCE with soft targets (severity values, not binary)
- Target: for each of the 17 issue types, the max severity across all issues of that type in the document. 0.0 if the issue type is absent.
- Note: This is a document-level prediction. Per-sentence localization (spans) is not predicted by the model — it's a known limitation for MVP.

**Diagnoses head (soft multi-label, 8 outputs)**:
- Predicts: severity of each diagnosis type
- Each output in [0, 1] (apply sigmoid)
- Loss: BCE with soft targets (severity values, not binary)
- Target: for each of the 8 diagnosis types, the severity if the diagnosis activated, 0.0 otherwise.

### Loss Function

```python
L = w1 * L_scores + w2 * L_issues + w3 * L_diagnoses
```

Use uncertainty-weighted multi-task loss (Kendall et al. 2018) to learn the weights automatically, or normalize each loss by its initial magnitude before applying fixed weights. Starting point for fixed weights:

```python
w1 = 1.0   # scores (5 regression targets)
w2 = 1.5   # issues (17 soft multi-label targets)
w3 = 2.0   # diagnoses (8 soft multi-label targets, most important)
```

### Soft Labels (Critical Design Choice)

Use continuous severity/priority values as labels instead of binary 0/1:

| Head | Label Source | Example |
|------|-------------|---------|
| Scores | `scores["difficulty"]` | 0.72 |
| Issues | `max_severity(issues, "sentence_complexity")` | 0.83 |
| Diagnoses | `diagnosis.severity` if activated, else 0.0 | 0.78 |

This preserves the calibrated continuous signals from the pipeline. The model learns not just "is there a problem" but "how severe is the problem."

## Dataset Construction

### Step 1: Export Script

Build a script/method (`export_training_data.py`) that converts existing pipeline output JSONs into training format:

```
Input:  results_sample/*.json + results_hedc4/*.json (existing batch output)
        + feature_stats_merged.json (corpus statistics)

Process: For each pipeline JSON:
         1. Extract raw text
         2. Extract features → run analysis layer (detect_issues + run_diagnoses + map_interventions)
         3. Flatten into training record

Output: training_data.jsonl
```

Each line in the JSONL:

```json
{
  "text": "...",
  "scores": {"difficulty": 0.72, "style": 0.21, "fluency": 0.41, "cohesion": 0.33, "complexity": 0.59},
  "issues": {"sentence_complexity": 0.99, "weak_cohesion": 0.74, "low_lexical_diversity": 0.84, ...},
  "diagnoses": {"sentence_over_complexity": 0.78, "low_cohesion": 0.0, ...}
}
```

Key: The export/method script does NOT re-run Stanza/YAP. It reads existing pipeline JSONs (which already contain features and scores) and runs only the fast analysis + diagnosis layers on top.

### Step 2: Corpus Scale

| Source | Documents | Status |
|--------|-----------|--------|
| `results_sample/` (Wikipedia) | ~1,000 | Already processed |
| `results_hedc4/` (HeDC4 web) | ~500–1,000 | Already processed |
| Additional HeDC4 | Up to 50K | Requires batch processing |
| Additional Wikipedia | Up to 20K | Requires batch processing |

**Minimum viable**: 5K–10K documents (achievable by expanding HeDC4 download)
**Target**: 50K+ for production quality

### Step 3: Data Quality Considerations

- **Label noise**: The pipeline labels are deterministic but imperfect (as observed in our testing — sentence_over_complexity was over-triggering before the mean-severity fix). The model will learn these imperfections. Iterate: fix pipeline rules → re-export → retrain.
- **Class imbalance**: Most documents will have 0–2 active diagnoses out of 8. Use focal loss or class-weighted BCE to handle this.
- **Score None values**: When a pipeline score is None (missing Stanza/YAP), substitute 0.0 in the training label and add a binary mask feature indicating which scores are valid. Alternatively, exclude documents with None scores from training.

### Data Augmentation (Optional, High Value)

Since the pipeline is deterministic, augmentation creates new training signal:

| Augmentation | Effect on Labels |
|-------------|-----------------|
| Sentence shuffling | Changes cohesion labels, preserves complexity |
| Sentence deletion | Changes fragmentation labels |
| Synonym substitution | Changes lexical diversity labels |
| Sentence merging | Changes complexity and fragmentation labels |

## Training

### Baseline Training Recipe

```
Encoder: dicta-il/dictabert (or alephbert-base)
Dataset: 5K–10K documents
Batch size: 16
Learning rate: 2e-5 (encoder), 1e-3 (heads)
Epochs: 3–5
Optimizer: AdamW
Scheduler: Linear warmup (10% of steps) + linear decay
Max sequence length: 512
```

### Evaluation

**Against teacher (automated)**:
- RMSE for scores (5 values)
- F1 for issues (threshold at 0.5 on predicted severity)
- F1 for diagnoses (threshold at 0.5 on predicted severity)
- Rank correlation (Spearman) for severity ordering

**Human evaluation (small set, 50–100 documents)**:
- Are the predicted diagnoses correct?
- Are the derived interventions useful?
- Where does the model disagree with the pipeline, and who is right?

## Inference

### Fast Path (Model Only)

```python
text → tokenize → encoder → heads → {scores, issues, diagnoses}
diagnoses → map_interventions() → interventions
```

Latency: ~50ms on GPU, ~200ms on CPU. No Stanza, no YAP, no corpus statistics needed.

### Hybrid Mode that will be used initialy to gain confidence in the model

```python
model_output = model.predict(text)
confidence = mean(max(model_output["diagnoses"]))

if confidence > threshold:
    return model_output  # fast path
else:
    return pipeline.run(text)  # fallback to full pipeline
```

Confidence estimation options:
- Mean of max probabilities across heads
- Entropy-based: low entropy = high confidence
- Calibrated threshold learned on validation set

### Intervention Derivation

Interventions are NOT predicted by the model. They are derived deterministically:

```python
# At inference time:
predicted_diagnoses = model.predict(text)["diagnoses"]
# Convert soft predictions to Diagnosis objects (threshold at 0.5)
active_diagnoses = [Diagnosis(type=t, severity=s, ...) for t, s in predicted_diagnoses.items() if s > 0.5]
# Use existing mapper
interventions = map_interventions(active_diagnoses)
```

This reuses the existing `INTERVENTION_MAP` and `map_interventions()` from `analysis/intervention_mapper.py` — no duplication of intervention logic.

## Iteration Loop

```
1. Run pipeline on corpus → generate labeled dataset
2. Train model on dataset
3. Run model on new (unlabeled) corpus
4. Compare model predictions with pipeline on a sample
5. Identify disagreement cases
6. Add disagreement cases to training set (with pipeline labels)
7. Fix pipeline rules if model reveals systematic errors
8. Re-export dataset with fixed rules
9. Retrain model
10. Repeat
```

## Future Enhancements (Not in MVP)

### Sentence-Level Predictions
Add a sentence-level attention layer or token-level head to predict per-sentence issue spans (sentence_complexity, weak_cohesion). This would restore the localization that the document-level CLS approach loses.

### Contrastive Learning
Add a contrastive loss that pulls similar texts (same diagnosis profile) together and pushes different texts apart in embedding space. Improves style detection and cohesion understanding.

### Direct Diagnosis Prediction
Skip the issues layer entirely — train the model to predict diagnoses directly from text without the intermediate issue predictions. This simplifies the architecture but loses interpretability.

### Genre-Aware Predictions
Add a genre classification head (news, legal, blog, academic, informal) and condition the diagnosis thresholds on genre. This addresses the over-triggering of sentence_over_complexity on professional writing.

#
## Implementation Plan (High Level)

### Phase 1: Data Export
- Build `export_training_data.py` script
- Convert existing pipeline JSONs to training JSONL
- Validate label distributions and class balance

### Phase 2: Model Training
- Implement `LinguisticModel` with encoder + 3 heads
- Implement multi-task loss with uncertainty weighting
- Train baseline on 5K–10K documents
- Evaluate against teacher pipeline

### Phase 3: Inference Integration
- Build inference wrapper with `map_interventions()` for intervention derivation
- Implement hybrid mode with confidence-based fallback
- Add CLI support (`run_pipeline.py single --ml-model <path>`)

### Phase 4: Iteration
- Run model on new corpus, compare with pipeline
- Mine disagreement cases
- Retrain with expanded dataset
- Tune confidence thresholds

## Dependencies (New)

| Package | Purpose | Required |
|---------|---------|----------|
| `torch` | Model training and inference | Yes |
| `transformers` | DictaBERT / AlephBERT encoder | Yes |
| `datasets` | HuggingFace dataset loading | Optional (for corpus expansion) |
| `accelerate` | Multi-GPU training | Optional |
| `wandb` or `tensorboard` | Training monitoring | Optional |

Note: `torch` and `sentence-transformers` (which depends on `torch`) are already in the project's dependency tree for the embedding-based cohesion detection.
