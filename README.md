# Hebrew Linguistic Profiling Engine

A deterministic, multi-stage NLP pipeline that ingests raw Hebrew text and produces structured JSON output containing multi-layer linguistic features, heuristic scores, and probabilistic linguistic diagnostics.

Designed for corpus-scale processing (5,000–20,000 documents/day) with batch mode, parallel execution, JSONL export for ML training, and resilient error handling that preserves partial results.

## Pipeline Architecture

```
Raw Text → Normalizer → Tokenizer → Stanza Adapter ─┐
                                  → YAP Adapter    ──┤
                                                      ├→ IR Builder → Feature Extractor → Scorer → JSON
                                                      │
                                                      └→ Analysis Layer → Issues → Diagnoses → Interventions → JSON
```

Stanza (morphological analysis) and YAP (syntactic parsing) run independently and their results are merged in the IR Builder. If either fails, the pipeline degrades gracefully — available features are still computed and missing ones are set to `null`.

The **Analysis Layer** runs downstream of the main pipeline. It uses corpus-derived statistics to convert raw feature values into soft (sigmoid-based) severity scores, detects 17 linguistic issue types across 6 groups, and returns the top-5 ranked issues as JSON. The **Diagnosis Engine** (Layer 4) aggregates patterns of issues and composite scores into 8 linguistically meaningful diagnoses using weighted severity formulas and confidence-aware activation thresholds. The **Intervention Mapper** (Layer 5) maps each diagnosis to one of 4 pedagogical intervention types with actions, exercises, and focus features.

## Features Extracted

| Category | Features |
|----------|----------|
| Morphology | verb_ratio, binyan_distribution, prefix_density, suffix_pronoun_ratio, morphological_ambiguity, agreement_error_rate, binyan_entropy, construct_ratio |
| Syntax | avg_sentence_length, avg_tree_depth, max_tree_depth, avg_dependency_distance, clauses_per_sentence, subordinate_clause_ratio, right_branching_ratio, dependency_distance_variance |
| Lexicon | type_token_ratio, hapax_ratio, avg_token_length, lemma_diversity, rare_word_ratio, content_word_ratio |
| Structure | sentence_length_variance, long_sentence_ratio, punctuation_ratio, short_sentence_ratio, missing_terminal_punctuation_ratio |
| Discourse | connective_ratio, sentence_overlap, pronoun_to_noun_ratio |
| Style | sentence_length_trend, pos_distribution_variance |

Heuristic scores are computed from these features:
- **Difficulty** (0.0–1.0): weighted combination of sentence length, tree depth, hapax ratio, and morphological ambiguity
- **Style**: weighted combination of suffix pronoun ratio, hapax ratio (negative), sentence length trend, POS distribution variance (negative), and pronoun-to-noun ratio — sentence length variance was moved to fluency
- **Fluency**: optional score derived from structural fluency features
- **Cohesion**: optional score derived from discourse cohesion features
- **Complexity**: optional score derived from morpho-syntactic elaboration features (subordination, dependency distance, verb pattern diversity, agreement errors, branching direction)

## Prerequisites

- Python 3.12+
- [YAP](https://github.com/OnlpLab/yap) running as an API server (for syntactic parsing)
- Stanza Hebrew language model (for morphological analysis)
- `sentence-transformers` *(optional)* — enables embedding-based cohesion detection in the analysis layer; falls back to Jaccard similarity when not installed
- `datasets` *(optional)* — required only for `download_hedc4.py` to stream the HeDC4 corpus from HuggingFace

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the Stanza Hebrew model
python -c "import stanza; stanza.download('he')"

# Start the YAP API server (in a separate terminal)
./yapproj/src/yap/yap api
```

## Usage

### Single Document

```bash
# Pipeline output only
python run_pipeline.py single --input sample.txt --output result.json --pretty

# Pipeline + analysis layer (requires pre-built feature_stats.json)
python run_pipeline.py single --input sample.txt --output result.json --pretty \
    --analyze --stats-cache results/feature_stats.json

# With frequency dictionary and semantic cohesion
python run_pipeline.py single --input sample.txt --output result.json --pretty \
    --freq-dict freq_dict.json \
    --analyze --stats-cache results/feature_stats.json --embed --top-k 10

# Custom YAP URL
python run_pipeline.py single --input sample.txt --yap-url http://localhost:9000/yap/heb/joint
```

### Corpus Preparation

The pipeline supports multiple corpus sources. Documents are stored as individual `.txt` files — one per document — which `run_pipeline.py batch` processes into result JSONs.

**Wikipedia corpus** — split from `wikipedia.raw` (one sentence per line):

```bash
# Split entire file, 20 sentences per document
python split_corpus.py wikipedia.raw corpus/

# First 1000 documents only (for testing)
python split_corpus.py wikipedia.raw corpus_sample/ --max-docs 1000

# Split corpus AND build a frequency dictionary from it
python split_corpus.py wikipedia.raw corpus/ --build-freq-dict freq_dict.json

# Build frequency dictionary only (no splitting)
python split_corpus.py wikipedia.raw --build-freq-dict freq_dict.json --no-split
```

**HeDC4 web corpus** — diverse Hebrew text from Common Crawl (news, legal, commercial, blog, forum, government). Requires `pip install datasets`. Quality filters ensure downloaded documents are meaningful prose suitable for linguistic analysis: minimum/maximum length, minimum sentence count, minimum Hebrew character ratio, maximum sentence repetition, and optional URL-heavy document rejection.

```bash
# Download 500 documents with default quality filters
python download_hedc4.py --output corpus_hedc4/

# Download 5000 documents, stricter filters
python download_hedc4.py --output corpus_hedc4_5k/ --max-docs 5000 \
    --min-length 300 --min-sentences 5 --min-hebrew-ratio 0.6

# Reproducible sample with seed
python download_hedc4.py --output corpus_hedc4/ --max-docs 500 --seed 42

# Skip first N documents (useful for creating non-overlapping samples)
python download_hedc4.py --output corpus_hedc4/ --max-docs 500 --skip 1000

# Disable URL filtering (keep URL-heavy documents)
python download_hedc4.py --output corpus_hedc4/ --allow-url-heavy
```

**Merging multiple corpora** — combine result directories into a single statistics baseline:

```bash
# Process each corpus through the pipeline
python run_pipeline.py batch --input corpus_sample/ --output results_sample/ --workers 4 --freq-dict freq_dict.json
python run_pipeline.py batch --input corpus_hedc4/ --output results_hedc4/ --workers 4 --freq-dict freq_dict.json

# Merge statistics from both (with embedding stats)
python merge_stats.py --results-dirs results_sample/ results_hedc4/ \
    --output feature_stats_merged.json --embed

# Use merged stats for analysis
python run_pipeline.py single --input sample.txt --pretty \
    --analyze --stats-cache feature_stats_merged.json --embed --top-k 10 \
    --output analysis_result.json
```

A diverse corpus baseline improves analysis accuracy — Wikipedia alone produces skewed baselines for features like `missing_terminal_punctuation_ratio` and `short_sentence_ratio` that behave differently in legal, news, and informal text.

The frequency dictionary is a JSON file mapping tokens to corpus occurrence counts. Pass it to the pipeline via `--freq-dict` to enable `rare_word_ratio` computation. Note that `--build-freq-dict` and `--freq-dict` are mutually exclusive.

### Batch Processing

```bash
# Process all .txt files in a directory
python run_pipeline.py batch --input corpus/ --output results/

# With parallel workers and JSONL export
python run_pipeline.py batch --input corpus/ --output results/ --workers 8 --jsonl dataset.jsonl

# Pretty-print individual JSON files
python run_pipeline.py batch --input corpus/ --output results/ --pretty

# Build analysis layer corpus statistics immediately after batch processing
python run_pipeline.py batch --input corpus/ --output results/ --build-stats

# Also compute sentence embedding statistics (requires sentence-transformers)
python run_pipeline.py batch --input corpus/ --output results/ --build-stats --embed
```

`--build-stats` computes `feature_stats.json` from the batch output and saves it to the output directory. This is the recommended way to prepare the corpus baseline before running `run_analysis.py` — it avoids a separate stats-computation step. Pass `--embed` alongside `--build-stats` to also compute the `sentence_cosine_similarity` baseline needed for `run_analysis.py --embed`.

Batch processing automatically skips input files that already have a corresponding JSON result in the output directory. This makes it safe to re-run after interruptions or when new files are added to the corpus — only unprocessed documents are run through the pipeline.

### Analyzing Results

After batch processing, inspect feature distributions, outliers, and normalization range fitness:

```bash
# Analyze the default results_sample/ directory
python analyze_results.py

# Analyze a custom results directory
python analyze_results.py results/
```

The report includes:
- Per-feature summary statistics (min, max, mean, median, stdev, IQR)
- Outlier detection using 1.5×IQR fences
- Normalization range analysis — flags when >20% of values are clamped and suggests tighter ranges
- Score distribution with bucket histogram for difficulty, style, fluency, cohesion, and complexity

### Health Check

Quick sanity check on batch results — reports how many documents have complete features vs missing layers (Stanza/YAP failures), and identifies the first document where each layer started failing:

```bash
# Single directory
python check_results.py results_hedc4/

# Multiple directories
python check_results.py results_sample/ results_hedc4/
```

### CLI Options

| Flag | Mode | Description | Default |
|------|------|-------------|---------|
| `--input` | both | Input file (single) or directory (batch) | required |
| `--output` | both | Output file (single) or directory (batch) | stdout (single), required (batch) |
| `--pretty` | both | Pretty-print JSON with 2-space indent | false |
| `--yap-url` | both | YAP API endpoint URL | `http://localhost:8000/yap/heb/joint` |
| `--freq-dict` | both | Path to JSON frequency dictionary for `rare_word_ratio` computation | none |
| `--analyze` | single | Run the analysis layer after the pipeline and include ranked issue diagnostics, diagnoses, and interventions in the output | false |
| `--stats-cache` | single | Path to `feature_stats.json` required by `--analyze` (produced by `--build-stats`) | none |
| `--top-k` | single | Number of top issues to return when `--analyze` is set | 5 |
| `--embed` | both | single: use sentence embeddings for cohesion detection with `--analyze`; batch: compute embedding stats with `--build-stats` (both require `sentence-transformers`) | false |
| `--embed-model` | both | Sentence-transformers model name for `--embed` | `paraphrase-multilingual-mpnet-base-v2` |
| `--workers` | batch | Number of parallel workers | 4 |
| `--jsonl` | batch | Path for JSONL export file | none |
| `--build-stats` | batch | Compute analysis layer corpus statistics after batch and save to `feature_stats.json` in the output directory | false |
| `--strict` | batch | Abort batch processing immediately if YAP is unresponsive (forces sequential processing; ignores `--workers`) | false |

## Output Format

### JSON (single document)

```json
{
  "text": "הילד הלך לבית הספר",
  "features": {
    "morphology": {
      "verb_ratio": 0.25,
      "binyan_distribution": {"PAAL": 1},
      "prefix_density": 0.15,
      "suffix_pronoun_ratio": 0.05,
      "morphological_ambiguity": 3.2
    },
    "syntax": {
      "avg_sentence_length": 12.5,
      "avg_tree_depth": 4.0,
      "max_tree_depth": 6,
      "avg_dependency_distance": 2.3,
      "clauses_per_sentence": 1.2
    },
    "lexicon": {
      "type_token_ratio": 0.78,
      "hapax_ratio": 0.45,
      "avg_token_length": 4.1,
      "lemma_diversity": 0.72
    },
    "structure": {
      "sentence_length_variance": 15.3,
      "long_sentence_ratio": 0.2
    }
  },
  "scores": {
    "difficulty": 0.62,
    "style": 0.45,
    "fluency": null,
    "cohesion": null,
    "complexity": null
  },
  "sentence_metrics": [
    {"index": 0, "token_count": 5, "tree_depth": 3, "lemmas": ["ילד", "הלך", "בית", "ספר"]}
  ]
}
```

The `sentence_metrics` array contains per-sentence data extracted from the IR (real Stanza morphological analysis + YAP dependency parse): token count, dependency tree depth, and lemma list. It is present when the pipeline successfully builds an IR and omitted otherwise. The ML export layer (`ml/export.py`) uses this field to construct `SentenceMetrics` for accurate per-sentence training label generation, falling back to synthetic approximation for older pipeline JSONs that lack it.

When an upstream module fails, affected features are `null` and available features are still computed.

### JSONL (batch export)

Each line is a JSON object:

```json
{"raw_text": "...", "normalized_text": "...", "features": {...}, "scores": {...}}
```

All output uses UTF-8 encoding with `ensure_ascii=False`, preserving Hebrew characters in their native form.

## Programmatic Usage

```python
from hebrew_profiler.models import PipelineConfig
from hebrew_profiler.pipeline import process_document, pipeline_output_to_json

config = PipelineConfig(
    yap_url="http://localhost:8000/yap/heb/joint",
    pretty_output=True,
)

output = process_document("הילד הלך לבית הספר", config)
print(pipeline_output_to_json(output, pretty=True))
```

## Analysis Layer

The analysis layer converts raw pipeline outputs into ranked linguistic diagnostics. It runs the pipeline once, extracts per-sentence metrics from the IR, and uses corpus-derived statistics to score every feature with a soft (sigmoid-based) severity — no hard thresholds.

### Corpus Statistics

Before running analysis, compute statistics from the corpus sample:

```python
import json
from pathlib import Path
from analysis.statistics import flatten_corpus_json, compute_feature_stats, save_stats, load_stats

# Load corpus JSON files and compute stats
corpus_jsons = [json.loads(p.read_text()) for p in Path("results_sample").glob("doc_*.json")]
feature_dicts = [flatten_corpus_json(doc) for doc in corpus_jsons]
feature_stats = compute_feature_stats(feature_dicts)
save_stats(feature_stats)  # writes feature_stats.json

# On subsequent runs, load from disk
feature_stats = load_stats()
```

### Analysis Layer CLI

```bash
# Basic usage — compute stats from results_sample/, analyse input.txt
python run_analysis.py --results-dir results_sample/ --text input.txt

# With a word-frequency dictionary (enables rare_word_ratio)
python run_analysis.py --results-dir results_sample/ --text input.txt --freq-dict freq_dict.json

# Write to file, pretty-printed, return top 10 issues
python run_analysis.py --results-dir results_sample/ --text input.txt \
    --output analysis.json --pretty --top-k 10

# Reuse a pre-computed stats cache (skips corpus loading on subsequent runs)
python run_analysis.py --results-dir results_sample/ --text input.txt \
    --stats-cache feature_stats.json

# Semantic cohesion via sentence embeddings (requires sentence-transformers)
# First run: downloads model (~420 MB) + computes corpus embedding stats (~1-2 min)
python run_analysis.py --results-dir results_sample/ --text input.txt \
    --embed --output analysis.json

# Subsequent runs: loads cached stats, only embeds the input document (~1 sec)
python run_analysis.py --results-dir results_sample/ --text input.txt --embed
```

On the first run, corpus statistics are computed from `--results-dir` and saved to `feature_stats.json`. When `--embed` is first used, corpus embedding statistics (`sentence_cosine_similarity`) are computed and appended to the same cache automatically.

| Flag | Description | Default |
|------|-------------|---------|
| `--results-dir` / `-r` | Directory of profiler result JSON files | required |
| `--text` / `-t` | Input Hebrew text file | required |
| `--freq-dict` / `-f` | Word-frequency dictionary JSON | none |
| `--output` / `-o` | Output file (default: stdout) | stdout |
| `--pretty` | Pretty-print output JSON (always enabled when writing to a file) | false |
| `--stats-cache` | Path to cached `feature_stats.json` | `<results-dir>/feature_stats.json` |
| `--top-k` / `-k` | Number of top issues to return | 5 |
| `--yap-url` | YAP API endpoint | `http://localhost:8000/yap/heb/joint` |
| `--embed` | Use sentence embeddings for cohesion detection | false |
| `--embed-model` | Sentence-transformers model name | `paraphrase-multilingual-mpnet-base-v2` |

### Running Analysis (programmatic)

```python
from analysis import detect_issues, rank_issues
from analysis.statistics import load_stats, flatten_corpus_json
from analysis.serialization import serialize_issues
import json

# Load pre-computed corpus statistics
feature_stats = load_stats()

# Use raw features from a pipeline output JSON
with open("results_sample/doc_0000001.json") as f:
    doc = json.load(f)

raw_features = flatten_corpus_json(doc)

# Detect issues (pass sentence_metrics from IR for sentence-level issues)
issues = detect_issues(raw_features, sentence_metrics=[], feature_stats=feature_stats)

# Rank and serialize top-5
ranked = rank_issues(issues, k=5)
print(serialize_issues(ranked))
```

### Analysis Output Format

```json
{
  "issues": [
    {
      "type": "sentence_complexity",
      "group": "syntax",
      "severity": 0.999,
      "confidence": 0.999,
      "span": [4],
      "sentence": "בין היתר, ההאקרים פרסמו תצלומים של דרכונו ודרכונה של אשתו...",
      "evidence": {
        "token_count": 53.0,
        "tree_depth": 18.0,
        "rank_score": 0.899,
        "group_score": 0.668
      }
    },
    {
      "type": "weak_cohesion",
      "group": "discourse",
      "severity": 0.74,
      "confidence": 0.74,
      "span": [5, 6],
      "sentence": "כמו כן, פורסמו תמונות... / אירועי hack-and-leak הם סימן היכר...",
      "evidence": {
        "cosine_similarity": 0.31,
        "rank_score": 0.694,
        "group_score": 0.581
      }
    },
    {
      "type": "sentence_progression_drift",
      "group": "style",
      "severity": 0.986,
      "confidence": 0.986,
      "span": [0, 9],
      "sentence": null,
      "evidence": {
        "sentence_length_trend": 2.65,
        "rank_score": 0.882,
        "group_score": 0.640
      }
    }
  ],
  "scores": {
    "difficulty": 0.749,
    "style": 0.208,
    "fluency": 0.410,
    "cohesion": 0.327,
    "complexity": 0.594
  },
  "cohesion_method": "sentence_embeddings"
}
```

Each issue includes:
- `type` — specific issue name (e.g. `agreement_errors`, `sentence_complexity`)
- `group` — one of `morphology`, `syntax`, `lexicon`, `structure`, `discourse`, `style`
- `severity` — sigmoid-scored value in [0.0, 1.0]; higher = more problematic
- `confidence` — severity discounted by feature availability and corpus stability
- `span` — `[0, N]` document-level, `[i]` sentence-level, `[i-1, i]` discourse pair
- `sentence` — the sentence text for sentence-level issues; both sentences joined with ` / ` for discourse pairs; `null` for document-level issues
- `evidence` — raw feature values used to compute the score, plus `rank_score` and `group_score`; `weak_cohesion` uses `cosine_similarity` (embedding mode) or `jaccard` (fallback)

Top-level fields:
- `scores` — document-level heuristic scores (difficulty, style, fluency, cohesion, complexity)
- `cohesion_method` — `"sentence_embeddings"` or `"jaccard"`

### Full Analysis with Diagnoses and Interventions

When `--analyze` is set, the output includes diagnoses (Layer 4) and interventions (Layer 5) alongside the existing issues and scores. The diagnosis engine aggregates issue patterns and composite scores into higher-level diagnoses, and the intervention mapper produces pedagogical recommendations for each diagnosis.

```bash
# Run full analysis pipeline including diagnoses and interventions
python run_pipeline.py single --input sample.txt --output result.json --pretty \
    --analyze --stats-cache feature_stats_merged.json --embed --top-k 10
```

The JSON output extends the analysis format with `"diagnoses"` and `"interventions"` keys:

```json
{
  "issues": [ ... ],
  "scores": {
    "difficulty": 0.749,
    "style": 0.208,
    "fluency": 0.410,
    "cohesion": 0.327,
    "complexity": 0.594
  },
  "cohesion_method": "sentence_embeddings",
  "diagnoses": [
    {
      "type": "sentence_over_complexity",
      "confidence": 0.85,
      "severity": 0.78,
      "supporting_issues": ["sentence_complexity"],
      "supporting_spans": [[4], [7]],
      "evidence": {"mean_sentence_complexity_severity": 0.82, "difficulty_score": 0.71}
    },
    {
      "type": "low_cohesion",
      "confidence": 0.74,
      "severity": 0.68,
      "supporting_issues": ["weak_cohesion", "missing_connectives"],
      "supporting_spans": [[5, 6], [0, 9]],
      "evidence": {"max_weak_cohesion_severity": 0.74, "max_missing_connectives_severity": 0.59}
    }
  ],
  "interventions": [
    {
      "type": "sentence_simplification",
      "priority": 0.78,
      "target_diagnosis": "sentence_over_complexity",
      "actions": ["Break long sentences into shorter units", "Reduce subordinate clause nesting"],
      "exercises": ["Rewrite sentences exceeding 30 tokens", "Identify and extract embedded clauses"],
      "focus_features": ["avg_sentence_length", "avg_tree_depth", "clauses_per_sentence"]
    },
    {
      "type": "cohesion_improvement",
      "priority": 0.68,
      "target_diagnosis": "low_cohesion",
      "actions": ["Add discourse connectives between paragraphs", "Improve sentence-to-sentence overlap"],
      "exercises": ["Insert appropriate connectives", "Rewrite passages to improve lexical overlap"],
      "focus_features": ["connective_ratio", "sentence_overlap", "punctuation_ratio"]
    }
  ]
}
```

Diagnoses are sorted by severity descending, interventions by priority descending. Each intervention's `target_diagnosis` references a diagnosis in the list. When no diagnoses activate (all severities below threshold), both lists are empty.

### Issue Types

| Group | Issue Type | Severity Formula |
|-------|-----------|-----------------|
| morphology | `agreement_errors` | `soft_score(agreement_error_rate)` |
| morphology | `morphological_ambiguity` | `soft_score(morphological_ambiguity)` |
| morphology | `low_morphological_diversity` | `1 - soft_score(binyan_entropy)` |
| syntax | `sentence_complexity` | `0.6·soft_score(token_count) + 0.4·soft_score(tree_depth)` |
| syntax | `dependency_spread` | `soft_score(dependency_distance_variance)` |
| syntax | `excessive_branching` | `soft_score(right_branching_ratio)` |
| lexicon | `low_lexical_diversity` | `0.6·(1-soft_score(lemma_diversity)) + 0.4·(1-soft_score(type_token_ratio))` |
| lexicon | `rare_word_overuse` | `soft_score(rare_word_ratio)` |
| lexicon | `low_content_density` | `1 - soft_score(content_word_ratio)` |
| structure | `sentence_length_variability` | `soft_score(sentence_length_variance)` |
| structure | `punctuation_issues` | `0.5·(1-soft_score(punctuation_ratio)) + 0.5·soft_score(missing_terminal_punctuation_ratio)` |
| structure | `fragmentation` | `soft_score(short_sentence_ratio)` |
| discourse | `weak_cohesion` | `1 - soft_score(jaccard(lemma_set[i-1], lemma_set[i]))` |
| discourse | `missing_connectives` | `1 - soft_score(connective_ratio)` |
| discourse | `pronoun_ambiguity` | `soft_score(pronoun_to_noun_ratio)` |
| style | `structural_inconsistency` | `soft_score(pos_distribution_variance)` |
| style | `sentence_progression_drift` | `abs(soft_score(sentence_length_trend) - 0.5) * 2` |

## Layer 6 — ML Distillation

Layer 6 adds a multi-task transformer student model that learns to predict linguistic scores, issues, and diagnoses directly from raw Hebrew text, using the existing deterministic pipeline (Layers 1–5) as a teacher. The student replaces the full pipeline's heavy NLP passes (Stanza ~30s + YAP ~5s) with a single forward pass (~50ms GPU / ~200ms CPU), enabling interactive use cases such as real-time writing feedback.

### Model Architecture

The student model uses a DictaBERT Hebrew encoder with five prediction heads — three document-level and two sentence-level:

- **Scores head** — `Linear(hidden_dim, 5)` → sigmoid → 5 composite scores (difficulty, style, fluency, cohesion, complexity)
- **Issues head** — `Linear(hidden_dim, 17)` → sigmoid → 17 issue type severities
- **Diagnoses head** — `Linear(hidden_dim, 8)` → sigmoid → 8 diagnosis type severities
- **Sentence head** — `Linear(hidden_dim, 1)` → sigmoid → per-sentence complexity prediction
- **Pair head** — `Linear(hidden_dim × 2, 1)` → sigmoid → per-adjacent-pair cohesion weakness prediction

A single DictaBERT forward pass produces all outputs. The CLS token feeds the three document-level heads. Token embeddings are mean-pooled per sentence boundary to produce sentence representations, which feed the sentence head. Adjacent sentence representations are concatenated and fed to the pair head.

All outputs are continuous values in [0, 1] (soft labels). Interventions are NOT predicted — they are derived deterministically from predicted diagnoses via the existing `map_interventions()` function.

### Training Workflow

```
Export → Train → Evaluate → Iterate
```

1. **Export** — Read existing pipeline output JSONs, run the fast analysis + diagnosis layers, flatten results into JSONL training records with soft labels. Training records include `sentence_complexities` (per-sentence severity values) and `cohesion_pairs` (per-adjacent-pair severity values) for sentence-level training. Pass `--embed` to use sentence embeddings for higher-quality cohesion labels (cosine similarity instead of Jaccard fallback).
2. **Train** — Fine-tune DictaBERT encoder + 5 heads using multi-task loss: `L = w1·MSE(scores) + w2·BCE(issues) + w3·BCE(diagnoses) + w4·BCE(sentence_complexity) + w5·BCE(weak_cohesion)`. Default weights: (1.0, 1.5, 2.0, 1.5, 1.5). Uses three-tier differential learning rates: encoder (2e-5), document-level heads (1e-3), and sentence-level heads (5e-3).
3. **Evaluate** — RMSE per score, F1 per issue/diagnosis at threshold 0.5, Spearman rank correlation.
4. **Iterate** — Disagreement mining compares model predictions against pipeline labels, flags divergence cases, and expands the training set.

### Inference Modes

- **Fast path** — Single forward pass (~50ms GPU). No Stanza or YAP required. Returns scores, issues, diagnoses, interventions, and sentence-level predictions (`sentence_complexity` and `weak_cohesion` lists).
- **Hybrid mode** — Uses the model when confidence exceeds a threshold (default 0.7), falls back to the full pipeline otherwise. Output includes a `"source"` field (`"model"` or `"pipeline"`).

### ML Distillation CLI

```bash
# Export training data from pipeline output directories
python export_training_data.py export \
    --input-dirs results_sample/ results_hedc4/ \
    --stats-path feature_stats_merged.json \
    --output training_data.jsonl \
    --stats-output label_stats.json

# Export with embedding-based cohesion labels (requires sentence-transformers)
# Uses cosine similarity instead of Jaccard fallback for cohesion labels
python export_training_data.py export \
    --input-dirs results_sample/ results_hedc4/ \
    --stats-path feature_stats_merged.json \
    --output training_data.jsonl \
    --embed

# Export with a custom embedding model
python export_training_data.py export \
    --input-dirs results_sample/ results_hedc4/ \
    --stats-path feature_stats_merged.json \
    --output training_data.jsonl \
    --embed --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Train locally (auto-detects CUDA / CPU; use --device mps to force Apple Silicon GPU)
python train_model.py --data training_data.jsonl --output model_checkpoint/

# Train with custom hyperparameters
python train_model.py --data training_data.jsonl --output model_checkpoint/ \
    --encoder dicta-il/dictabert --batch-size 16 --epochs 3 \
    --encoder-lr 2e-5 --heads-lr 1e-3 --sentence-heads-lr 5e-3

# Resume training from a checkpoint
python train_model.py --data training_data.jsonl --output model_checkpoint/ \
    --resume model_checkpoint/

# Train on SageMaker (defaults to eu-west-1, s3://hebrew-profiler-ml-training/output/)
# --role defaults to SageMakerTrainingRole in account 921400262514
python launch_sagemaker_training.py \
    --data training_data.jsonl \
    --instance-type ml.g4dn.xlarge

# Train on SageMaker with SSO profile and custom region
python launch_sagemaker_training.py \
    --data training_data.jsonl \
    --profile my-sso-profile \
    --region us-east-1 \
    --output-s3 s3://my-bucket/model-output/

# Train on SageMaker with a custom IAM role
python launch_sagemaker_training.py \
    --data training_data.jsonl \
    --role arn:aws:iam::123456789012:role/CustomRole \
    --instance-type ml.g4dn.xlarge

# Run inference (fast path)
python export_training_data.py infer \
    --text "Hebrew text here" \
    --model model_checkpoint/

# Run inference (hybrid mode with pipeline fallback)
python export_training_data.py infer \
    --text "Hebrew text here" \
    --model model_checkpoint/ \
    --hybrid

# Disagreement mining
python export_training_data.py disagree \
    --predictions predictions.jsonl \
    --labels pipeline_labels.jsonl \
    --output disagreements.jsonl

# Merge disagreements into training data
python export_training_data.py merge \
    --base training_data.jsonl \
    --disagreements disagreements.jsonl \
    --output training_data_expanded.jsonl
```

## Hebrew Writing Coach

A full-stack web application that wraps the DictaBERT distillation model (Layer 6) in a real-time Hebrew writing feedback experience. Users write or paste Hebrew text in an RTL Monaco editor, receive instant linguistic analysis (scores, diagnoses, sentence annotations, cohesion gaps), and improve their writing through guided rewrites powered by Amazon Bedrock LLMs.

The entire UI is in Hebrew. No user accounts — admin pages are password-protected via environment variable.

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Frontend   │────▶│   API Layer  │────▶│  ML Model Service│
│   (React)    │◀────│   (FastAPI)  │◀────│  (DictaBERT)     │
└──────────────┘     └──────────────┘     └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  Amazon Bedrock  │
                     │  (LLM Rewrite)   │
                     └──────────────────┘
```

- **Frontend** — React 18 + TypeScript, Monaco Editor (RTL), Tailwind CSS, Zustand, recharts (spider chart), Vite
- **Backend** — FastAPI, DictaBERT model loaded in memory at startup, Pydantic schemas, Hebrew localization layer
- **LLM Rewrite** — Amazon Bedrock integration for AI-powered Hebrew rewrite suggestions
- **Deployment** — Docker Compose for local dev, ECS task definitions + ALB routing for production

### Docker Compose Quickstart

```bash
# Build and start both containers (frontend on port 3000, backend on port 8000)
docker-compose up --build

# The backend expects the trained model at ./model_v5/
# Set environment variables in docker-compose.yml:
#   MODEL_PATH=/app/model_v5
#   AWS_REGION=us-east-1
#   ADMIN_PASSWORD=admin
#   FRONTEND_ORIGIN=http://localhost:3000
```

Open `http://localhost:3000` in your browser. The backend API is available at `http://localhost:8000`.

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/analyze` | None | Full text analysis — scores, diagnoses, interventions, sentence annotations, cohesion gaps |
| POST | `/api/revise` | None | Compare original and edited text — delta scores, resolved/new diagnoses |
| POST | `/api/rewrite` | None | AI rewrite suggestion via Amazon Bedrock for a specific diagnosis |
| GET | `/api/examples` | None | List available example texts (tweet, news, legal, essay, blog) |
| GET | `/api/examples/{id}` | None | Get full text of a specific example |
| GET | `/api/health` | None | Health check — returns model loaded status (200 healthy / 503 unhealthy) |
| GET | `/admin/config` | Admin | Get current admin configuration |
| POST | `/admin/config` | Admin | Update Bedrock model, severity threshold, display limits |
| GET | `/admin/models` | Admin | List available Bedrock foundation models |

Admin endpoints require the `X-Admin-Password` header matching the `ADMIN_PASSWORD` environment variable.

### Admin Configuration

The admin panel at `/admin` allows configuring:

- **Bedrock Model** — Select which Amazon Bedrock model to use for rewrite suggestions
- **Severity Threshold** (0.0–1.0) — Minimum severity for a diagnosis to be shown to users
- **Max Diagnoses** — Maximum number of diagnoses displayed (1–10)
- **Max Interventions** — Maximum number of interventions displayed (1–10)

### Frontend Pages

- **/** — Main editor + analysis panel with spider chart, diagnoses, interventions, and guided rewrite workflow
- **/methodology** — Public page explaining the analysis methodology in Hebrew (features, scores, issues, diagnoses, interventions)
- **/admin** — Password-protected admin configuration page

## Project Structure

```
hebrew_profiler/
├── __init__.py
├── normalizer.py          # Unicode NFKC normalization + punctuation standardization
├── tokenizer.py           # Hebrew-aware tokenization with prefix/suffix annotation
├── stanza_setup.py        # Stanza model verification and pipeline caching
├── stanza_adapter.py      # Morphological analysis via Stanza (local)
├── yap_adapter.py         # Syntactic parsing via YAP API (pre-splits sentences)
├── ir_builder.py          # Intermediate Representation construction
├── feature_extractor.py   # Morphological, syntactic, lexical, structural features
├── scorer.py              # Difficulty and style scoring
├── pipeline.py            # Pipeline orchestration + JSON serialization
├── batch.py               # Batch processing + JSONL export
├── models.py              # All dataclass definitions
├── config.py              # Default configuration values
└── errors.py              # Custom exception types

analysis/
├── __init__.py            # Public API exports
├── normalization.py       # soft_score / inverted_soft_score (sigmoid-based)
├── statistics.py          # FeatureStats, corpus stats computation + JSON persistence
├── issue_models.py        # Issue dataclass
├── diagnosis_models.py    # Diagnosis and Intervention dataclasses
├── sentence_metrics.py    # SentenceMetrics + extract_sentence_metrics(ir, sentences, embedder)
├── issue_detector.py      # detect_issues — 17 issue types; weak_cohesion uses embeddings or Jaccard
├── issue_ranker.py        # rank_issues — composite scoring, top-K selection
├── diagnosis_engine.py    # 8 diagnosis rules + run_diagnoses() — Layer 4
├── intervention_mapper.py # INTERVENTION_MAP + map_interventions() — Layer 5
├── interpretation.py      # run_interpretation() entry point for Layers 4–5
├── analysis_pipeline.py   # flatten_features, AnalysisInput, run_analysis_pipeline(text, config, embedder)
├── embedder.py            # SentenceEmbedder — lazy singleton wrapping sentence-transformers
├── serialization.py       # serialize_issues, serialize_interpretation → JSON
├── test_normalization.py  # Properties 5–6
├── test_statistics.py     # Properties 1–4
├── test_sentence_metrics.py  # Property 8
├── test_issue_detector.py    # Properties 7, 9, 10, 13
├── test_issue_ranker.py      # Property 11
├── test_serialization.py     # Property 12
├── test_integration.py       # End-to-end corpus + analysis integration tests
├── test_diagnosis_engine.py  # Diagnosis properties 1–3
├── test_intervention_mapper.py # Intervention properties 4–5
└── test_interpretation.py    # Integration + serialization properties 6–7

ml/
├── __init__.py            # Package init, public API exports
├── model.py               # LinguisticModel: DictaBERT encoder + 5 prediction heads (3 document + 2 sentence)
├── export.py              # Data export: pipeline JSONs → training JSONL
├── dataset.py             # LinguisticDataset: JSONL → PyTorch tensors
├── sentence_utils.py      # Sentence splitting and token-boundary mapping utilities
├── trainer.py             # Shared training logic (model construction, loop, eval, checkpointing)
├── inference.py           # Fast path, hybrid mode, intervention derivation
├── disagreement.py        # Model vs pipeline comparison, training set expansion
├── requirements.txt       # SageMaker training container dependencies (transformers)
├── test_export.py         # Properties 1–5, export unit tests
├── test_model.py          # Property 6, model architecture tests
├── test_dataset.py        # Properties 7, 9, dataset unit tests
├── test_trainer.py        # Property 8, training logic tests
├── test_inference.py      # Properties 10–12, inference tests
├── test_disagreement.py   # Properties 13–14, disagreement mining tests
└── test_cli.py            # CLI argument parsing tests

app/                       # Hebrew Writing Coach backend (FastAPI)
├── main.py                # FastAPI app, CORS, lifespan (model + example loading)
├── config.py              # Pydantic Settings (MODEL_PATH, AWS_REGION, ADMIN_PASSWORD, etc.)
├── Dockerfile             # Backend container image
├── requirements.txt       # Backend Python dependencies
├── api/
│   ├── analyze.py         # POST /api/analyze — full text analysis
│   ├── revise.py          # POST /api/revise — revision comparison
│   ├── rewrite.py         # POST /api/rewrite — Bedrock LLM rewrite
│   ├── examples.py        # GET /api/examples — example text management
│   ├── admin.py           # Admin endpoints (password-protected)
│   └── health.py          # GET /api/health — health check
├── services/
│   ├── model_service.py   # DictaBERT model loading + inference wrapper
│   ├── bedrock_service.py # Amazon Bedrock LLM integration
│   ├── localization.py    # Hebrew labels, explanations, tips for all diagnosis/score types
│   └── example_service.py # Example text loading from JSON files
├── models/
│   └── schemas.py         # Pydantic request/response models
└── data/
    └── examples/          # Example text JSON files (tweet, news, legal, essay, blog)

frontend/                  # Hebrew Writing Coach frontend (React + TypeScript)
├── Dockerfile             # Multi-stage build: Node → Nginx SPA
├── nginx.conf             # Nginx config for SPA routing
├── vite.config.ts         # Vite dev server with API proxy
├── vitest.config.ts       # Vitest test configuration
└── src/
    ├── App.tsx            # Root component (RTL, Hebrew, React Router)
    ├── api/
    │   └── client.ts      # Axios-based API client
    ├── components/
    │   ├── Header.tsx             # Logo, title, export/share buttons
    │   ├── MainLayout.tsx         # Two-pane layout (stacks on mobile)
    │   ├── MonacoEditor.tsx       # RTL Monaco editor with 800ms debounce
    │   ├── useInlineAnnotations.ts # Sentence highlights + cohesion gap decorations
    │   ├── ScoreSpiderChart.tsx   # 5-axis radar chart (recharts)
    │   ├── DiagnosisList.tsx      # Sorted diagnosis cards
    │   ├── DiagnosisCard.tsx      # Diagnosis severity, label, explanation, tip
    │   ├── InterventionCard.tsx   # Practice + AI Rewrite buttons
    │   ├── RewriteModal.tsx       # Guided rewrite with diff view + delta scores
    │   ├── ExampleSelector.tsx    # Example text category buttons
    │   └── ProgressFeedback.tsx   # Before/after score deltas
    ├── pages/
    │   ├── HomePage.tsx           # Main editor + analysis page
    │   ├── MethodologyPage.tsx    # Public methodology explanation (Hebrew)
    │   └── AdminPage.tsx          # Password-protected admin config
    ├── store/
    │   └── useAppStore.ts         # Zustand state management
    ├── types/
    │   └── index.ts               # TypeScript interfaces matching Pydantic schemas
    └── utils/
        ├── exportPdf.ts           # PDF export (jsPDF)
        └── shareUrl.ts            # Share URL encoding/decoding

deploy/                    # AWS deployment configuration
├── ecs-backend.json       # ECS task definition for backend (2048 CPU / 4096MB)
├── ecs-frontend.json      # ECS task definition for frontend (256 CPU / 512MB)
├── alb-rules.json         # ALB routing rules (/ → frontend, /api/* → backend)
└── iam-policy.json        # IAM policy for Bedrock access

run_pipeline.py            # CLI entry point (single + batch modes)
run_analysis.py            # Standalone analysis layer CLI
download_hedc4.py          # Stream HeDC4 corpus from HuggingFace → .txt files
merge_stats.py             # Merge feature statistics from multiple result directories
split_corpus.py            # Splits wikipedia.raw into per-document .txt files; builds/loads freq dicts
analyze_results.py         # Outlier detection + normalization range analysis for batch results
check_results.py           # Batch results health check — feature completeness & layer failure detection
train_model.py             # Local training CLI for ML distillation (Layer 6)
sagemaker_train.py         # SageMaker training entry point (Layer 6)
launch_sagemaker_training.py  # SageMaker job launcher (Layer 6)
export_training_data.py    # Data export, inference, and disagreement mining CLI (Layer 6)
test_webapp.py             # Playwright e2e tests for the Hebrew Writing Coach frontend

tests/
├── test_normalizer.py     # Properties 1-2
├── test_tokenizer.py      # Properties 3-5
├── test_stanza_setup.py   # Stanza model check tests
├── test_stanza_adapter.py # Property 6
├── test_yap_adapter.py    # Properties 7-8
├── test_ir_builder.py     # Properties 9-11
├── test_feature_extractor.py  # Properties 12-15, 24
├── test_scorer.py         # Properties 16-19
├── test_serialization.py  # Properties 20-21
├── test_batch.py          # Properties 22-23
├── test_pipeline.py       # Pipeline integration tests
└── test_cli.py            # CLI argument parsing + exit codes
```

## Testing

The project uses pytest with Hypothesis for property-based testing. 24 correctness properties are formally specified and tested for the core pipeline, 13 for the analysis layer, 7 for the diagnosis and intervention layers, 14 for the ML distillation layer, and 12 for the Hebrew Writing Coach.

```bash
# Run all pipeline tests
python -m pytest tests/ hebrew_profiler/normalizer_test.py -v

# Run all analysis layer tests (including diagnosis and intervention)
python -m pytest analysis/ -v

# Run all ML distillation layer tests
python -m pytest ml/ -v

# Run all Hebrew Writing Coach backend tests
python -m pytest app/ -v

# Run Hebrew Writing Coach frontend tests
cd frontend && npx vitest run

# Run Hebrew Writing Coach Playwright e2e tests (requires frontend on port 3001)
python test_webapp.py

# Run everything (backend)
python -m pytest tests/ analysis/ ml/ app/ hebrew_profiler/normalizer_test.py -v
```

### Pipeline Correctness Properties (tests/)

| # | Property | Module |
|---|----------|--------|
| 1 | Normalization idempotence | normalizer |
| 2 | Punctuation standardization completeness | normalizer |
| 3 | Token-offset round-trip consistency | tokenizer |
| 4 | Prefix particle annotation | tokenizer |
| 5 | Suffix pronoun annotation | tokenizer |
| 6 | Stanza response parsing completeness | stanza_adapter |
| 7 | CoNLL dependency tree parsing completeness | yap_adapter |
| 8 | Multi-sentence CoNLL segmentation | yap_adapter |
| 9 | IR structural completeness | ir_builder |
| 10 | IR token-to-dependency alignment | ir_builder |
| 11 | IR graceful degradation | ir_builder |
| 12 | Morphological feature formulas | feature_extractor |
| 13 | Syntactic feature formulas | feature_extractor |
| 14 | Lexical feature formulas | feature_extractor |
| 15 | Structural feature formulas | feature_extractor |
| 16 | Min-max normalization range invariant | scorer |
| 17 | Difficulty score formula and range | scorer |
| 18 | Style score formula | scorer |
| 19 | Absent feature handling in scoring | scorer |
| 20 | Hebrew character preservation in serialization | pipeline |
| 21 | Partial features serialized as null | pipeline |
| 22 | JSONL record completeness | batch |
| 23 | Batch count consistency | batch |
| 24 | Feature extraction with missing IR layers | feature_extractor |

### Analysis Layer Correctness Properties (analysis/)

| # | Property | Module |
|---|----------|--------|
| 1 | FeatureStats coverage | statistics |
| 2 | FeatureStats correctness | statistics |
| 3 | Stability and degeneracy flags | statistics |
| 4 | Statistics round-trip | statistics |
| 5 | soft_score formula correctness | normalization |
| 6 | soft_score range invariant | normalization |
| 7 | Issue field invariants | issue_detector |
| 8 | SentenceMetrics count matches IR sentences | sentence_metrics |
| 9 | Span correctness | issue_detector |
| 10 | Confidence formula | issue_detector |
| 11 | Ranker ordering and selection | issue_ranker |
| 12 | JSON serialization completeness | serialization |
| 13 | Jaccard similarity correctness | issue_detector |

### Diagnosis & Intervention Properties (analysis/)

| # | Property | Module |
|---|----------|--------|
| 1 | Helper function correctness | diagnosis_engine |
| 2 | Diagnosis severity formula and threshold correctness | diagnosis_engine |
| 3 | Diagnosis aggregation ordering | diagnosis_engine |
| 4 | Intervention mapping correctness | intervention_mapper |
| 5 | Intervention aggregation ordering | intervention_mapper |
| 6 | Integration output structure | interpretation |
| 7 | Serialization round-trip | interpretation |

### ML Distillation Layer Properties (ml/)

| # | Property | Module |
|---|----------|--------|
| 1 | Issue flattening completeness and correctness | export |
| 2 | Diagnosis flattening completeness and correctness | export |
| 3 | Training record round-trip serialization | export |
| 4 | Null score substitution in export | export |
| 5 | Malformed JSON rejection in export | export |
| 6 | Model output shape and range invariants | model |
| 7 | Dataset tensor shape correctness | dataset |
| 8 | Multi-task loss as weighted sum | trainer |
| 9 | Tokenization truncation bound | dataset |
| 10 | Inference output round-trip serialization | inference |
| 11 | Diagnosis threshold conversion for interventions | inference |
| 12 | Hybrid mode confidence-based routing | inference |
| 13 | Disagreement detection correctness | disagreement |
| 14 | Training data deduplication on merge | disagreement |

### Hebrew Writing Coach Properties (app/, frontend/)

| # | Property | Module |
|---|----------|--------|
| 1 | Score range invariant | analyze |
| 2 | Diagnosis structure completeness | analyze |
| 3 | Intervention referential integrity | analyze |
| 4 | Sentence annotation consistency | analyze |
| 5 | Cohesion gap adjacency | analyze |
| 6 | Delta score arithmetic | revise |
| 7 | Diagnosis set transitions | revise |
| 8 | Rewrite prompt completeness | bedrock_service |
| 9 | Invalid diagnosis type rejection | rewrite |
| 10 | Diagnosis display filtering and ordering | analyze |
| 11 | Localization completeness | localization |
| 12 | Share URL round-trip | shareUrl (frontend) |

## Configuration

Default scoring weights and normalization ranges can be customized via `PipelineConfig`:

```python
from hebrew_profiler.models import PipelineConfig, DifficultyWeights, StyleWeights, NormalizationRanges

config = PipelineConfig(
    difficulty_weights=DifficultyWeights(w1=0.30, w2=0.25, w3=0.25, w4=0.20),
    style_weights=StyleWeights(a1=0.25, a3=0.25, a4=0.20, a5=0.15, a6=0.15),
    normalization_ranges=NormalizationRanges(
        avg_sentence_length=(10.0, 40.0),
        avg_tree_depth=(4.0, 15.0),
        hapax_ratio=(0.15, 0.55),
        morphological_ambiguity=(4.0, 10.0),
        suffix_pronoun_ratio=(0.05, 0.50),
        sentence_length_variance=(0.0, 400.0),
        rare_word_ratio=(0.0, 0.3),
        content_word_ratio=(0.4, 0.8),
        connective_ratio=(0.0, 2.0),
        sentence_overlap=(0.0, 0.6),
        agreement_error_rate=(0.0, 0.3),
        dependency_distance_variance=(0.0, 20.0),
    ),
    long_sentence_threshold=20,
)
```

## Error Handling

The pipeline is designed for resilience:

- If Stanza fails, morphological features are `null` but syntactic, lexical, and structural features are still computed
- If YAP fails for individual sentences, those sentences are skipped and partial results are returned from the remaining sentences. If 3+ consecutive sentences fail, the adapter waits up to 60 seconds for YAP to recover (pinging every 5 seconds). If YAP comes back, the failing sentence is retried and processing continues. If YAP does not recover, the request returns an error — syntactic features become `null` but morphological, lexical, and structural features are still computed
- In batch mode, a single document failure is logged and processing continues
- Invalid UTF-8 files are skipped with an encoding error logged
- All errors are logged to stderr in the format: `[{timestamp}] ERROR [{document_id}] {error_type}: {message}`

## License

This project is for internal/research use.
