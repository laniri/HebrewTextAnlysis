# Implementation Spec: Sentence-Level Architecture for ML Distillation Model

## Problem Statement

The current ML distillation model (Layer 6) uses only the `[CLS]` token as a document-level representation for all predictions. Two issue types that require sub-document granularity are stuck at F1=0.0:

- **`sentence_complexity`** (per-sentence, span `(i,)`) — needs to know which specific sentences are complex
- **`weak_cohesion`** (per-sentence-pair, span `(i-1, i)`) — needs to compare adjacent sentence representations

Their dependent diagnoses are also stuck at 0.0:
- **`sentence_over_complexity`** — depends on sentence_complexity
- **`low_cohesion`** — depends on weak_cohesion

## Design: Single Pass with Sentence Pooling

### Architecture

```
Input text (tokenized, max 512 tokens)
    │
    ▼
DictaBERT Encoder
    │
    ├── last_hidden_state[:, 0, :] ──► [CLS] (hidden_dim)
    │       │
    │       ├── scores_head (Linear → 5, sigmoid)          [EXISTING, UNCHANGED]
    │       ├── issues_head (Linear → 17, sigmoid)         [EXISTING, UNCHANGED]
    │       └── diagnoses_head (Linear → 8, sigmoid)       [EXISTING, UNCHANGED]
    │
    └── last_hidden_state[:, 1:, :] ──► token embeddings (seq_len, hidden_dim)
            │
            ▼
        Sentence Boundary Detection
        (map token positions to sentence indices)
            │
            ▼
        Mean Pooling per Sentence ──► sentence_embeddings (num_sentences, hidden_dim)
            │
            ├── sentence_head (Linear → 1, sigmoid) ──► per-sentence complexity score
            │       Applied independently to each sentence embedding
            │       Output: (num_sentences,) float values in [0, 1]
            │
            └── Adjacent Pair Concatenation ──► pair_embeddings (num_pairs, 2 * hidden_dim)
                    │
                    └── pair_head (Linear → 1, sigmoid) ──► per-pair cohesion weakness score
                            Output: (num_pairs,) float values in [0, 1]
```

### Key Design Decisions

1. **Single forward pass.** The encoder runs once on the full document. Sentence-level representations are derived by pooling token embeddings within sentence boundaries. No per-sentence encoding.

2. **Existing heads unchanged.** The 3 document-level heads (scores, issues, diagnoses) continue to use the CLS token exactly as before. The sentence-level heads are additive — they don't affect existing predictions.

3. **Mean pooling over sentence tokens.** For each sentence, average the token embeddings within that sentence's token span. This is simple, fast, and effective for capturing sentence-level semantics.

4. **Pair concatenation for cohesion.** For adjacent sentence pairs (i, i+1), concatenate their sentence embeddings into a (2 * hidden_dim) vector and pass through a linear head. This lets the model learn what makes two sentences cohesive or not.

5. **Variable-length outputs.** Different documents have different numbers of sentences and pairs. The model outputs variable-length tensors. The loss function handles this via masking.

## Data Format Changes

### Current Training Record (JSONL)

```json
{
  "text": "...",
  "scores": {"difficulty": 0.72, ...},
  "issues": {"sentence_complexity": 0.83, "weak_cohesion": 0.74, ...},
  "diagnoses": {"sentence_over_complexity": 0.78, ...}
}
```

The `issues` dict contains document-level max severity for each type. Per-sentence information is lost.

### New Training Record (JSONL)

```json
{
  "text": "...",
  "scores": {"difficulty": 0.72, ...},
  "issues": {"sentence_complexity": 0.83, "weak_cohesion": 0.74, ...},
  "diagnoses": {"sentence_over_complexity": 0.78, ...},
  "sentence_complexities": [0.27, 0.69, 0.31, 0.99, 0.82, 0.35],
  "cohesion_pairs": [0.65, 0.54, 0.88, 0.74, 0.41]
}
```

New fields:
- **`sentence_complexities`**: List of float severity values, one per sentence in the document. Each value is the `sentence_complexity` issue severity for that sentence (from the pipeline's `detect_issues` output). Sentences without a complexity issue get 0.0.
- **`cohesion_pairs`**: List of float severity values, one per adjacent sentence pair. Each value is the `weak_cohesion` issue severity for that pair (from the pipeline's `detect_issues` output). Pairs without a cohesion issue get 0.0.

Both lists have variable length depending on the document's sentence count.

## Module Changes

### 1. Export Module (`ml/export.py`)

**New function: `_extract_sentence_labels(issues: list[Issue], sentence_count: int) -> dict`**

```python
def _extract_sentence_labels(issues: list[Issue], sentence_count: int) -> dict:
    """Extract per-sentence and per-pair labels from the issue list.
    
    Returns:
        {
            "sentence_complexities": [float, ...],  # length = sentence_count
            "cohesion_pairs": [float, ...],          # length = max(0, sentence_count - 1)
        }
    """
```

Logic:
- Initialize `sentence_complexities` as `[0.0] * sentence_count`
- For each issue with `type == "sentence_complexity"` and `span == (i,)`, set `sentence_complexities[i] = issue.severity`
- Initialize `cohesion_pairs` as `[0.0] * max(0, sentence_count - 1)`
- For each issue with `type == "weak_cohesion"` and `span == (i, i+1)`, set `cohesion_pairs[i] = issue.severity`

**Change to `export_training_data()`:**
- After detecting issues, also extract sentence-level labels
- Need sentence count — derive from the pipeline JSON's text (count sentence-ending punctuation, or use the sentence_metrics if available)
- Add `sentence_complexities` and `cohesion_pairs` to each training record

**Sentence count derivation:**
The pipeline JSON doesn't directly store sentence count, but we can derive it from the `sentence_complexity` issues (their spans tell us which sentences exist) or by splitting the text on sentence-ending punctuation. The simplest approach: count the max sentence index from all per-sentence issues, plus 1.

Alternatively, since we're running `detect_issues()` during export, we can count the sentence_complexity issues — the pipeline creates one per sentence regardless of severity. The count of `sentence_complexity` issues equals the sentence count.

### 2. Model (`ml/model.py`)

**New components in `LinguisticModel`:**

```python
class LinguisticModel(nn.Module):
    def __init__(self, encoder_name="dicta-il/dictabert", ...):
        # ... existing init ...
        
        # New: sentence-level heads
        hidden = self.encoder.config.hidden_size
        self.sentence_head = nn.Linear(hidden, 1)       # per-sentence complexity
        self.pair_head = nn.Linear(hidden * 2, 1)        # per-pair cohesion weakness
    
    def forward(self, input_ids, attention_mask, sentence_boundaries=None):
        encoder_output = self.encoder(input_ids, attention_mask)
        cls = encoder_output.last_hidden_state[:, 0, :]
        
        # Existing document-level predictions (unchanged)
        doc_output = {
            "scores": torch.sigmoid(self.scores_head(cls)),
            "issues": torch.sigmoid(self.issues_head(cls)),
            "diagnoses": torch.sigmoid(self.diagnoses_head(cls)),
        }
        
        # New: sentence-level predictions (only when boundaries provided)
        if sentence_boundaries is not None:
            token_embeddings = encoder_output.last_hidden_state  # (B, seq_len, hidden)
            
            # For each item in the batch, pool tokens per sentence
            # sentence_boundaries: list of list of (start, end) tuples per batch item
            sentence_preds = []
            pair_preds = []
            
            for b in range(token_embeddings.shape[0]):
                boundaries = sentence_boundaries[b]  # [(start, end), ...]
                
                # Mean-pool tokens per sentence
                sent_embeds = []
                for start, end in boundaries:
                    if end > start:
                        sent_embed = token_embeddings[b, start:end, :].mean(dim=0)
                    else:
                        sent_embed = torch.zeros(token_embeddings.shape[-1], device=token_embeddings.device)
                    sent_embeds.append(sent_embed)
                
                if sent_embeds:
                    sent_matrix = torch.stack(sent_embeds)  # (num_sentences, hidden)
                    sent_scores = torch.sigmoid(self.sentence_head(sent_matrix)).squeeze(-1)  # (num_sentences,)
                else:
                    sent_scores = torch.tensor([], device=token_embeddings.device)
                
                sentence_preds.append(sent_scores)
                
                # Adjacent pair concatenation
                if len(sent_embeds) >= 2:
                    pairs = []
                    for i in range(len(sent_embeds) - 1):
                        pair = torch.cat([sent_embeds[i], sent_embeds[i + 1]])  # (2 * hidden,)
                        pairs.append(pair)
                    pair_matrix = torch.stack(pairs)  # (num_pairs, 2 * hidden)
                    pair_scores = torch.sigmoid(self.pair_head(pair_matrix)).squeeze(-1)  # (num_pairs,)
                else:
                    pair_scores = torch.tensor([], device=token_embeddings.device)
                
                pair_preds.append(pair_scores)
            
            doc_output["sentence_complexity"] = sentence_preds  # list of tensors (variable length)
            doc_output["weak_cohesion"] = pair_preds             # list of tensors (variable length)
        
        return doc_output
```

**Backward compatibility:** When `sentence_boundaries` is None (inference without sentence info, or existing code), the model returns only the 3 document-level outputs — identical to the current behavior.

### 3. Dataset (`ml/dataset.py`)

**Changes to `LinguisticDataset.__getitem__`:**

The dataset needs to:
1. Detect sentence boundaries in the tokenized text
2. Return sentence boundary indices alongside the existing tensors
3. Return per-sentence and per-pair label tensors

**Sentence boundary detection:**
Split the original text into sentences (using the same sentence-ending punctuation regex as the pipeline). For each sentence, tokenize it separately to find its token span in the full tokenized sequence. This gives us `(start_token_idx, end_token_idx)` per sentence.

Alternative (simpler): Use the tokenizer's `offset_mapping` to map character positions to token positions. Split text into sentences by character position, then map to token positions.

**New return fields:**

```python
{
    "input_ids": (max_length,),
    "attention_mask": (max_length,),
    "scores": (5,),
    "issues": (17,),
    "diagnoses": (8,),
    # New:
    "sentence_boundaries": [(start, end), ...],       # variable length
    "sentence_complexities": (num_sentences,),         # variable length float tensor
    "cohesion_pairs": (num_pairs,),                    # variable length float tensor
}
```

**Collation:** Since sentence counts vary per document, we can't use the default PyTorch collator. We need a custom `collate_fn` that:
- Pads/stacks the fixed-size tensors normally (input_ids, attention_mask, scores, issues, diagnoses)
- Keeps sentence_boundaries, sentence_complexities, and cohesion_pairs as lists (not stacked)

### 4. Trainer (`ml/trainer.py`)

**Loss function changes:**

```python
L = (
    w1 * L_scores +
    w2 * L_issues +
    w3 * L_diagnoses +
    w4 * L_sentence_complexity +    # NEW
    w5 * L_weak_cohesion            # NEW
)
```

New loss terms:
- `L_sentence_complexity`: BCE between predicted per-sentence scores and target per-sentence severities. Computed per batch item (variable length), then averaged.
- `L_weak_cohesion`: BCE between predicted per-pair scores and target per-pair severities. Same variable-length handling.

**Default weights:** `w4 = 1.5`, `w5 = 1.5` (same as issues weight).

**TrainConfig changes:**
```python
@dataclass
class TrainConfig:
    # ... existing fields ...
    loss_weights: tuple[float, ...] = (1.0, 1.5, 2.0, 1.5, 1.5)  # scores, issues, diagnoses, sent_complexity, cohesion
```

**Evaluation metrics changes:**
- Add per-sentence F1 for sentence_complexity (threshold 0.5)
- Add per-pair F1 for weak_cohesion (threshold 0.5)
- These are computed across all sentences/pairs in the validation set, not per-document

**Custom collate_fn:**
The trainer needs to use a custom collate function for the DataLoader that handles variable-length sentence data.

### 5. Inference (`ml/inference.py`)

**Changes to `predict()`:**

```python
def predict(text, model_path, device=None):
    # ... existing code ...
    
    # NEW: detect sentence boundaries
    sentences = split_into_sentences(text)
    sentence_boundaries = find_token_boundaries(sentences, tokenizer, text)
    
    # Forward pass with boundaries
    output = model(input_ids, attention_mask, sentence_boundaries=[sentence_boundaries])
    
    # Extract sentence-level predictions
    sent_complexity_scores = output["sentence_complexity"][0]  # tensor (num_sentences,)
    cohesion_scores = output["weak_cohesion"][0]               # tensor (num_pairs,)
    
    # Include in output
    result["sentence_complexity"] = [
        {"sentence": i, "severity": float(s)}
        for i, s in enumerate(sent_complexity_scores)
        if s > 0.3  # only include notable ones
    ]
    result["weak_cohesion"] = [
        {"pair": [i, i+1], "severity": float(s)}
        for i, s in enumerate(cohesion_scores)
        if s > 0.3
    ]
```

**The document-level `issues` dict still contains the max severity** for sentence_complexity and weak_cohesion (aggregated from the sentence-level predictions). This maintains backward compatibility with the existing output format.

### 6. Sentence Boundary Utilities (`ml/sentence_utils.py`)

New utility module for sentence boundary detection in tokenized text:

```python
def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using the same logic as the pipeline."""

def find_token_boundaries(
    sentences: list[str],
    tokenizer: PreTrainedTokenizer,
    full_text: str,
    max_length: int = 512,
) -> list[tuple[int, int]]:
    """Map sentence character spans to token index spans.
    
    Returns list of (start_token_idx, end_token_idx) tuples.
    Sentences that fall beyond the 512-token truncation boundary
    are excluded (empty list for those).
    """
```

This module is shared by dataset.py (training) and inference.py (prediction).

## Training Data Generation

The export script needs access to per-sentence issues, which requires knowing the sentence count. Two approaches:

**Approach A: Use sentence_complexity issue count.**
The pipeline's `detect_issues()` creates one `sentence_complexity` issue per sentence (even if severity is low). Count them to get sentence_count. Extract their severities indexed by span.

**Approach B: Re-split the text.**
Split the text into sentences using the same regex as the pipeline's sentence splitter. Count sentences. Match issues to sentences by span index.

Approach A is more reliable since it uses the exact same sentence segmentation the pipeline used. But it requires that `detect_issues()` was called with `sentence_metrics` (which provides per-sentence data). During export, we pass empty `sentence_metrics=[]`, so sentence_complexity issues may not be generated.

**Resolution:** During export, we need to also extract sentence metrics from the pipeline JSON. The pipeline JSON contains the full text — we can split it into sentences and count them. For sentence_complexity labels, we use the issues with `type == "sentence_complexity"` and their spans. For weak_cohesion labels, we use issues with `type == "weak_cohesion"` and their spans.

**Important:** The current export passes `sentence_metrics=[]` to `detect_issues()`, which means sentence-level issues (sentence_complexity, weak_cohesion) are NOT generated during export. We need to fix this by either:
1. Building SentenceMetrics from the pipeline JSON features (requires IR reconstruction — too complex)
2. Running the full pipeline during export (too slow — defeats the purpose)
3. Storing per-sentence issues in the pipeline JSON output and reading them during export

**Option 3 is the right approach.** The pipeline JSON from `run_pipeline.py single --analyze` already contains per-sentence issues in the `"issues"` array with span fields. During export, we should read these directly from the analysis output JSON rather than re-running `detect_issues()`.

This means the export flow changes to:
1. Read pipeline JSON with `--analyze` output (contains issues with spans)
2. Extract per-sentence labels directly from the issues array
3. No need to re-run `detect_issues()` for sentence-level data

**Prerequisite:** The batch pipeline output must include analysis results. This means running `run_pipeline.py batch` with `--analyze` to produce JSONs that contain the issues array. Currently batch mode doesn't support `--analyze` — this would need to be added, or the export script runs analysis on top of existing batch output (which is what it does now, but without sentence_metrics).

**Practical solution for MVP:** 
- Keep the current export flow (re-run detect_issues during export)
- Pass proper sentence_metrics by reconstructing them from the pipeline JSON features
- The pipeline JSON contains `avg_sentence_length` and sentence count can be derived from `features.structure.sentence_length_variance` and `features.syntax.avg_sentence_length`

Actually, the simplest solution: **run the full analysis pipeline during export** for the sentence-level labels only. The analysis pipeline (`run_analysis_pipeline`) already produces sentence_metrics. We just need to call it with the text from the pipeline JSON. This is slow (requires Stanza + YAP) but only needs to be done once for training data generation.

**Alternative (recommended):** Add a `--analyze` flag to batch mode that also runs the analysis layer and stores the full issues array (with spans) in the output JSON. Then the export script reads these directly. This is a one-time batch processing cost.

## Implementation Order

1. **`ml/sentence_utils.py`** — sentence splitting and token boundary mapping
2. **`ml/export.py`** — add sentence-level label extraction
3. **`ml/model.py`** — add sentence_head and pair_head
4. **`ml/dataset.py`** — add sentence boundaries and labels, custom collate_fn
5. **`ml/trainer.py`** — add sentence-level loss terms, update TrainConfig
6. **`ml/inference.py`** — add sentence-level prediction extraction
7. **Tests** — update existing tests, add new ones for sentence-level components

## Backward Compatibility

- The model's `forward()` method is backward compatible: when `sentence_boundaries=None`, it returns only document-level outputs
- The training record format is backward compatible: old records without `sentence_complexities` and `cohesion_pairs` fields are treated as having no sentence-level labels (sentence loss terms are skipped)
- The inference output is backward compatible: sentence-level predictions are additional fields, not replacements
- Existing model checkpoints can be loaded — the new heads are randomly initialized and fine-tuned

## Expected Impact

Based on the current metrics:
- `sentence_complexity` F1: 0.0 → expected 0.5–0.7 (the model can now see per-sentence signals)
- `weak_cohesion` F1: 0.0 → expected 0.4–0.6 (pair concatenation captures inter-sentence relationships)
- `sentence_over_complexity` diagnosis F1: 0.0 → expected 0.4–0.6 (depends on improved sentence_complexity)
- `low_cohesion` diagnosis F1: 0.0 → expected 0.3–0.5 (depends on improved weak_cohesion)
- Other metrics: unchanged or slightly improved (the encoder learns better representations when trained with sentence-level signal)

## Open Questions

1. **Max sentences per document:** With 512 tokens and average sentence length of ~24 tokens, we get ~20 sentences max. Documents with more sentences will have their later sentences truncated. Is this acceptable?

2. **Sentence boundary accuracy:** The regex-based sentence splitter may not match the pipeline's Stanza-based sentence segmentation exactly. Small misalignments between training labels (from pipeline) and model predictions (from regex) could add noise. Consider using the same sentence splitter as the pipeline.

3. **Loss balancing:** The sentence-level losses have variable sample counts per batch (more sentences = more loss terms). Should we normalize by sentence count or let longer documents contribute more to the loss?

4. **Inference without sentence info:** In hybrid mode, when the model is confident, should we return sentence-level predictions? Or only return them when explicitly requested?
