#!/usr/bin/env python3
"""Download a sample of Hebrew documents from the HeDC4 dataset.

HeDC4 (Hebrew Deduplicated and Cleaned Common Crawl Corpus) contains 910K
documents across diverse genres: news, legal, commercial, blog, forum, and
government text. This script streams documents from HuggingFace and writes
them as individual .txt files — the same format that split_corpus.py produces
and run_pipeline.py batch consumes.

Quality filters ensure downloaded documents are meaningful prose suitable
for linguistic analysis training data:

- Minimum / maximum character length
- Minimum sentence count (rejects navigation menus, cookie banners)
- Minimum Hebrew character ratio (rejects English-heavy / code pages)
- Maximum sentence repetition ratio (rejects pages with repeated boilerplate)
- Optional URL-heavy document rejection

Requires: pip install datasets

Usage:
    # Download 500 documents with default quality filters
    python download_hedc4.py --output corpus_hedc4/

    # Download 5000 documents, stricter filters
    python download_hedc4.py --output corpus_hedc4_5k/ --max-docs 5000 \
        --min-length 300 --min-sentences 5 --min-hebrew-ratio 0.6

    # Download with a specific random seed for reproducibility
    python download_hedc4.py --output corpus_hedc4/ --max-docs 500 --seed 42

    # Skip first N documents (useful for creating non-overlapping samples)
    python download_hedc4.py --output corpus_hedc4/ --max-docs 500 --skip 1000

    # Disable URL filtering (keep URL-heavy documents)
    python download_hedc4.py --output corpus_hedc4/ --allow-url-heavy
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter


def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

_SENTENCE_ENDERS = re.compile(r"[.!?]")
_HEBREW_RANGE = re.compile(r"[\u0590-\u05FF]")
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+", re.ASCII
)


def _count_sentences(text: str) -> int:
    """Count sentences by terminal punctuation marks."""
    return len(_SENTENCE_ENDERS.findall(text))


def _hebrew_ratio(text: str) -> float:
    """Fraction of characters that are Hebrew (U+0590–U+05FF)."""
    if not text:
        return 0.0
    hebrew_count = len(_HEBREW_RANGE.findall(text))
    # Count only non-whitespace characters as the denominator
    non_ws = sum(1 for c in text if not c.isspace())
    return hebrew_count / non_ws if non_ws > 0 else 0.0


def _sentence_repetition_ratio(text: str) -> float:
    """Fraction of sentences that are duplicates of another sentence.

    Splits on sentence-ending punctuation, normalises whitespace, and
    counts how many sentences appear more than once.
    """
    raw_sentences = _SENTENCE_ENDERS.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0
    counts = Counter(sentences)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(sentences)


def _url_heavy_ratio(text: str) -> float:
    """Fraction of tokens that look like URLs or email addresses."""
    tokens = text.split()
    if not tokens:
        return 0.0
    url_tokens = sum(1 for t in tokens if _URL_PATTERN.search(t))
    return url_tokens / len(tokens)


def _passes_quality_filters(
    text: str,
    min_length: int,
    max_length: int,
    min_sentences: int,
    min_hebrew_ratio: float,
    max_repetition_ratio: float,
    reject_url_heavy: bool,
    url_heavy_threshold: float = 0.2,
) -> tuple[bool, str]:
    """Check whether a document passes all quality filters.

    Returns ``(passed, reason)`` where *reason* describes why the
    document was rejected (empty string if it passed).
    """
    length = len(text.strip())

    if length < min_length:
        return False, "too_short"

    if length > max_length:
        return False, "too_long"

    if _count_sentences(text) < min_sentences:
        return False, "few_sentences"

    hr = _hebrew_ratio(text)
    if hr < min_hebrew_ratio:
        return False, "low_hebrew"

    rr = _sentence_repetition_ratio(text)
    if rr > max_repetition_ratio:
        return False, "repetitive"

    if reject_url_heavy and _url_heavy_ratio(text) > url_heavy_threshold:
        return False, "url_heavy"

    return True, ""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(
    output_dir: str,
    max_docs: int = 500,
    min_length: int = 200,
    max_length: int = 3000,
    min_sentences: int = 3,
    min_hebrew_ratio: float = 0.5,
    max_repetition_ratio: float = 0.3,
    reject_url_heavy: bool = True,
    skip: int = 0,
    seed: int | None = None,
) -> int:
    """Stream HeDC4 from HuggingFace and write documents as .txt files.

    Args:
        output_dir: Directory to write doc_NNNNNNN.txt files.
        max_docs: Maximum number of documents to download.
        min_length: Minimum character length to include a document.
        max_length: Maximum character length to include a document.
        min_sentences: Minimum number of sentences (by terminal punctuation).
        min_hebrew_ratio: Minimum fraction of Hebrew characters.
        max_repetition_ratio: Maximum fraction of repeated sentences.
        reject_url_heavy: Reject documents with >20% URL-like tokens.
        skip: Number of documents to skip before collecting.
        seed: Random seed for shuffling (None = no shuffle, stream in order).

    Returns:
        Number of documents written.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        _die(
            "The 'datasets' library is required. Install with:\n"
            "  pip install datasets"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"[info] Streaming HeDC4 from HuggingFace ...", file=sys.stderr)
    print(f"[info] Filters: length={min_length}–{max_length}, "
          f"sentences≥{min_sentences}, hebrew≥{min_hebrew_ratio:.0%}, "
          f"repetition≤{max_repetition_ratio:.0%}, "
          f"url_filter={'on' if reject_url_heavy else 'off'}",
          file=sys.stderr)

    ds = load_dataset("HeNLP/HeDC4", split="train", streaming=True)

    if seed is not None:
        ds = ds.shuffle(seed=seed)

    doc_count = 0
    skipped_phase = 0
    reject_counts: dict[str, int] = {
        "too_short": 0,
        "too_long": 0,
        "few_sentences": 0,
        "low_hebrew": 0,
        "repetitive": 0,
        "url_heavy": 0,
    }

    for row in ds:
        text = row.get("text") or ""

        # Skip phase
        if skipped_phase < skip:
            skipped_phase += 1
            continue

        # Quality filters
        passed, reason = _passes_quality_filters(
            text,
            min_length=min_length,
            max_length=max_length,
            min_sentences=min_sentences,
            min_hebrew_ratio=min_hebrew_ratio,
            max_repetition_ratio=max_repetition_ratio,
            reject_url_heavy=reject_url_heavy,
        )
        if not passed:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue

        doc_count += 1
        out_path = os.path.join(output_dir, f"doc_{doc_count:07d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

        if doc_count % 100 == 0:
            total_rejected = sum(reject_counts.values())
            print(
                f"[info] Downloaded {doc_count}/{max_docs} "
                f"(rejected {total_rejected} so far)",
                file=sys.stderr,
            )

        if doc_count >= max_docs:
            break

    total_rejected = sum(reject_counts.values())
    print(
        f"[info] Done. {doc_count} documents written to {output_dir}.",
        file=sys.stderr,
    )
    print(f"[info] Rejected {total_rejected} documents:", file=sys.stderr)
    for reason, count in sorted(reject_counts.items()):
        if count > 0:
            print(f"  {reason}: {count}", file=sys.stderr)

    return doc_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a sample of Hebrew documents from the HeDC4 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output directory for .txt files (created if it doesn't exist).",
    )
    parser.add_argument(
        "--max-docs", "-n",
        type=int,
        default=500,
        metavar="N",
        help="Maximum number of documents to download (default: 500).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        metavar="CHARS",
        help="Minimum character length to include a document (default: 200).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3000,
        metavar="CHARS",
        help="Maximum character length to include a document (default: 3000).",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=3,
        metavar="N",
        help="Minimum number of sentences by terminal punctuation (default: 3).",
    )
    parser.add_argument(
        "--min-hebrew-ratio",
        type=float,
        default=0.5,
        metavar="F",
        help="Minimum fraction of Hebrew characters (default: 0.5).",
    )
    parser.add_argument(
        "--max-repetition-ratio",
        type=float,
        default=0.3,
        metavar="F",
        help="Maximum fraction of repeated sentences (default: 0.3).",
    )
    parser.add_argument(
        "--allow-url-heavy",
        action="store_true",
        help="Allow documents with >20%% URL-like tokens (default: reject them).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N documents before collecting (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for shuffling. Omit for sequential streaming.",
    )

    args = parser.parse_args()
    download(
        output_dir=args.output,
        max_docs=args.max_docs,
        min_length=args.min_length,
        max_length=args.max_length,
        min_sentences=args.min_sentences,
        min_hebrew_ratio=args.min_hebrew_ratio,
        max_repetition_ratio=args.max_repetition_ratio,
        reject_url_heavy=not args.allow_url_heavy,
        skip=args.skip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
