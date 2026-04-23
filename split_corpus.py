#!/usr/bin/env python3
"""Split wikipedia.raw into document .txt files for batch processing.

The AlephBERT wikipedia.raw file is one sentence per line with no article
boundaries. This script groups consecutive sentences into documents of a
configurable size and writes each as a separate .txt file.

It can also build or load a Hebrew lemma frequency dictionary from the raw
corpus, which enables rare_word_ratio computation in the pipeline.

Usage:
    # Split corpus only
    python split_corpus.py wikipedia.raw corpus/ [--sentences-per-doc 20] [--max-docs 0]

    # Split corpus AND build a frequency dictionary from it
    python split_corpus.py wikipedia.raw corpus/ --build-freq-dict freq_dict.json

    # Split corpus and load an existing frequency dictionary (validates it loads)
    python split_corpus.py wikipedia.raw corpus/ --freq-dict freq_dict.json

    # Build frequency dictionary only (no splitting)
    python split_corpus.py wikipedia.raw --build-freq-dict freq_dict.json --no-split
"""

import argparse
import json
import os
import sys
from collections import Counter


def split(input_path: str, output_dir: str, sentences_per_doc: int, max_docs: int) -> int:
    os.makedirs(output_dir, exist_ok=True)

    doc_id = 0
    buffer = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            buffer.append(stripped)

            if len(buffer) >= sentences_per_doc:
                doc_id += 1
                out_path = os.path.join(output_dir, f"doc_{doc_id:07d}.txt")
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write("\n".join(buffer) + "\n")
                buffer = []

                if max_docs > 0 and doc_id >= max_docs:
                    break

    # Flush remaining sentences
    if buffer and (max_docs == 0 or doc_id < max_docs):
        doc_id += 1
        out_path = os.path.join(output_dir, f"doc_{doc_id:07d}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(buffer) + "\n")

    return doc_id


def build_freq_dict(input_path: str, output_path: str) -> dict[str, int]:
    """Build a word frequency dictionary from the raw corpus.

    Counts every whitespace-separated token in the raw file as a lemma proxy.

    Args:
        input_path: Path to the raw corpus file (one sentence per line).
        output_path: Path to write the resulting JSON frequency dictionary.

    Returns:
        The frequency dictionary {token: count}.
    """
    print(f"Building frequency dictionary from {input_path}...")
    counts: Counter[str] = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            counts.update(tokens)

    freq_dict = dict(counts)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=False)

    total_tokens = sum(freq_dict.values())
    unique_tokens = len(freq_dict)
    print(f"  {total_tokens:,} total tokens, {unique_tokens:,} unique types → {output_path}")
    return freq_dict


def build_freq_dict_from_dirs(corpus_dirs: list[str], output_path: str) -> dict[str, int]:
    """Build a word frequency dictionary from multiple directories of .txt files.

    Reads all .txt files from each directory and counts whitespace-separated
    tokens across all of them.

    Args:
        corpus_dirs: List of directory paths containing .txt files.
        output_path: Path to write the resulting JSON frequency dictionary.

    Returns:
        The frequency dictionary {token: count}.
    """
    counts: Counter[str] = Counter()
    total_files = 0

    for dir_path in corpus_dirs:
        txt_files = sorted(
            f for f in os.listdir(dir_path)
            if f.endswith(".txt")
        )
        print(f"  Scanning {dir_path}: {len(txt_files)} .txt files")
        for fname in txt_files:
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    counts.update(tokens)
            total_files += 1

    freq_dict = dict(counts)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=False)

    total_tokens = sum(freq_dict.values())
    unique_tokens = len(freq_dict)
    print(
        f"Built frequency dictionary from {total_files} files across {len(corpus_dirs)} directories"
    )
    print(f"  {total_tokens:,} total tokens, {unique_tokens:,} unique types → {output_path}")
    return freq_dict


def load_freq_dict(path: str) -> dict[str, int]:
    """Load and validate an existing frequency dictionary JSON file.

    Args:
        path: Path to the JSON frequency dictionary.

    Returns:
        The frequency dictionary {lemma: count}.

    Raises:
        SystemExit: If the file cannot be read or is not a valid dict.
    """
    print(f"Loading frequency dictionary from {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error loading frequency dictionary: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Error: {path} must be a JSON object (dict), got {type(data).__name__}",
              file=sys.stderr)
        sys.exit(1)

    total = sum(data.values()) if data else 0
    print(f"  Loaded {len(data):,} entries, {total:,} total occurrences")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Split wikipedia.raw into document files and optionally manage "
                    "a Hebrew frequency dictionary for rare_word_ratio computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to wikipedia.raw")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output directory for .txt files (omit with --no-split)",
    )
    parser.add_argument(
        "--sentences-per-doc", type=int, default=20,
        help="Sentences per document (default: 20)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Max documents to create (0 = all, default: 0)",
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Skip corpus splitting (useful when only building a freq dict)",
    )

    # Frequency dictionary options (mutually exclusive)
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--build-freq-dict", metavar="PATH",
        help="Build a frequency dictionary from the raw corpus and save to PATH",
    )
    freq_group.add_argument(
        "--freq-dict", metavar="PATH",
        help="Load and validate an existing frequency dictionary from PATH",
    )
    freq_group.add_argument(
        "--build-freq-dict-from-dirs", nargs="+", metavar=("OUTPUT", "DIR"),
        help=(
            "Build a frequency dictionary from multiple .txt corpus directories. "
            "First argument is the output JSON path, remaining are corpus directories. "
            "Example: --build-freq-dict-from-dirs freq_dict.json corpus_sample/ corpus_hedc4/"
        ),
    )

    args = parser.parse_args()

    # Validate: need output dir unless --no-split
    if not args.no_split and args.output is None:
        parser.error("output directory is required unless --no-split is specified")

    # Handle frequency dictionary
    if args.build_freq_dict:
        build_freq_dict(args.input, args.build_freq_dict)
    elif args.freq_dict:
        load_freq_dict(args.freq_dict)
    elif args.build_freq_dict_from_dirs:
        dirs_args = args.build_freq_dict_from_dirs
        if len(dirs_args) < 2:
            parser.error("--build-freq-dict-from-dirs requires OUTPUT_PATH DIR [DIR ...]")
        output_path = dirs_args[0]
        corpus_dirs = dirs_args[1:]
        build_freq_dict_from_dirs(corpus_dirs, output_path)

    # Handle corpus splitting
    if not args.no_split:
        count = split(args.input, args.output, args.sentences_per_doc, args.max_docs)
        print(f"Created {count} documents in {args.output}")


if __name__ == "__main__":
    main()
