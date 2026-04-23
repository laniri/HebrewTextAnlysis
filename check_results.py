#!/usr/bin/env python3
"""Quick health check for a batch results directory.

Reports how many documents have complete features vs missing layers,
and identifies the first document where YAP/Stanza started failing.

Usage:
    python check_results.py results_hedc4/
    python check_results.py results_sample/ results_hedc4/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_dir(results_dir: Path) -> None:
    files = sorted(results_dir.glob("doc_*.json"))
    if not files:
        print(f"  No doc_*.json files found in {results_dir}")
        return

    total = 0
    stanza_ok = 0
    yap_ok = 0
    both_ok = 0
    first_yap_fail: str | None = None
    first_stanza_fail: str | None = None

    for f in files:
        total += 1
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        features = doc.get("features", {})
        morph = features.get("morphology", {})
        syntax = features.get("syntax", {})

        has_stanza = morph.get("verb_ratio") is not None
        has_yap = syntax.get("avg_sentence_length") is not None

        if has_stanza:
            stanza_ok += 1
        elif first_stanza_fail is None:
            first_stanza_fail = f.name

        if has_yap:
            yap_ok += 1
        elif first_yap_fail is None:
            first_yap_fail = f.name

        if has_stanza and has_yap:
            both_ok += 1

    stanza_fail = total - stanza_ok
    yap_fail = total - yap_ok

    print(f"  {results_dir}/")
    print(f"    Total documents:    {total}")
    print(f"    Full features:      {both_ok} ({both_ok/total*100:.1f}%)")
    print(f"    Stanza OK:          {stanza_ok} ({stanza_ok/total*100:.1f}%)")
    print(f"    YAP OK:             {yap_ok} ({yap_ok/total*100:.1f}%)")
    if stanza_fail > 0:
        print(f"    Stanza failures:    {stanza_fail} — first at {first_stanza_fail}")
    if yap_fail > 0:
        print(f"    YAP failures:       {yap_fail} — first at {first_yap_fail}")
    if yap_fail == 0 and stanza_fail == 0:
        print(f"    ✓ All documents have complete features")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python check_results.py DIR [DIR ...]")
        sys.exit(1)

    print("Batch results health check:")
    for dir_path in sys.argv[1:]:
        d = Path(dir_path)
        if not d.is_dir():
            print(f"  {dir_path}: not a directory, skipping")
            continue
        check_dir(d)
    print()


if __name__ == "__main__":
    main()
