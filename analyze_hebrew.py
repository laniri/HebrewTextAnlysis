#!/usr/bin/env python3
"""
Hebrew text analyzer using YAP (Yet Another Parser).
Reads a Hebrew text file, sends it to the YAP API for morphological analysis,
disambiguation and dependency parsing, and outputs structured JSON results.

Requires YAP API server running: ./yap api (default port 8000)
"""

import argparse
import json
import sys
import requests


DEFAULT_YAP_URL = "http://localhost:8000/yap/heb/joint"


def read_input_file(filepath: str) -> str:
    """Read Hebrew text from a UTF-8 file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_yap_api(text: str, yap_url: str) -> dict:
    """Send text to YAP API and return raw JSON response."""
    # YAP expects input to end with two spaces
    payload = json.dumps({"text": f"{text}  "})
    headers = {"Content-Type": "application/json"}
    resp = requests.get(yap_url, data=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_lattice(raw: str) -> list[dict]:
    """Parse a lattice string (MA or MD) into structured records."""
    records = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        records.append({
            "from": int(parts[0]),
            "to": int(parts[1]),
            "form": parts[2],
            "lemma": parts[3],
            "cpostag": parts[4],
            "postag": parts[5],
            "features": parse_features(parts[6]),
            "token_id": int(parts[7]),
        })
    return records


def parse_dep_tree(raw: str) -> list[dict]:
    """Parse a CoNLL dependency tree string into structured records."""
    records = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        records.append({
            "id": int(parts[0]),
            "form": parts[1],
            "lemma": parts[2],
            "cpostag": parts[3],
            "postag": parts[4],
            "features": parse_features(parts[5]),
            "head": int(parts[6]),
            "deprel": parts[7],
        })
    return records


def parse_features(feat_str: str) -> dict:
    """Parse a pipe-separated feature string like 'gen=M|num=S' into a dict."""
    if not feat_str or feat_str == "_":
        return {}
    result = {}
    for pair in feat_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def analyze(filepath: str, yap_url: str) -> dict:
    """Run full analysis pipeline and return structured JSON."""
    text = read_input_file(filepath)
    raw = call_yap_api(text, yap_url)

    return {
        "input_text": text,
        "morphological_analysis": parse_lattice(raw.get("ma_lattice", "")),
        "morphological_disambiguation": parse_lattice(raw.get("md_lattice", "")),
        "dependency_tree": parse_dep_tree(raw.get("dep_tree", "")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Hebrew text using YAP and output JSON results."
    )
    parser.add_argument("input", help="Path to Hebrew text file (UTF-8)")
    parser.add_argument(
        "-o", "--output", help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--yap-url",
        default=DEFAULT_YAP_URL,
        help=f"YAP API URL (default: {DEFAULT_YAP_URL})",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    args = parser.parse_args()

    try:
        result = analyze(args.input, args.yap_url)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except requests.ConnectionError:
        print(
            "Error: cannot connect to YAP API. Is the server running?\n"
            "  Start it with: ./yapproj/src/yap/yap api",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"Error: YAP API returned {e}", file=sys.stderr)
        sys.exit(1)

    indent = 2 if args.pretty else None
    output_json = json.dumps(result, ensure_ascii=False, indent=indent)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json + "\n")
        print(f"Results written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
