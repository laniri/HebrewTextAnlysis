"""YAP adapter for syntactic parsing via the YAP API.

Refactored from analyze_hebrew.py. Sends Hebrew text to the YAP API,
parses CoNLL output into structured dataclasses, and segments
multi-sentence output into per-sentence dependency trees.

Text is pre-split into sentences before sending to YAP so that each
sentence gets its own dependency tree (YAP does not reliably segment
multi-sentence paragraphs on its own).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time

import requests

from hebrew_profiler.errors import YAPConnectionError, YAPHTTPError
from hebrew_profiler.models import DepTreeNode, SentenceTree, YAPError, YAPResult


# ---------------------------------------------------------------------------
# YAP Server Manager
# ---------------------------------------------------------------------------

class YAPServerManager:
    """Manages the YAP API server process lifecycle.

    Can start, stop, and restart the YAP server. Used by parse_syntax
    to auto-recover when YAP crashes during batch processing.

    Set the YAP binary path via:
    - Constructor argument: YAPServerManager(yap_bin="/path/to/yap")
    - Environment variable: YAP_BIN=/path/to/yap
    - Default: searches for "yap" in PATH
    """

    def __init__(
        self,
        yap_bin: str | None = None,
        port: int = 8000,
        startup_timeout: int = 120,
    ) -> None:
        self.yap_bin = yap_bin or os.environ.get("YAP_BIN", "yap")
        self.port = port
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None

    @property
    def yap_url(self) -> str:
        return f"http://localhost:{self.port}/yap/heb/joint"

    def _kill_stale_port_holder(self) -> None:
        """Kill any process holding the YAP port from a previous session."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid.isdigit():
                    print(f"[yap-mgr] Killing stale process {pid} on port {self.port}", file=sys.stderr)
                    os.kill(int(pid), signal.SIGTERM)
            if pids:
                time.sleep(2)  # give it time to release the port
        except Exception:
            pass  # lsof not available or no process found — fine

    def is_alive(self) -> bool:
        """Check if YAP is responding to requests."""
        try:
            resp = requests.get(
                self.yap_url,
                data=json.dumps({"text": "בדיקה  "}),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            return resp.ok
        except Exception:
            return False

    def start(self) -> bool:
        """Start the YAP server. Returns True if it becomes responsive."""
        if self.is_alive():
            print("[yap-mgr] YAP is already running.", file=sys.stderr)
            return True

        # Kill any stale process on the port
        self._kill_stale_port_holder()

        print(f"[yap-mgr] Starting YAP server: {self.yap_bin} api", file=sys.stderr)
        try:
            self._process = subprocess.Popen(
                [self.yap_bin, "api"],
                stdout=sys.stderr,   # show YAP output so we can see model loading
                stderr=sys.stderr,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            print(
                f"[yap-mgr] YAP binary not found: {self.yap_bin}. "
                f"Set YAP_BIN env var or pass --yap-bin.",
                file=sys.stderr,
            )
            return False
        except Exception as exc:
            print(f"[yap-mgr] Failed to start YAP: {exc}", file=sys.stderr)
            return False

        # Wait for YAP to become responsive — it needs time to load models
        for waited in range(0, self.startup_timeout, 3):
            time.sleep(3)
            # Check if process died
            if self._process.poll() is not None:
                print(
                    f"[yap-mgr] YAP process exited with code {self._process.returncode}",
                    file=sys.stderr,
                )
                self._process = None
                return False
            if self.is_alive():
                print(
                    f"[yap-mgr] YAP server ready (pid={self._process.pid}, "
                    f"port={self.port}, {waited+3}s)",
                    file=sys.stderr,
                )
                return True
            print(
                f"[yap-mgr] Waiting for YAP to load models... ({waited+3}/{self.startup_timeout}s)",
                file=sys.stderr,
            )

        print(
            f"[yap-mgr] YAP did not become responsive within {self.startup_timeout}s",
            file=sys.stderr,
        )
        self.stop()
        return False

    def stop(self) -> None:
        """Stop the managed YAP server process."""
        if self._process is None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            self._process.wait(timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        self._process = None
        print("[yap-mgr] YAP server stopped.", file=sys.stderr)

    def restart(self) -> bool:
        """Stop and restart the YAP server. Returns True if successful."""
        print("[yap-mgr] Restarting YAP server...", file=sys.stderr)
        self.stop()
        time.sleep(1)
        return self.start()


# Module-level singleton — set by callers who want auto-restart
_yap_manager: YAPServerManager | None = None


def set_yap_manager(manager: YAPServerManager) -> None:
    """Register a YAP server manager for auto-restart on crash."""
    global _yap_manager
    _yap_manager = manager


def get_yap_manager() -> YAPServerManager | None:
    """Return the registered YAP server manager, or None."""
    return _yap_manager


# Regex to split Hebrew text on sentence-ending punctuation or newlines.
# Splits on:
#   - period, question mark, exclamation mark, or Hebrew sof-pasuq (׃)
#     followed by whitespace or end-of-string (delimiter kept on preceding segment), OR
#   - one or more newlines — split position is AFTER the newline so that lines
#     ending with punctuation keep it attached (e.g. "...מעסיקו.\n" → "...מעסיקו.")
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?׃])\s+|(?<=\n)\n*|(?<!\S)\n+')


def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences on sentence-ending punctuation or newlines.

    Returns a list of non-empty sentence strings. If no sentence boundaries
    are found, returns the original text as a single-element list.
    Filters out sentences with no whitespace (concatenated text that crashes YAP).
    """
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [
        s.strip() for s in parts
        if s.strip() and " " in s.strip()  # must have at least one space
    ]


def _parse_features(feat_str: str) -> dict[str, str]:
    """Parse a pipe-separated feature string like 'gen=M|num=S' into a dict."""
    if not feat_str or feat_str == "_":
        return {}
    result: dict[str, str] = {}
    for pair in feat_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _sanitize_for_yap(text: str) -> str:
    """Clean text before sending to YAP to avoid known crash triggers.

    YAP's Go parser panics on:
    - Tab characters (conflicts with its tab-separated lattice format)
    - Certain quoted strings with embedded punctuation that produce
      empty CPOSTAG fields in the morphological lattice

    This function replaces known problematic patterns with safe
    alternatives.
    """
    # Replace tabs with spaces — YAP uses TSV internally
    text = text.replace("\t", " ")
    # Collapse multiple spaces into one
    import re
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _call_yap_api(text: str, yap_url: str) -> dict:
    """Send text to YAP API and return raw JSON response.

    Appends two trailing spaces per YAP protocol.
    Sanitizes input to avoid known YAP crash triggers.
    Raises YAPConnectionError if the API is unreachable or the connection drops.
    Raises YAPHTTPError if the API returns a non-2xx status.
    """
    clean_text = _sanitize_for_yap(text)
    payload = json.dumps({"text": f"{clean_text}  "})
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.get(yap_url, data=payload, headers=headers, timeout=30)
    except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
        raise YAPConnectionError(
            f"Connection error to {yap_url}: {exc}"
        ) from exc
    except requests.exceptions.ReadTimeout as exc:
        raise YAPConnectionError(
            f"YAP timed out (30s) — sentence may be too long or malformed: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise YAPConnectionError(
            f"Request failed to {yap_url}: {exc}"
        ) from exc

    if not resp.ok:
        raise YAPHTTPError(
            http_status=resp.status_code,
            message=f"YAP API returned HTTP {resp.status_code}",
        )

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise YAPConnectionError(
            f"YAP returned invalid JSON (possible server panic): {exc}"
        ) from exc


def _parse_lattice(raw: str) -> list[dict]:
    """Parse a lattice string (MA or MD) into structured records."""
    records: list[dict] = []
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
            "features": _parse_features(parts[6]),
            "token_id": int(parts[7]),
        })
    return records


def _parse_dep_tree(raw: str) -> list[DepTreeNode]:
    """Parse a CoNLL dependency tree string into DepTreeNode objects."""
    nodes: list[DepTreeNode] = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        nodes.append(DepTreeNode(
            id=int(parts[0]),
            form=parts[1],
            lemma=parts[2],
            cpostag=parts[3],
            postag=parts[4],
            features=_parse_features(parts[5]),
            head=int(parts[6]),
            deprel=parts[7],
        ))
    return nodes


def _segment_sentences(raw: str) -> list[SentenceTree]:
    """Segment multi-sentence CoNLL output on blank lines into SentenceTree objects."""
    sentences: list[SentenceTree] = []
    current_nodes: list[DepTreeNode] = []

    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            # Blank line marks sentence boundary
            if current_nodes:
                sentences.append(SentenceTree(nodes=current_nodes))
                current_nodes = []
            continue
        parts = stripped.split("\t")
        if len(parts) < 8:
            continue
        current_nodes.append(DepTreeNode(
            id=int(parts[0]),
            form=parts[1],
            lemma=parts[2],
            cpostag=parts[3],
            postag=parts[4],
            features=_parse_features(parts[5]),
            head=int(parts[6]),
            deprel=parts[7],
        ))

    # Flush any remaining nodes (last sentence may not end with blank line)
    if current_nodes:
        sentences.append(SentenceTree(nodes=current_nodes))

    return sentences


def _compute_ambiguity_counts(ma_lattice: list[dict]) -> list[int]:
    """Count the number of MA lattice analyses per token_id.

    The MA lattice contains all candidate morphological analyses before
    disambiguation. Multiple rows share the same token_id — the count
    per token_id is the morphological ambiguity for that token.

    Returns a list of counts ordered by token_id (1-based in YAP, but
    the returned list is 0-indexed for easy alignment).
    """
    if not ma_lattice:
        return []

    counts: dict[int, int] = {}
    for record in ma_lattice:
        tid = record.get("token_id", 0)
        counts[tid] = counts.get(tid, 0) + 1

    if not counts:
        return []

    max_tid = max(counts.keys())
    return [counts.get(tid, 1) for tid in range(1, max_tid + 1)]


def _wait_for_yap(yap_url: str, max_wait: int = 60, interval: int = 5) -> bool:
    """Wait for YAP to become responsive again after a crash.

    If a YAPServerManager is registered, attempts to restart YAP automatically.
    Otherwise, pings the URL every `interval` seconds hoping for external recovery.
    Returns True if YAP responds within `max_wait` seconds, False otherwise.
    """
    manager = get_yap_manager()

    # Try auto-restart first
    if manager is not None:
        print("[yap] Attempting auto-restart via YAPServerManager...", file=sys.stderr)
        if manager.restart():
            return True
        print("[yap] Auto-restart failed. Waiting for manual recovery...", file=sys.stderr)

    # Fall back to polling
    waited = 0
    while waited < max_wait:
        try:
            resp = requests.get(
                yap_url,
                data=json.dumps({"text": "בדיקה  "}),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.ok:
                print("[yap] YAP recovered.", file=sys.stderr)
                return True
        except Exception:
            pass
        print(
            f"[yap] Waiting for YAP to recover... ({waited}/{max_wait}s)",
            file=sys.stderr,
        )
        time.sleep(interval)
        waited += interval
    return False


def parse_syntax(text: str, yap_url: str) -> YAPResult | YAPError:
    """Send text to YAP API, return disambiguated morphology and dependency trees.

    Pre-splits the input into sentences so that each sentence gets its own
    dependency tree. This avoids YAP treating entire paragraphs as single
    sentences, which produces unrealistically deep trees.

    If a single sentence fails, it is skipped and the remaining sentences
    are still processed. If 3+ consecutive sentences fail (YAP is down),
    waits up to 60 seconds for YAP to recover before giving up.

    Returns YAPResult on success, YAPError on failure.
    """
    sentences_text = _split_sentences(text)
    if not sentences_text:
        return YAPResult(morphological_disambiguation=[], sentences=[], ambiguity_counts=[])

    all_md: list[dict] = []
    all_sentences: list[SentenceTree] = []
    all_ambiguity: list[int] = []
    consecutive_failures = 0

    for sent_text in sentences_text:
        try:
            raw = _call_yap_api(sent_text, yap_url)
            consecutive_failures = 0  # reset on success
        except (YAPConnectionError, YAPHTTPError) as exc:
            consecutive_failures += 1

            if consecutive_failures >= 3:
                # YAP appears to be down — wait for it to recover
                print(
                    f"[yap] 3 consecutive failures — waiting for YAP to recover...",
                    file=sys.stderr,
                )
                if _wait_for_yap(yap_url):
                    # YAP is back — retry this sentence
                    consecutive_failures = 0
                    try:
                        raw = _call_yap_api(sent_text, yap_url)
                    except (YAPConnectionError, YAPHTTPError):
                        continue  # still failing, skip this sentence
                else:
                    # YAP didn't recover — give up for this document
                    return YAPError(
                        error_type="YAPConnectionError",
                        http_status=None,
                        message=f"YAP did not recover after 60s wait: {exc}",
                    )
            else:
                # Skip this sentence, try the next
                continue

        # Parse MA lattice for ambiguity counts (before disambiguation)
        ma_lattice = _parse_lattice(raw.get("ma_lattice", ""))
        all_ambiguity.extend(_compute_ambiguity_counts(ma_lattice))

        all_md.extend(_parse_lattice(raw.get("md_lattice", "")))
        dep_tree_raw = raw.get("dep_tree", "")
        all_sentences.extend(_segment_sentences(dep_tree_raw))

    return YAPResult(
        morphological_disambiguation=all_md,
        sentences=all_sentences,
        ambiguity_counts=all_ambiguity,
    )
