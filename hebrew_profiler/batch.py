"""Batch processor for the Hebrew Linguistic Profiling Engine.

Discovers text files in an input directory, processes each through the full
pipeline using a multiprocessing pool, writes individual JSON result files,
and optionally produces a single JSONL export.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from hebrew_profiler.models import BatchResult, PipelineConfig
from hebrew_profiler.pipeline import pipeline_output_to_dict, process_document


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_error(document_id: str, error_type: str, message: str) -> None:
    """Log an error to stderr in the required format.

    Format: [{ISO-8601 timestamp}] ERROR [{document_id}] {error_type}: {message}
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    print(
        f"[{ts}] ERROR [{document_id}] {error_type}: {message}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Single-file worker (top-level so it is picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _process_single_file(args: tuple[str, str, PipelineConfig, bool]) -> dict[str, Any]:
    """Process one text file through the pipeline.

    Returns a dict with keys:
        success (bool), document_id (str),
        result_dict (dict | None), raw_text (str | None),
        normalized_text (str | None),
        error_type (str | None), error_message (str | None),
        yap_failed (bool)
    """
    input_path, output_dir, config, strict = args
    document_id = input_path

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] PROCESSING [{document_id}]", file=sys.stderr)

    # --- read file with UTF-8 validation ---
    try:
        text = Path(input_path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _log_error(document_id, "EncodingError", str(exc))
        return {
            "success": False,
            "document_id": document_id,
            "result_dict": None,
            "raw_text": None,
            "normalized_text": None,
            "error_type": "EncodingError",
            "error_message": str(exc),
            "yap_failed": False,
        }
    except OSError as exc:
        _log_error(document_id, "IOError", str(exc))
        return {
            "success": False,
            "document_id": document_id,
            "result_dict": None,
            "raw_text": None,
            "normalized_text": None,
            "error_type": "IOError",
            "error_message": str(exc),
            "yap_failed": False,
        }

    # --- run pipeline ---
    try:
        output = process_document(text, config)
    except Exception as exc:
        error_type = type(exc).__name__
        _log_error(document_id, error_type, str(exc))
        return {
            "success": False,
            "document_id": document_id,
            "result_dict": None,
            "raw_text": text,
            "normalized_text": None,
            "error_type": error_type,
            "error_message": str(exc),
            "yap_failed": False,
        }

    # --- serialise result ---
    result_dict = pipeline_output_to_dict(output)

    # --- check for YAP failure in strict mode ---
    yap_failed = result_dict["features"]["syntax"]["avg_tree_depth"] is None
    if strict and yap_failed:
        # Try to determine if YAP is still alive by pinging it
        import requests
        yap_alive = False
        try:
            # Quick health check — send a tiny request
            test_resp = requests.get(
                config.yap_url.rsplit("/", 1)[0] if "/" in config.yap_url else config.yap_url,
                timeout=5,
            )
            yap_alive = True
        except Exception:
            yap_alive = False

        if yap_alive:
            # YAP is alive but this specific document caused a failure
            # (likely a sentence that crashed YAP). Skip this document.
            msg = (
                f"YAP returned null syntax for {document_id} but is still responsive. "
                f"Skipping this document (likely contains a sentence that crashes YAP)."
            )
            _log_error(document_id, "YAPDocumentSkipped", msg)
            return {
                "success": False,
                "document_id": document_id,
                "result_dict": None,
                "raw_text": text,
                "normalized_text": None,
                "error_type": "YAPDocumentSkipped",
                "error_message": msg,
                "yap_failed": False,  # Not a fatal YAP failure — just a bad document
            }
        else:
            # YAP is truly down
            msg = (
                f"YAP produced null syntax features for {document_id} and is not responsive at {config.yap_url}. "
                f"Check that YAP is running: curl {config.yap_url} "
                f"or start it with: ./yapproj/src/yap/yap api"
            )
            _log_error(document_id, "YAPUnresponsive", msg)
            return {
                "success": False,
                "document_id": document_id,
                "result_dict": None,
                "raw_text": text,
                "normalized_text": None,
                "error_type": "YAPUnresponsive",
                "error_message": msg,
                "yap_failed": True,
            }

    # --- write individual JSON file ---
    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"{stem}.json"
    indent = 2 if config.pretty_output else None
    out_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )

    # PipelineOutput doesn't carry normalized_text directly, so re-derive it.
    from hebrew_profiler.normalizer import normalize as _normalize
    norm_result = _normalize(text)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] DONE [{document_id}]", file=sys.stderr)

    return {
        "success": True,
        "document_id": document_id,
        "result_dict": result_dict,
        "raw_text": text,
        "normalized_text": norm_result.normalized_text,
        "error_type": None,
        "error_message": None,
        "yap_failed": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_batch(
    input_dir: str,
    output_dir: str,
    config: PipelineConfig,
    workers: int = 4,
    jsonl_path: str | None = None,
    strict: bool = False,
) -> BatchResult:
    """Discover text files, process in parallel, write JSON results.

    Args:
        input_dir:  Path to directory containing ``.txt`` files.
        output_dir: Path to directory where per-document JSON files are written.
        config:     Pipeline configuration.
        workers:    Number of multiprocessing workers.
        jsonl_path: If provided, all results are also written to this JSONL file.
        strict:     If True, abort when YAP is unresponsive instead of
                    producing results with null syntax features.

    Returns:
        BatchResult with total_processed, error_count, and per-document errors.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Discover text files
    input_files = sorted(
        str(p) for p in Path(input_dir).iterdir() if p.is_file() and p.suffix == ".txt"
    )

    if not input_files:
        return BatchResult(total_processed=0, error_count=0, errors=[])

    # Skip files that already have results in the output directory
    out_dir = Path(output_dir)
    before = len(input_files)
    input_files = [
        f for f in input_files
        if not (out_dir / f"{Path(f).stem}.json").exists()
    ]
    skipped_existing = before - len(input_files)
    if skipped_existing > 0:
        print(
            f"[info] Skipping {skipped_existing} files with existing results "
            f"in {output_dir}",
            file=sys.stderr,
        )

    if not input_files:
        print(
            f"[info] All {before} files already have results. Nothing to do.",
            file=sys.stderr,
        )
        return BatchResult(total_processed=0, error_count=0, errors=[])

    # Build argument tuples for the worker
    task_args = [(f, output_dir, config, strict) for f in input_files]

    # In strict mode, process sequentially so we can abort immediately
    # on YAP failure instead of waiting for all parallel workers to finish.
    effective_workers = 1 if strict else workers

    # Process — parallel when workers > 1, sequential otherwise
    if effective_workers > 1:
        with Pool(processes=effective_workers) as pool:
            results = pool.map(_process_single_file, task_args)
    else:
        # Sequential — allows immediate abort on YAP failure
        results: list[dict[str, Any]] = []
        for i, a in enumerate(task_args):
            r = _process_single_file(a)
            results.append(r)
            if r.get("yap_failed"):
                print(
                    f"\n[FATAL] Aborting batch: YAP is not responsive.\n"
                    f"  Document: {r['document_id']}\n"
                    f"  Error: {r['error_message']}\n"
                    f"  Processed {i + 1}/{len(task_args)} files "
                    f"({sum(1 for x in results if x['success'])} succeeded).\n"
                    f"  Results written so far are in {output_dir}/\n"
                    f"  Re-run with the same command to resume from where it stopped.\n",
                    file=sys.stderr,
                )
                return BatchResult(
                    total_processed=i + 1,
                    error_count=sum(1 for x in results if not x["success"]),
                    errors=[
                        {"document": x["document_id"], "error_type": x["error_type"], "message": x["error_message"]}
                        for x in results if not x["success"]
                    ],
                )
            if (i + 1) % 50 == 0:
                print(
                    f"[info] Processed {i + 1}/{len(task_args)} files ...",
                    file=sys.stderr,
                )

    # Aggregate results
    errors: list[dict] = []
    total_processed = 0
    error_count = 0
    jsonl_records: list[dict] = []

    for r in results:
        total_processed += 1
        if not r["success"]:
            error_count += 1
            errors.append({
                "document": r["document_id"],
                "error_type": r["error_type"],
                "message": r["error_message"],
            })
        else:
            # Collect JSONL record for successful documents
            if jsonl_path is not None:
                jsonl_records.append({
                    "raw_text": r["raw_text"],
                    "normalized_text": r["normalized_text"],
                    "features": r["result_dict"]["features"],
                    "scores": r["result_dict"]["scores"],
                })

    # Write JSONL if requested
    if jsonl_path is not None:
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for record in jsonl_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return BatchResult(
        total_processed=total_processed,
        error_count=error_count,
        errors=errors,
    )
