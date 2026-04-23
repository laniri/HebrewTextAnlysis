"""ExampleService — loads and serves pre-built example texts.

Scans ``app/data/examples/`` for ``.json`` files at startup and keeps them
in memory for fast retrieval.  Each JSON file is expected to contain::

    {"id": "...", "label": "...", "category": "...", "text": "...", "preview": "..."}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.models.schemas import ExampleFull, ExampleSummary

logger = logging.getLogger(__name__)

# Resolve the examples directory relative to *this* module so it works
# regardless of the working directory the process was started from.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "examples"


class ExampleService:
    """Loads example text JSON files and provides lookup helpers."""

    def __init__(self, examples_dir: Path | None = None) -> None:
        self._examples_dir = examples_dir or _EXAMPLES_DIR
        self._examples: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Scan the examples directory for ``.json`` files and load them."""
        if not self._examples_dir.is_dir():
            logger.warning(
                "Examples directory not found: %s — starting with no examples",
                self._examples_dir,
            )
            return

        for path in sorted(self._examples_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                example_id = data.get("id")
                if not example_id:
                    logger.warning("Skipping %s — missing 'id' field", path.name)
                    continue
                self._examples[example_id] = data
                logger.info("Loaded example: %s (%s)", example_id, path.name)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", path.name, exc)

        logger.info("ExampleService loaded %d example(s)", len(self._examples))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_examples(self) -> list[ExampleSummary]:
        """Return id, label, category, and preview for every loaded example."""
        return [
            ExampleSummary(
                id=ex["id"],
                label=ex["label"],
                category=ex["category"],
                preview=ex["preview"],
            )
            for ex in self._examples.values()
        ]

    def get_example(self, example_id: str) -> ExampleFull | None:
        """Return the full example for *example_id*, or ``None`` if not found."""
        ex = self._examples.get(example_id)
        if ex is None:
            return None
        return ExampleFull(
            id=ex["id"],
            label=ex["label"],
            category=ex["category"],
            text=ex["text"],
        )
