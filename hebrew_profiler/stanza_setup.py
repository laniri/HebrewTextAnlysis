"""Stanza NLP library setup and Hebrew model verification.

Handles dependency checking, Hebrew language model availability verification,
and cached pipeline initialization for the Stanza NLP library.
"""

from __future__ import annotations

import os
from typing import Any

from hebrew_profiler.errors import StanzaSetupError

STANZA_INSTALL_INSTRUCTIONS = """
Stanza Hebrew NLP requires:
  1. Install the stanza package: pip install stanza (declared in requirements.txt)
  2. Download the Hebrew language model:
       python -c "import stanza; stanza.download('he')"
  3. No additional system dependencies required (pure Python).
"""

# Module-level cache for the Stanza pipeline instance.
_cached_pipeline: Any = None


def check_stanza_model(lang: str = "he") -> bool:
    """Verify that the Stanza Hebrew language model is downloaded and available.

    Returns ``True`` if the model directory exists on disk.

    Raises:
        StanzaSetupError: If stanza is not installed or the Hebrew model
            has not been downloaded.  The error message includes full
            installation / download instructions.
    """
    try:
        import stanza  # noqa: F811
        from stanza.resources.common import DEFAULT_MODEL_DIR
    except ImportError:
        raise StanzaSetupError(
            f"Stanza package is not installed. "
            f"Cannot load the '{lang}' language model."
            f"{STANZA_INSTALL_INSTRUCTIONS}"
        )

    # Stanza stores models under <DEFAULT_MODEL_DIR>/<lang>/
    model_dir = os.path.join(DEFAULT_MODEL_DIR, lang)
    if not os.path.isdir(model_dir):
        raise StanzaSetupError(
            f"Stanza Hebrew language model ('{lang}') not found at "
            f"{model_dir}. Please download it first."
            f"{STANZA_INSTALL_INSTRUCTIONS}"
        )

    return True


def ensure_stanza_pipeline() -> Any:
    """Initialize and return a Stanza Pipeline for Hebrew.

    The pipeline is configured with ``tokenize``, ``mwt``, ``pos``, and
    ``lemma`` processors.  The instance is cached at module level so that
    repeated calls return the same object without re-loading the model.

    Returns:
        A ``stanza.Pipeline`` instance ready for Hebrew text processing.

    Raises:
        StanzaSetupError: If stanza is not installed or the Hebrew model
            is not available.
    """
    global _cached_pipeline

    if _cached_pipeline is not None:
        return _cached_pipeline

    # Verify the model is present before attempting to build the pipeline.
    check_stanza_model(lang="he")

    try:
        import stanza  # noqa: F811
    except ImportError:
        raise StanzaSetupError(
            "Stanza package is not installed."
            f"{STANZA_INSTALL_INSTRUCTIONS}"
        )

    try:
        pipeline = stanza.Pipeline(
            lang="he",
            processors="tokenize,mwt,pos,lemma",
        )
    except Exception as exc:
        raise StanzaSetupError(
            f"Failed to initialize Stanza Hebrew pipeline: {exc}"
            f"{STANZA_INSTALL_INSTRUCTIONS}"
        ) from exc

    _cached_pipeline = pipeline
    return _cached_pipeline
