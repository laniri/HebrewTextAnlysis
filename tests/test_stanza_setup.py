"""Unit tests for hebrew_profiler.stanza_setup."""

from __future__ import annotations

import importlib
import os
import sys
from unittest import mock

import pytest

from hebrew_profiler.errors import StanzaSetupError


class TestStanzaInstallInstructions:
    """Verify the STANZA_INSTALL_INSTRUCTIONS constant."""

    def test_contains_pip_install(self):
        from hebrew_profiler.stanza_setup import STANZA_INSTALL_INSTRUCTIONS

        assert "pip install stanza" in STANZA_INSTALL_INSTRUCTIONS

    def test_contains_download_command(self):
        from hebrew_profiler.stanza_setup import STANZA_INSTALL_INSTRUCTIONS

        assert 'stanza.download(\'he\')' in STANZA_INSTALL_INSTRUCTIONS

    def test_contains_requirements_txt_mention(self):
        from hebrew_profiler.stanza_setup import STANZA_INSTALL_INSTRUCTIONS

        assert "requirements.txt" in STANZA_INSTALL_INSTRUCTIONS


class TestCheckStanzaModel:
    """Tests for check_stanza_model()."""

    def test_raises_when_stanza_not_installed(self):
        """If stanza cannot be imported, StanzaSetupError is raised."""
        import hebrew_profiler.stanza_setup as mod

        with mock.patch.dict(sys.modules, {"stanza": None, "stanza.resources": None, "stanza.resources.common": None}):
            # Force re-import to hit the ImportError path
            with pytest.raises(StanzaSetupError, match="not installed"):
                # We need to call the function in a way that triggers the
                # inner import.  Reload the module so the cached import is
                # cleared, then call.
                importlib.reload(mod)
                mod.check_stanza_model("he")

    def test_raises_when_model_dir_missing(self, tmp_path):
        """If the model directory doesn't exist, StanzaSetupError is raised."""
        fake_stanza = mock.MagicMock()
        fake_common = mock.MagicMock()
        # Point DEFAULT_MODEL_DIR to a temp dir that has no 'he' subfolder
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            import hebrew_profiler.stanza_setup as mod
            importlib.reload(mod)

            with pytest.raises(StanzaSetupError, match="not found"):
                mod.check_stanza_model("he")

    def test_returns_true_when_model_exists(self, tmp_path):
        """If the model directory exists, returns True."""
        model_dir = tmp_path / "he"
        model_dir.mkdir()

        fake_stanza = mock.MagicMock()
        fake_common = mock.MagicMock()
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            import hebrew_profiler.stanza_setup as mod
            importlib.reload(mod)

            assert mod.check_stanza_model("he") is True


class TestEnsureStanzaPipeline:
    """Tests for ensure_stanza_pipeline()."""

    def _reload_module(self):
        """Reload the module to reset the cached pipeline."""
        import hebrew_profiler.stanza_setup as mod
        importlib.reload(mod)
        mod._cached_pipeline = None
        return mod

    def test_raises_when_model_not_available(self, tmp_path):
        """StanzaSetupError propagates from check_stanza_model."""
        fake_stanza = mock.MagicMock()
        fake_common = mock.MagicMock()
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)  # no 'he' subdir

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            mod = self._reload_module()
            with pytest.raises(StanzaSetupError):
                mod.ensure_stanza_pipeline()

    def test_creates_pipeline_with_correct_processors(self, tmp_path):
        """Pipeline is created with tokenize, mwt, pos, lemma processors."""
        model_dir = tmp_path / "he"
        model_dir.mkdir()

        fake_pipeline_instance = mock.MagicMock()
        fake_stanza = mock.MagicMock()
        fake_stanza.Pipeline.return_value = fake_pipeline_instance
        fake_common = mock.MagicMock()
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            mod = self._reload_module()
            result = mod.ensure_stanza_pipeline()

            fake_stanza.Pipeline.assert_called_once_with(
                lang="he",
                processors="tokenize,mwt,pos,lemma",
            )
            assert result is fake_pipeline_instance

    def test_caches_pipeline_instance(self, tmp_path):
        """Calling ensure_stanza_pipeline twice returns the same object."""
        model_dir = tmp_path / "he"
        model_dir.mkdir()

        fake_pipeline_instance = mock.MagicMock()
        fake_stanza = mock.MagicMock()
        fake_stanza.Pipeline.return_value = fake_pipeline_instance
        fake_common = mock.MagicMock()
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            mod = self._reload_module()
            first = mod.ensure_stanza_pipeline()
            second = mod.ensure_stanza_pipeline()

            assert first is second
            # Pipeline constructor should only be called once
            assert fake_stanza.Pipeline.call_count == 1

    def test_raises_on_pipeline_init_failure(self, tmp_path):
        """If stanza.Pipeline() raises, StanzaSetupError is raised."""
        model_dir = tmp_path / "he"
        model_dir.mkdir()

        fake_stanza = mock.MagicMock()
        fake_stanza.Pipeline.side_effect = RuntimeError("model corrupt")
        fake_common = mock.MagicMock()
        fake_common.DEFAULT_MODEL_DIR = str(tmp_path)

        with mock.patch.dict(
            sys.modules,
            {
                "stanza": fake_stanza,
                "stanza.resources": mock.MagicMock(),
                "stanza.resources.common": fake_common,
            },
        ):
            mod = self._reload_module()
            with pytest.raises(StanzaSetupError, match="Failed to initialize"):
                mod.ensure_stanza_pipeline()
