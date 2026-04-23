"""Unit tests for hebrew_profiler.pipeline module.

Tests process_document orchestration, JSON serialization helpers,
and Hebrew character preservation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hebrew_profiler.models import (
    DiscourseFeatures,
    Features,
    LexicalFeatures,
    MorphFeatures,
    NormalizationRanges,
    PipelineConfig,
    PipelineOutput,
    Scores,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)
from hebrew_profiler.pipeline import (
    pipeline_output_to_dict,
    pipeline_output_to_json,
    process_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(
    morph_none: bool = False,
    syntax_none: bool = False,
) -> Features:
    """Build a Features object, optionally with None sub-fields."""
    if morph_none:
        morphology = MorphFeatures(
            verb_ratio=None,
            binyan_distribution=None,
            prefix_density=None,
            suffix_pronoun_ratio=None,
            morphological_ambiguity=None,
            agreement_error_rate=None,
            binyan_entropy=None,
            construct_ratio=None,
        )
    else:
        morphology = MorphFeatures(
            verb_ratio=0.25,
            binyan_distribution={"PAAL": 2, "PIEL": 1},
            prefix_density=0.15,
            suffix_pronoun_ratio=0.05,
            morphological_ambiguity=3.2,
            agreement_error_rate=0.0,
            binyan_entropy=0.0,
            construct_ratio=0.0,
        )

    if syntax_none:
        syntax = SyntaxFeatures(
            avg_sentence_length=None,
            avg_tree_depth=None,
            max_tree_depth=None,
            avg_dependency_distance=None,
            clauses_per_sentence=None,
            subordinate_clause_ratio=None,
            right_branching_ratio=None,
            dependency_distance_variance=None,
        )
    else:
        syntax = SyntaxFeatures(
            avg_sentence_length=12.5,
            avg_tree_depth=4.0,
            max_tree_depth=6.0,
            avg_dependency_distance=2.3,
            clauses_per_sentence=1.2,
            subordinate_clause_ratio=0.0,
            right_branching_ratio=0.0,
            dependency_distance_variance=0.0,
        )

    lexicon = LexicalFeatures(
        type_token_ratio=0.78,
        hapax_ratio=0.45,
        avg_token_length=4.1,
        lemma_diversity=0.72,
        rare_word_ratio=None,
        content_word_ratio=0.0,
    )

    structure = StructuralFeatures(
        sentence_length_variance=15.3,
        long_sentence_ratio=0.2,
        punctuation_ratio=0.0,
        short_sentence_ratio=0.0,
        missing_terminal_punctuation_ratio=0.0,
    )

    return Features(
        morphology=morphology,
        syntax=syntax,
        lexicon=lexicon,
        structure=structure,
        discourse=DiscourseFeatures(
            connective_ratio=0.0,
            sentence_overlap=0.0,
            pronoun_to_noun_ratio=0.0,
        ),
        style=StyleFeatures(
            sentence_length_trend=0.0,
            pos_distribution_variance=0.0,
        ),
    )


def _make_output(
    text: str = "שלום עולם",
    morph_none: bool = False,
    syntax_none: bool = False,
) -> PipelineOutput:
    """Build a PipelineOutput for testing."""
    features = _make_features(morph_none=morph_none, syntax_none=syntax_none)
    scores = Scores(difficulty=0.62, style=0.45)
    return PipelineOutput(text=text, features=features, scores=scores)


# ---------------------------------------------------------------------------
# pipeline_output_to_dict tests
# ---------------------------------------------------------------------------

class TestPipelineOutputToDict:
    def test_top_level_keys(self):
        output = _make_output()
        d = pipeline_output_to_dict(output)
        assert set(d.keys()) == {"text", "features", "scores"}

    def test_features_sub_keys(self):
        output = _make_output()
        d = pipeline_output_to_dict(output)
        assert set(d["features"].keys()) == {
            "morphology", "syntax", "lexicon", "structure", "discourse", "style"
        }

    def test_scores_sub_keys(self):
        output = _make_output()
        d = pipeline_output_to_dict(output)
        assert set(d["scores"].keys()) == {"difficulty", "style", "fluency", "cohesion", "complexity"}

    def test_text_preserved(self):
        output = _make_output(text="טקסט בעברית")
        d = pipeline_output_to_dict(output)
        assert d["text"] == "טקסט בעברית"

    def test_none_features_preserved(self):
        output = _make_output(morph_none=True)
        d = pipeline_output_to_dict(output)
        morph = d["features"]["morphology"]
        assert morph["verb_ratio"] is None
        assert morph["binyan_distribution"] is None
        assert morph["prefix_density"] is None
        assert morph["suffix_pronoun_ratio"] is None
        assert morph["morphological_ambiguity"] is None

    def test_numeric_features_present(self):
        output = _make_output()
        d = pipeline_output_to_dict(output)
        assert d["features"]["morphology"]["verb_ratio"] == 0.25
        assert d["features"]["syntax"]["avg_sentence_length"] == 12.5
        assert d["features"]["lexicon"]["type_token_ratio"] == 0.78
        assert d["features"]["structure"]["sentence_length_variance"] == 15.3


# ---------------------------------------------------------------------------
# pipeline_output_to_json tests
# ---------------------------------------------------------------------------

class TestPipelineOutputToJson:
    def test_valid_json(self):
        output = _make_output()
        json_str = pipeline_output_to_json(output)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_hebrew_preserved_no_escapes(self):
        output = _make_output(text="שלום עולם")
        json_str = pipeline_output_to_json(output)
        # Hebrew chars should appear literally, not as \\uXXXX
        assert "שלום" in json_str
        assert "\\u05E9" not in json_str

    def test_pretty_output(self):
        output = _make_output()
        json_str = pipeline_output_to_json(output, pretty=True)
        # Pretty output should have newlines and indentation
        assert "\n" in json_str
        parsed = json.loads(json_str)
        assert parsed["text"] == output.text

    def test_compact_output(self):
        output = _make_output()
        json_str = pipeline_output_to_json(output, pretty=False)
        # Compact output should not have leading indentation
        assert "\n  " not in json_str

    def test_null_for_none_features(self):
        output = _make_output(morph_none=True, syntax_none=True)
        json_str = pipeline_output_to_json(output)
        parsed = json.loads(json_str)
        assert parsed["features"]["morphology"]["verb_ratio"] is None
        assert parsed["features"]["syntax"]["avg_sentence_length"] is None

    def test_scores_in_json(self):
        output = _make_output()
        json_str = pipeline_output_to_json(output)
        parsed = json.loads(json_str)
        assert parsed["scores"]["difficulty"] == 0.62
        assert parsed["scores"]["style"] == 0.45

    def test_none_scores_serialized_as_null(self):
        features = _make_features()
        scores = Scores(difficulty=None, style=None)
        output = PipelineOutput(text="test", features=features, scores=scores)
        json_str = pipeline_output_to_json(output)
        parsed = json.loads(json_str)
        assert parsed["scores"]["difficulty"] is None
        assert parsed["scores"]["style"] is None


# ---------------------------------------------------------------------------
# process_document tests (mocked external dependencies)
# ---------------------------------------------------------------------------

class TestProcessDocument:
    """Test process_document orchestration with mocked Stanza/YAP."""

    @patch("hebrew_profiler.pipeline.parse_syntax")
    @patch("hebrew_profiler.pipeline.analyze_morphology")
    @patch("hebrew_profiler.pipeline.check_stanza_model")
    def test_returns_pipeline_output(
        self, mock_check, mock_stanza, mock_yap
    ):
        from hebrew_profiler.models import (
            StanzaResult,
            MorphAnalysis,
            YAPResult,
            SentenceTree,
            DepTreeNode,
        )

        mock_check.return_value = True
        mock_stanza.return_value = StanzaResult(analyses=[
            MorphAnalysis(
                surface="שלום",
                lemma="שלום",
                pos="NOUN",
                gender="Masc",
                number="Sing",
                prefixes=[],
                suffix=None,
                binyan=None,
                tense=None,
                ambiguity_count=1,
                top_k_analyses=[],
            ),
        ])
        mock_yap.return_value = YAPResult(
            morphological_disambiguation=[],
            sentences=[
                SentenceTree(nodes=[
                    DepTreeNode(
                        id=1, form="שלום", lemma="שלום",
                        cpostag="NN", postag="NN",
                        features={}, head=0, deprel="ROOT",
                    ),
                ]),
            ],
            ambiguity_counts=[4],
        )

        config = PipelineConfig()
        result = process_document("שלום", config)

        assert isinstance(result, PipelineOutput)
        assert result.text == "שלום"
        assert result.features is not None
        assert result.scores is not None

    @patch("hebrew_profiler.pipeline.parse_syntax")
    @patch("hebrew_profiler.pipeline.analyze_morphology")
    @patch("hebrew_profiler.pipeline.check_stanza_model")
    def test_calls_check_stanza_model(
        self, mock_check, mock_stanza, mock_yap
    ):
        from hebrew_profiler.models import StanzaError, YAPError

        mock_check.return_value = True
        mock_stanza.return_value = StanzaError(
            error_type="TestError", message="test"
        )
        mock_yap.return_value = YAPError(
            error_type="TestError", http_status=None, message="test"
        )

        config = PipelineConfig()
        process_document("test", config)

        mock_check.assert_called_once_with(lang=config.stanza_lang)

    @patch("hebrew_profiler.pipeline.parse_syntax")
    @patch("hebrew_profiler.pipeline.analyze_morphology")
    @patch("hebrew_profiler.pipeline.check_stanza_model")
    def test_handles_both_adapters_failing(
        self, mock_check, mock_stanza, mock_yap
    ):
        from hebrew_profiler.models import StanzaError, YAPError

        mock_check.return_value = True
        mock_stanza.return_value = StanzaError(
            error_type="RuntimeError", message="Stanza failed"
        )
        mock_yap.return_value = YAPError(
            error_type="YAPConnectionError",
            http_status=None,
            message="Connection refused",
        )

        config = PipelineConfig()
        result = process_document("שלום עולם", config)

        assert isinstance(result, PipelineOutput)
        assert result.text == "שלום עולם"
        # Morphology features should be None (stanza failed)
        assert result.features.morphology.verb_ratio is None
        # Syntax features should be None (yap failed)
        assert result.features.syntax.avg_sentence_length is None
        # Lexical/structural features should still be computed
        assert result.features.lexicon.type_token_ratio is not None
        assert result.features.structure.sentence_length_variance is not None

    @patch("hebrew_profiler.pipeline.check_stanza_model")
    def test_stanza_check_failure_propagates(self, mock_check):
        from hebrew_profiler.errors import StanzaSetupError

        mock_check.side_effect = StanzaSetupError("Model not found")

        config = PipelineConfig()
        with pytest.raises(StanzaSetupError):
            process_document("test", config)
