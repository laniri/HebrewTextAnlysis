"""Property-based tests for Hebrew Linguistic Profiling Engine serialization.

Tests Properties 20 and 21 from the design document:
- Property 20: Hebrew character preservation in serialization
- Property 21: Partial features serialized as null
"""

from __future__ import annotations

import json
import re

import hypothesis.strategies as st
from hypothesis import given, settings

from hebrew_profiler.models import (
    DiscourseFeatures,
    Features,
    LexicalFeatures,
    MorphFeatures,
    PipelineOutput,
    Scores,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)
from hebrew_profiler.pipeline import pipeline_output_to_dict, pipeline_output_to_json


# ── Strategies ───────────────────────────────────────────────────────

# Hebrew Unicode range: U+0590 – U+05FF (Hebrew block)
hebrew_chars = st.text(
    alphabet=st.characters(min_codepoint=0x0590, max_codepoint=0x05FF),
    min_size=1,
    max_size=200,
)


def _optional_float(min_value: float = 0.0, max_value: float = 100.0):
    """Return a strategy that yields either a finite float or None."""
    return st.one_of(
        st.none(),
        st.floats(min_value=min_value, max_value=max_value),
    )


@st.composite
def pipeline_output_with_hebrew(draw):
    """Generate a PipelineOutput whose text field contains Hebrew characters."""
    text = draw(hebrew_chars)
    return PipelineOutput(
        text=text,
        features=Features(
            morphology=MorphFeatures(
                verb_ratio=0.2,
                binyan_distribution={"paal": 1},
                prefix_density=0.1,
                suffix_pronoun_ratio=0.05,
                morphological_ambiguity=2.0,
                agreement_error_rate=0.0,
                binyan_entropy=0.0,
                construct_ratio=0.0,
            ),
            syntax=SyntaxFeatures(
                avg_sentence_length=10.0,
                avg_tree_depth=3.0,
                max_tree_depth=5.0,
                avg_dependency_distance=2.0,
                clauses_per_sentence=1.0,
                subordinate_clause_ratio=0.0,
                right_branching_ratio=0.0,
                dependency_distance_variance=0.0,
            ),
            lexicon=LexicalFeatures(
                type_token_ratio=0.8,
                hapax_ratio=0.4,
                avg_token_length=4.0,
                lemma_diversity=0.7,
                rare_word_ratio=None,
                content_word_ratio=0.0,
            ),
            structure=StructuralFeatures(
                sentence_length_variance=10.0,
                long_sentence_ratio=0.2,
                punctuation_ratio=0.0,
                short_sentence_ratio=0.0,
                missing_terminal_punctuation_ratio=0.0,
            ),
            discourse=DiscourseFeatures(
                connective_ratio=0.0,
                sentence_overlap=0.0,
                pronoun_to_noun_ratio=0.0,
            ),
            style=StyleFeatures(
                sentence_length_trend=0.0,
                pos_distribution_variance=0.0,
            ),
        ),
        scores=Scores(difficulty=0.5, style=0.3),
    )


@st.composite
def features_with_random_nones(draw):
    """Generate a Features object where each numeric field is randomly None or a float."""
    return Features(
        morphology=MorphFeatures(
            verb_ratio=draw(_optional_float(0.0, 1.0)),
            binyan_distribution=draw(st.one_of(
                st.none(),
                st.just({"paal": 1}),
            )),
            prefix_density=draw(_optional_float(0.0, 1.0)),
            suffix_pronoun_ratio=draw(_optional_float(0.0, 1.0)),
            morphological_ambiguity=draw(_optional_float(1.0, 8.0)),
            agreement_error_rate=draw(_optional_float(0.0, 0.3)),
            binyan_entropy=draw(_optional_float(0.0, 3.0)),
            construct_ratio=draw(_optional_float(0.0, 0.5)),
        ),
        syntax=SyntaxFeatures(
            avg_sentence_length=draw(_optional_float(1.0, 100.0)),
            avg_tree_depth=draw(_optional_float(1.0, 20.0)),
            max_tree_depth=draw(_optional_float(1.0, 20.0)),
            avg_dependency_distance=draw(_optional_float(0.0, 20.0)),
            clauses_per_sentence=draw(_optional_float(0.0, 10.0)),
            subordinate_clause_ratio=draw(_optional_float(0.0, 1.0)),
            right_branching_ratio=draw(_optional_float(0.0, 1.0)),
            dependency_distance_variance=draw(_optional_float(0.0, 20.0)),
        ),
        lexicon=LexicalFeatures(
            type_token_ratio=draw(_optional_float(0.0, 1.0)),
            hapax_ratio=draw(_optional_float(0.0, 1.0)),
            avg_token_length=draw(_optional_float(1.0, 20.0)),
            lemma_diversity=draw(_optional_float(0.0, 1.0)),
            rare_word_ratio=draw(_optional_float(0.0, 0.3)),
            content_word_ratio=draw(_optional_float(0.0, 1.0)),
        ),
        structure=StructuralFeatures(
            sentence_length_variance=draw(_optional_float(0.0, 500.0)),
            long_sentence_ratio=draw(_optional_float(0.0, 1.0)),
            punctuation_ratio=draw(_optional_float(0.0, 1.0)),
            short_sentence_ratio=draw(_optional_float(0.0, 1.0)),
            missing_terminal_punctuation_ratio=draw(_optional_float(0.0, 1.0)),
        ),
        discourse=DiscourseFeatures(
            connective_ratio=draw(_optional_float(0.0, 2.0)),
            sentence_overlap=draw(_optional_float(0.0, 1.0)),
            pronoun_to_noun_ratio=draw(_optional_float(0.0, 5.0)),
        ),
        style=StyleFeatures(
            sentence_length_trend=draw(_optional_float(-10.0, 10.0)),
            pos_distribution_variance=draw(_optional_float(0.0, 1.0)),
        ),
    )


# Regex matching \uXXXX escape sequences for Hebrew code points (U+0590–U+05FF)
_HEBREW_ESCAPE_RE = re.compile(r"\\u05[0-9a-fA-F]{2}")


# ── Property 20: Hebrew character preservation in serialization ──────
# Feature: hebrew-linguistic-profiling-engine, Property 20
# **Validates: Requirements 13.4, 15.3**

class TestProperty20HebrewPreservation:
    @given(output=pipeline_output_with_hebrew())
    @settings(max_examples=200)
    def test_no_hebrew_escape_sequences_in_json(self, output: PipelineOutput):
        """For any PipelineOutput containing Hebrew text, serializing to JSON
        with ensure_ascii=False SHALL preserve all Hebrew characters in their
        native Unicode form — no \\uXXXX escape sequences for Hebrew code
        points (U+0590–U+05FF)."""
        json_str = pipeline_output_to_json(output)

        # The raw JSON string must not contain \u05XX escape sequences
        assert not _HEBREW_ESCAPE_RE.search(json_str), (
            f"Found Hebrew escape sequence in JSON output: "
            f"{_HEBREW_ESCAPE_RE.findall(json_str)}"
        )

        # Round-trip: parsing the JSON back must recover the original Hebrew text
        parsed = json.loads(json_str)
        assert parsed["text"] == output.text


# ── Property 21: Partial features serialized as null ─────────────────
# Feature: hebrew-linguistic-profiling-engine, Property 21
# **Validates: Requirements 13.5**

class TestProperty21PartialFeaturesNull:
    @given(features=features_with_random_nones())
    @settings(max_examples=200)
    def test_none_values_serialize_as_null(self, features: Features):
        """For any PipelineOutput where some feature values are None, the JSON
        serialization SHALL represent those values as null and all non-None
        feature values SHALL be present as their numeric values."""
        output = PipelineOutput(
            text="test",
            features=features,
            scores=Scores(difficulty=0.5, style=None),
        )

        json_str = pipeline_output_to_json(output)
        parsed = json.loads(json_str)

        # Check morphology features
        morph = parsed["features"]["morphology"]
        _assert_none_null_match(features.morphology.verb_ratio, morph["verb_ratio"])
        _assert_none_null_match(features.morphology.binyan_distribution, morph["binyan_distribution"])
        _assert_none_null_match(features.morphology.prefix_density, morph["prefix_density"])
        _assert_none_null_match(features.morphology.suffix_pronoun_ratio, morph["suffix_pronoun_ratio"])
        _assert_none_null_match(features.morphology.morphological_ambiguity, morph["morphological_ambiguity"])

        # Check syntax features
        syn = parsed["features"]["syntax"]
        _assert_none_null_match(features.syntax.avg_sentence_length, syn["avg_sentence_length"])
        _assert_none_null_match(features.syntax.avg_tree_depth, syn["avg_tree_depth"])
        _assert_none_null_match(features.syntax.max_tree_depth, syn["max_tree_depth"])
        _assert_none_null_match(features.syntax.avg_dependency_distance, syn["avg_dependency_distance"])
        _assert_none_null_match(features.syntax.clauses_per_sentence, syn["clauses_per_sentence"])

        # Check lexical features
        lex = parsed["features"]["lexicon"]
        _assert_none_null_match(features.lexicon.type_token_ratio, lex["type_token_ratio"])
        _assert_none_null_match(features.lexicon.hapax_ratio, lex["hapax_ratio"])
        _assert_none_null_match(features.lexicon.avg_token_length, lex["avg_token_length"])
        _assert_none_null_match(features.lexicon.lemma_diversity, lex["lemma_diversity"])

        # Check structural features
        struct = parsed["features"]["structure"]
        _assert_none_null_match(features.structure.sentence_length_variance, struct["sentence_length_variance"])
        _assert_none_null_match(features.structure.long_sentence_ratio, struct["long_sentence_ratio"])

        # Check scores
        scores = parsed["scores"]
        _assert_none_null_match(output.scores.difficulty, scores["difficulty"])
        _assert_none_null_match(output.scores.style, scores["style"])


def _assert_none_null_match(python_value, json_value):
    """Assert that a Python None maps to JSON null and non-None maps to its value."""
    if python_value is None:
        assert json_value is None, (
            f"Expected null in JSON for Python None, got {json_value!r}"
        )
    else:
        assert json_value is not None, (
            f"Expected non-null in JSON for Python value {python_value!r}, got null"
        )
        # For dicts (like binyan_distribution), check equality directly
        if isinstance(python_value, dict):
            assert json_value == python_value
        else:
            assert json_value == python_value, (
                f"Expected {python_value!r} in JSON, got {json_value!r}"
            )
