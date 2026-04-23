"""Integration test for the probabilistic analysis layer.

Tests the full pipeline from corpus loading through issue detection and serialization.
Does NOT require Stanza or YAP to be running — works purely from corpus JSON files.

Requirements: 1.5, 14.1, 15.1, 15.2
"""

import json
import os
from pathlib import Path

import pytest

from analysis.analysis_pipeline import flatten_features
from analysis.issue_detector import detect_issues
from analysis.issue_ranker import rank_issues
from analysis.serialization import serialize_issues
from analysis.statistics import (
    compute_feature_stats,
    flatten_corpus_json,
    load_stats,
    save_stats,
)
from hebrew_profiler.models import (
    DiscourseFeatures,
    Features,
    LexicalFeatures,
    MorphFeatures,
    StructuralFeatures,
    StyleFeatures,
    SyntaxFeatures,
)

RESULTS_SAMPLE_DIR = Path(__file__).parent.parent / "results_sample"
DOC_0000001 = RESULTS_SAMPLE_DIR / "doc_0000001.json"


def _load_corpus_json_files():
    files = sorted(RESULTS_SAMPLE_DIR.glob("doc_*.json"))
    assert len(files) == 100, f"Expected 100 corpus files, found {len(files)}"
    corpus = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            corpus.append(json.load(fh))
    return corpus


def _build_features_from_corpus_json(corpus_json: dict) -> Features:
    """Construct a Features object from a corpus JSON file without running the pipeline."""
    f = corpus_json.get("features", {})
    morph = f.get("morphology", {})
    syntax = f.get("syntax", {})
    lexicon = f.get("lexicon", {})
    structure = f.get("structure", {})
    discourse = f.get("discourse", {})
    style = f.get("style", {})

    return Features(
        morphology=MorphFeatures(
            verb_ratio=morph.get("verb_ratio"),
            binyan_distribution=None,  # non-scalar, skipped by flatten_features
            prefix_density=morph.get("prefix_density"),
            suffix_pronoun_ratio=morph.get("suffix_pronoun_ratio"),
            morphological_ambiguity=morph.get("morphological_ambiguity"),
            agreement_error_rate=morph.get("agreement_error_rate"),
            binyan_entropy=morph.get("binyan_entropy"),
            construct_ratio=morph.get("construct_ratio"),
        ),
        syntax=SyntaxFeatures(
            avg_sentence_length=syntax.get("avg_sentence_length"),
            avg_tree_depth=syntax.get("avg_tree_depth"),
            max_tree_depth=syntax.get("max_tree_depth"),
            avg_dependency_distance=syntax.get("avg_dependency_distance"),
            clauses_per_sentence=syntax.get("clauses_per_sentence"),
            subordinate_clause_ratio=syntax.get("subordinate_clause_ratio"),
            right_branching_ratio=syntax.get("right_branching_ratio"),
            dependency_distance_variance=syntax.get("dependency_distance_variance"),
            clause_type_entropy=syntax.get("clause_type_entropy"),
        ),
        lexicon=LexicalFeatures(
            type_token_ratio=lexicon.get("type_token_ratio"),
            hapax_ratio=lexicon.get("hapax_ratio"),
            avg_token_length=lexicon.get("avg_token_length"),
            lemma_diversity=lexicon.get("lemma_diversity"),
            rare_word_ratio=lexicon.get("rare_word_ratio"),
            content_word_ratio=lexicon.get("content_word_ratio"),
        ),
        structure=StructuralFeatures(
            sentence_length_variance=structure.get("sentence_length_variance"),
            long_sentence_ratio=structure.get("long_sentence_ratio"),
            punctuation_ratio=structure.get("punctuation_ratio"),
            short_sentence_ratio=structure.get("short_sentence_ratio"),
            missing_terminal_punctuation_ratio=structure.get("missing_terminal_punctuation_ratio"),
        ),
        discourse=DiscourseFeatures(
            connective_ratio=discourse.get("connective_ratio"),
            sentence_overlap=discourse.get("sentence_overlap"),
            pronoun_to_noun_ratio=discourse.get("pronoun_to_noun_ratio"),
        ),
        style=StyleFeatures(
            sentence_length_trend=style.get("sentence_length_trend"),
            pos_distribution_variance=style.get("pos_distribution_variance"),
        ),
    )


class TestIntegration:

    def test_full_pipeline_on_one_document(self, tmp_path):
        """Integration test: load corpus, compute stats, run full analysis on one document.
        Requirements: 1.5, 14.1, 15.1, 15.2
        """
        # Step 1: Load all 100 corpus JSON files and flatten features
        corpus_jsons = _load_corpus_json_files()
        feature_dicts = [flatten_corpus_json(doc) for doc in corpus_jsons]
        assert len(feature_dicts) == 100

        # Step 2: Compute feature stats
        feature_stats = compute_feature_stats(feature_dicts)
        assert len(feature_stats) > 0

        # Step 3: Save stats and verify file is created
        stats_path = str(tmp_path / "feature_stats.json")
        save_stats(feature_stats, feature_path=stats_path)
        assert os.path.exists(stats_path)

        # Step 4: Load stats back and verify round-trip
        loaded_stats = load_stats(feature_path=stats_path)
        assert set(loaded_stats.keys()) == set(feature_stats.keys())
        for key in feature_stats:
            assert loaded_stats[key].mean == pytest.approx(feature_stats[key].mean)
            assert loaded_stats[key].valid_count == feature_stats[key].valid_count

        # Step 5: Run detect_issues + rank_issues on doc_0000001.json
        with open(DOC_0000001, encoding="utf-8") as fh:
            doc = json.load(fh)

        raw_features = flatten_corpus_json(doc)
        assert len(raw_features) > 0

        # Use empty sentence_metrics (no pipeline run needed for this test)
        issues = detect_issues(raw_features, [], loaded_stats)
        assert isinstance(issues, list)
        assert len(issues) > 0

        ranked = rank_issues(issues, k=5)
        assert isinstance(ranked, list)
        assert len(ranked) <= 5

        # Step 6: Serialize and verify JSON structure (Requirements 14.1-14.4)
        serialized = serialize_issues(ranked)
        output = json.loads(serialized)

        assert "issues" in output
        assert isinstance(output["issues"], list)

        required_fields = {"type", "group", "severity", "confidence", "span", "evidence"}
        valid_groups = {"morphology", "syntax", "lexicon", "structure", "discourse", "style"}

        for issue_obj in output["issues"]:
            assert required_fields.issubset(issue_obj.keys())
            assert issue_obj["group"] in valid_groups
            assert 0.0 <= issue_obj["severity"] <= 1.0
            assert 0.0 <= issue_obj["confidence"] <= 1.0
            assert isinstance(issue_obj["span"], list)
            assert all(isinstance(x, int) for x in issue_obj["span"])
            assert isinstance(issue_obj["evidence"], dict)

    def test_flatten_corpus_json_and_flatten_features_same_key_set(self):
        """Verify flatten_corpus_json and flatten_features produce the same key set.
        Requirements: 15.1, 15.2
        """
        with open(DOC_0000001, encoding="utf-8") as fh:
            corpus_json = json.load(fh)

        corpus_keys = set(flatten_corpus_json(corpus_json).keys())

        features = _build_features_from_corpus_json(corpus_json)
        pipeline_keys = set(flatten_features(features).keys())

        assert corpus_keys == pipeline_keys, (
            f"Key mismatch.\n"
            f"Only in corpus JSON: {corpus_keys - pipeline_keys}\n"
            f"Only in Features: {pipeline_keys - corpus_keys}"
        )
