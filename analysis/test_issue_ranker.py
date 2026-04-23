# Feature: probabilistic-analysis-layer, Property 11: Ranker ordering and selection

from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.issue_models import Issue
from analysis.issue_ranker import compute_group_scores, rank_issues

VALID_GROUPS = ["morphology", "syntax", "lexicon", "structure", "discourse", "style"]

issue_strategy = st.builds(
    Issue,
    type=st.text(min_size=1, max_size=30),
    group=st.sampled_from(VALID_GROUPS),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    span=st.just((0, 1)),
    evidence=st.just({}),
)


@given(issues=st.lists(issue_strategy, min_size=0, max_size=20))
@settings(max_examples=100)
def test_ranker_ordering_and_selection(issues):
    """Property 11: Ranker ordering and selection.
    Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
    """
    fresh_issues = [
        Issue(
            type=iss.type,
            group=iss.group,
            severity=iss.severity,
            confidence=iss.confidence,
            span=iss.span,
            evidence={},
        )
        for iss in issues
    ]

    result = rank_issues(fresh_issues, k=5)

    # Length is min(len(issues), 5) — no threshold filtering
    assert len(result) == min(len(fresh_issues), 5)

    if not result:
        return

    expected_group_scores = compute_group_scores(fresh_issues)

    for issue in result:
        expected_gs = expected_group_scores[issue.group]
        assert abs(issue.evidence["group_score"] - expected_gs) < 1e-9

        expected_rs = 0.7 * issue.severity + 0.3 * expected_gs
        assert abs(issue.evidence["rank_score"] - expected_rs) < 1e-9

    rank_scores = [iss.evidence["rank_score"] for iss in result]
    assert rank_scores == sorted(rank_scores, reverse=True)
