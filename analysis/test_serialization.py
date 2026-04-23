# Feature: probabilistic-analysis-layer, Property 12: JSON serialization completeness

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from analysis.issue_models import Issue
from analysis.serialization import serialize_issues

GROUPS = ["morphology", "syntax", "lexicon", "structure", "discourse", "style"]

issue_strategy = st.builds(
    Issue,
    type=st.text(min_size=1, max_size=50),
    group=st.sampled_from(GROUPS),
    severity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    span=st.one_of(
        st.tuples(st.integers(min_value=0, max_value=100)),
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        ),
    ),
    evidence=st.dictionaries(
        keys=st.text(min_size=1, max_size=30),
        values=st.floats(allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
)


@given(issues=st.lists(issue_strategy, max_size=10))
@settings(max_examples=100)
def test_property12_json_serialization_completeness(issues):
    """Property 12: JSON serialization completeness.
    Validates: Requirements 14.1, 14.2, 14.3, 14.4
    """
    result = serialize_issues(issues)

    parsed = json.loads(result)

    assert "issues" in parsed
    assert isinstance(parsed["issues"], list)
    assert len(parsed["issues"]) == len(issues)

    required_fields = {"type", "group", "severity", "confidence", "span", "evidence"}

    for item in parsed["issues"]:
        assert required_fields.issubset(item.keys())

        assert isinstance(item["span"], list)
        assert all(isinstance(x, int) for x in item["span"])

        assert isinstance(item["evidence"], dict)
        for k, v in item["evidence"].items():
            assert isinstance(k, str)
            assert isinstance(v, (int, float))
