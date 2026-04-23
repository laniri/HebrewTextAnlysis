"""Issue ranker module for the probabilistic analysis layer."""

from typing import Dict, List

from analysis.issue_models import Issue


def compute_group_scores(issues: List[Issue]) -> Dict[str, float]:
    """Compute the mean severity per group.

    Returns a dict mapping group name to mean severity of all issues in that group.
    Groups with no issues are not included.
    """
    group_totals: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}

    for issue in issues:
        group_totals[issue.group] = group_totals.get(issue.group, 0.0) + issue.severity
        group_counts[issue.group] = group_counts.get(issue.group, 0) + 1

    return {group: group_totals[group] / group_counts[group] for group in group_totals}


def rank_issues(issues: List[Issue], k: int = 5) -> List[Issue]:
    """Rank issues by composite score and return top k. No threshold applied.

    1. Compute group_score = mean severity per group.
    2. For each issue: rank_score = 0.7 * severity + 0.3 * group_score.
    3. Sort by rank_score descending.
    4. Return top k.
    5. Add rank_score and group_score to each returned issue's evidence dict.
    """
    if not issues:
        return []

    group_scores = compute_group_scores(issues)

    scored = []
    for issue in issues:
        group_score = group_scores[issue.group]
        rank_score = 0.7 * issue.severity + 0.3 * group_score
        scored.append((rank_score, group_score, issue))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = []
    for rank_score, group_score, issue in scored[:k]:
        issue.evidence["rank_score"] = rank_score
        issue.evidence["group_score"] = group_score
        result.append(issue)

    return result
