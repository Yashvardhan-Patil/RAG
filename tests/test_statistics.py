from rag_eval_studio.models import CaseResult, MetricValue, Observation
from rag_eval_studio.report import summary_rows
from rag_eval_studio.statistics import summarize, summarize_by_category


def test_bootstrap_is_reproducible_and_excludes_skips():
    values = [
        MetricValue(value=0),
        MetricValue(value=1),
        MetricValue(skip_reason="no labels"),
    ]
    summary = summarize(values, resamples=100)
    assert summary == summarize(values, resamples=100)
    assert (summary.mean, summary.eligible, summary.total) == (0.5, 2, 3)
    assert summary.ci_low is None and summary.ci_high is None


def test_single_and_all_skipped_have_no_interval():
    assert summarize([MetricValue(value=0.3)]).ci_low is None
    assert summarize([MetricValue(value=0), MetricValue(value=1)]).ci_low is None
    missing = summarize([MetricValue(skip_reason="missing")])
    assert missing.mean is None and missing.eligible == 0


def test_known_mean_and_constant_bootstrap_interval():
    values = [MetricValue(value=2) for _ in range(5)]
    values.append(MetricValue(skip_reason="unavailable"))
    assert summarize(values, resamples=100).mean == 2
    assert summarize(values, resamples=100).eligible == 5
    assert summarize(values, resamples=100).total == 6
    assert summarize(values, resamples=100).ci_low == 2
    assert summarize(values, resamples=100).ci_high == 2


def test_categories_keep_distinct_denominators():
    cases = [
        ("A", {"hit_rate": MetricValue(value=1)}),
        ("A", {"hit_rate": MetricValue(skip_reason="missing")}),
        ("B", {"hit_rate": MetricValue(value=0)}),
    ]
    groups = summarize_by_category(cases)
    assert groups["A"]["hit_rate"].eligible == 1
    assert groups["A"]["hit_rate"].total == 2
    assert groups["B"]["hit_rate"].mean == 0


def test_category_counts_include_cases_with_unrecorded_metric():
    cases = [
        ("A", {"hit_rate": MetricValue(value=1)}),
        ("A", {}),
    ]
    summary = summarize_by_category(cases)["A"]["hit_rate"]
    assert (summary.mean, summary.eligible, summary.total) == (1, 1, 2)


def test_overview_counts_include_cases_with_unrecorded_metric():
    results = [
        CaseResult("a", None, Observation(), {"hit_rate": MetricValue(value=1)}),
        CaseResult("b", None, Observation(), {}),
    ]
    row = next(row for row in summary_rows(results) if row[1] == "Hit Rate@K")
    assert row[-1] == "1/2"
