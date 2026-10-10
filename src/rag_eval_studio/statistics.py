"""Aggregate eligible per-case metric values without filling missing scores."""

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

from rag_eval_studio.models import MetricValue


@dataclass(frozen=True, slots=True)
class MetricSummary:
    mean: float | None
    eligible: int
    total: int
    ci_low: float | None = None
    ci_high: float | None = None


def summarize(
    values: list[MetricValue], *, seed: int = 0, resamples: int = 2000
) -> MetricSummary:
    """Fixed-seed percentile bootstrap CI for the mean of eligible cases."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    scores = [item.value for item in values if item.value is not None]
    if not scores:
        return MetricSummary(mean=None, eligible=0, total=len(values))
    mean = fmean(scores)
    if len(scores) < 5:
        return MetricSummary(mean=mean, eligible=len(scores), total=len(values))

    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(scores, k=len(scores))) for _ in range(resamples))
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(0.975 * (resamples - 1))]
    return MetricSummary(mean, len(scores), len(values), low, high)


def summarize_by_category(
    cases: list[tuple[str | None, dict[str, MetricValue]]],
) -> dict[str, dict[str, MetricSummary]]:
    """Preserve eligibility per metric within each category."""
    grouped: dict[str, list[dict[str, MetricValue]]] = defaultdict(list)
    for category, metrics in cases:
        grouped[category or "Uncategorized"].append(metrics)
    return {
        category: {
            name: summarize(
                [
                    metrics.get(
                        name, MetricValue(skip_reason="metric was not recorded")
                    )
                    for metrics in records
                ]
            )
            for name in dict.fromkeys(name for metrics in records for name in metrics)
        }
        for category, records in grouped.items()
    }
