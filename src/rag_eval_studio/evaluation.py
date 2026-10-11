"""Quick preset evaluation over captured live or precomputed observations."""

from dataclasses import fields, replace

from rag_eval_studio.models import (
    CaseResult,
    GoldenCase,
    MetricValue,
    Observation,
    ProviderConfig,
)
from rag_eval_studio.retrieval import score_retrieval

_OBSERVATION_FIELDS = {field.name for field in fields(Observation)}


def precomputed_observation(case: GoldenCase) -> Observation:
    return Observation(**{name: getattr(case, name) for name in _OBSERVATION_FIELDS})


def _observed(value: float | int | None, reason: str) -> MetricValue:
    return (
        MetricValue(value=float(value))
        if value is not None
        else MetricValue(skip_reason=reason)
    )


def _include_judge_totals(metrics: dict[str, MetricValue]) -> None:
    """Include judge usage in run totals only when every stage is known."""
    for total, judge_fields in (
        ("total_tokens", ("judge_prompt_tokens", "judge_completion_tokens")),
        ("total_cost_usd", ("judge_cost_usd",)),
    ):
        values = [metrics[total].value, *(metrics[name].value for name in judge_fields)]
        metrics[total] = (
            MetricValue(value=sum(values))
            if all(value is not None for value in values)
            else MetricValue(skip_reason=f"complete {total} is unavailable")
        )


def evaluate_quick(
    case: GoldenCase, *, top_k: int, observation: Observation | None = None
) -> CaseResult:
    """Evaluate a single captured output; missing observations stay missing."""
    observed = observation if observation is not None else precomputed_observation(case)
    metrics = score_retrieval(
        replace(case, retrieved_doc_ids=observed.retrieved_doc_ids), top_k
    )
    for name in ("retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"):
        metrics[name] = _observed(getattr(observed, name), f"{name} was not recorded")
    for name in (
        "embedding_tokens",
        "generation_prompt_tokens",
        "generation_completion_tokens",
    ):
        metrics[name] = _observed(getattr(observed, name), f"{name} were not recorded")
    tokens = (
        observed.embedding_tokens,
        observed.generation_prompt_tokens,
        observed.generation_completion_tokens,
    )
    metrics["total_tokens"] = _observed(
        sum(tokens) if all(token is not None for token in tokens) else None,
        "complete token usage was not recorded",
    )
    for name in ("retrieval_cost_usd", "generation_cost_usd"):
        metrics[name] = _observed(getattr(observed, name), f"{name} is unavailable")
    costs = (observed.retrieval_cost_usd, observed.generation_cost_usd)
    metrics["total_cost_usd"] = _observed(
        sum(costs) if all(cost is not None for cost in costs) else None,
        "complete cost data is unavailable",
    )
    return CaseResult(case.id, case.category, observed, metrics)


def evaluate_standard(
    case: GoldenCase,
    *,
    top_k: int,
    judge: ProviderConfig,
    observation: Observation | None = None,
) -> CaseResult:
    """Extend the Quick result with claim-level judge scores."""
    from rag_eval_studio.judges import judge_standard

    result = evaluate_quick(case, top_k=top_k, observation=observation)
    judged = judge_standard(case, result.observation, judge)
    metrics = {**result.metrics, **judged.metrics}
    _include_judge_totals(metrics)
    return replace(
        result,
        metrics=metrics,
        judge_details=judged.details,
    )


def evaluate_deep(
    case: GoldenCase,
    *,
    top_k: int,
    judge: ProviderConfig,
    observation: Observation | None = None,
) -> CaseResult:
    """Extend Standard with noise and citation diagnostics."""
    from rag_eval_studio.judges import judge_deep

    standard = evaluate_standard(
        case, top_k=top_k, judge=judge, observation=observation
    )
    deep = judge_deep(case, standard.observation, judge, standard.judge_details)
    metrics = {**standard.metrics, **deep.metrics}
    for name in (
        "judge_latency_ms",
        "judge_prompt_tokens",
        "judge_completion_tokens",
        "judge_cost_usd",
    ):
        if name in deep.metrics:
            first = standard.metrics[name].value
            second = deep.metrics[name].value
            metrics[name] = (
                MetricValue(value=first + second)
                if first is not None and second is not None
                else MetricValue(skip_reason=f"complete {name} is unavailable")
            )
    _include_judge_totals(metrics)
    return replace(
        standard,
        metrics=metrics,
        judge_details={**(standard.judge_details or {}), **deep.details},
    )
