"""Run one preset over live or precomputed cases."""

from collections.abc import Callable
from dataclasses import dataclass

from rag_eval_studio.evaluation import (
    evaluate_deep,
    evaluate_quick,
    evaluate_standard,
    precomputed_observation,
)
from rag_eval_studio.judges import STANDARD_METRICS
from rag_eval_studio.models import (
    CaseResult,
    GoldenCase,
    MetricValue,
    Observation,
    ProviderConfig,
)
from rag_eval_studio.pipeline import run_existing_chroma
from rag_eval_studio.retrieval import METRICS as RETRIEVAL_METRICS

PRESETS = ("Quick", "Standard", "Deep")
MODES = ("Existing Chroma", "Precomputed Results")
QUICK_METRICS = (
    *RETRIEVAL_METRICS,
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "embedding_tokens",
    "generation_prompt_tokens",
    "generation_completion_tokens",
    "total_tokens",
    "retrieval_cost_usd",
    "generation_cost_usd",
    "total_cost_usd",
)
DEEP_METRICS = (
    "noise_sensitivity",
    "citation_support",
    "citation_coverage",
    "citation_unresolved_ids",
)
JUDGE_OPERATIONS = (
    "judge_latency_ms",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "judge_cost_usd",
)


@dataclass(frozen=True, slots=True)
class RunSettings:
    mode: str
    preset: str
    top_k: int
    chroma_path: str | None = None
    collection_name: str | None = None
    embedding: ProviderConfig | None = None
    generation: ProviderConfig | None = None
    judge: ProviderConfig | None = None

    def validate(self) -> None:
        if self.mode not in MODES or self.preset not in PRESETS or self.top_k < 1:
            raise ValueError("mode, preset, or top_k is invalid")
        if self.mode == "Existing Chroma":
            if not self.chroma_path or not self.collection_name:
                raise ValueError("Chroma needs a path and collection")
            if self.embedding is None:
                raise ValueError("Chroma needs an embedding provider and model")
            if self.generation is None:
                raise ValueError("Chroma needs a generation provider and model")
        if self.preset != "Quick" and self.judge is None:
            raise ValueError("Standard and Deep need a judge provider and model")


def _expected_metrics(preset: str) -> tuple[str, ...]:
    names = QUICK_METRICS
    if preset != "Quick":
        names += (*STANDARD_METRICS, *JUDGE_OPERATIONS)
    if preset == "Deep":
        names += DEEP_METRICS
    return names


def run_cases(
    cases: list[GoldenCase],
    settings: RunSettings,
    progress: Callable[[int, int], None] | None = None,
) -> list[CaseResult]:
    """Keep failed cases visible while allowing later cases to run."""
    settings.validate()
    results = []
    for index, case in enumerate(cases, 1):
        try:
            observation = (
                run_existing_chroma(
                    case,
                    chroma_path=settings.chroma_path,
                    collection_name=settings.collection_name,
                    embedding=settings.embedding,
                    generation=settings.generation,
                    top_k=settings.top_k,
                )
                if settings.mode == "Existing Chroma"
                else precomputed_observation(case)
            )
            result = evaluate_quick(case, top_k=settings.top_k, observation=observation)
            if settings.preset != "Quick":
                try:
                    if settings.preset == "Standard":
                        result = evaluate_standard(
                            case,
                            top_k=settings.top_k,
                            judge=settings.judge,
                            observation=observation,
                        )
                    else:
                        result = evaluate_deep(
                            case,
                            top_k=settings.top_k,
                            judge=settings.judge,
                            observation=observation,
                        )
                except Exception as exc:
                    reason = f"judge failed ({type(exc).__name__})"
                    names = (*STANDARD_METRICS, *JUDGE_OPERATIONS)
                    if settings.preset == "Deep":
                        names += DEEP_METRICS
                    result = CaseResult(
                        case.id,
                        case.category,
                        observation,
                        {
                            **result.metrics,
                            **{name: MetricValue(skip_reason=reason) for name in names},
                        },
                        judge_details={"error": reason},
                    )
        except Exception as exc:
            # Provider exceptions can contain request data; export only the type.
            reason = f"case failed ({type(exc).__name__})"
            result = CaseResult(
                case.id,
                case.category,
                Observation(),
                {
                    name: MetricValue(skip_reason=reason)
                    for name in _expected_metrics(settings.preset)
                },
                judge_details={"error": reason},
            )
        results.append(result)
        if progress:
            progress(index, len(cases))
    return results
