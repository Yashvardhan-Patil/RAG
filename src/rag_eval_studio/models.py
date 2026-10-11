"""Data shared by dataset import and evaluation."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Citation:
    claim: str
    doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    question: str
    reference_answer: str
    category: str | None = None
    relevant_doc_ids: tuple[str, ...] | None = None
    reference_contexts: tuple[str, ...] | None = None
    response: str | None = None
    retrieved_contexts: tuple[str, ...] | None = None
    retrieved_doc_ids: tuple[str, ...] | None = None
    citations: tuple[Citation, ...] | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    embedding_tokens: int | None = None
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    retrieval_cost_usd: float | None = None
    generation_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = field(default=None, repr=False)

    @property
    def model_id(self) -> str:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model are required")
        model = self.model.strip()
        return model if "/" in model else f"{self.provider.strip()}/{model}"


@dataclass(frozen=True, slots=True)
class Observation:
    response: str | None = None
    retrieved_contexts: tuple[str, ...] | None = None
    retrieved_doc_ids: tuple[str, ...] | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    embedding_tokens: int | None = None
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    retrieval_cost_usd: float | None = None
    generation_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    category: str | None
    observation: Observation
    metrics: dict[str, "MetricValue"]
    judge_details: dict | None = None


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.skip_reason is None):
            raise ValueError("exactly one of value or skip_reason is required")
