"""Small LiteLLM wrapper that passes credentials only to individual calls."""

import math
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from rag_eval_studio.models import ProviderConfig

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
litellm = import_module("litellm")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vector: list[float]
    tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None


def _kwargs(config: ProviderConfig) -> dict[str, str]:
    options = {"model": config.model_id}
    if config.api_key:
        options["api_key"] = config.api_key
    if config.base_url:
        options["api_base"] = config.base_url
    return options


def _usage(response: Any, name: str) -> int | None:
    usage = response.get("usage") if hasattr(response, "get") else response.usage
    if usage is None:
        return None
    value = usage.get(name) if hasattr(usage, "get") else getattr(usage, name, None)
    return value if type(value) is int and value >= 0 else None


def _cost(response: Any) -> float | None:
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        # Price lookup is optional; an unknown model must not fail evaluation.
        return None
    if cost is None:
        return None
    value = float(cost)
    return value if math.isfinite(value) and value >= 0 else None


def embed_query(text: str, config: ProviderConfig) -> EmbeddingResult:
    response = litellm.embedding(input=[text], **_kwargs(config))
    vector = response.data[0]["embedding"]
    return EmbeddingResult(
        vector=list(vector),
        tokens=_usage(response, "total_tokens"),
        cost_usd=_cost(response),
    )


def generate(
    messages: list[dict[str, str]],
    config: ProviderConfig,
    *,
    response_format: dict[str, Any] | None = None,
) -> GenerationResult:
    options: dict[str, Any] = _kwargs(config)
    if config.provider == "ollama":
        # Thinking models can return no final JSON while spending time in reasoning.
        options["reasoning_effort"] = "none"
    if response_format is not None:
        options["response_format"] = response_format
    response = litellm.completion(messages=messages, **options)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("generation model returned no text")
    return GenerationResult(
        text=content,
        prompt_tokens=_usage(response, "prompt_tokens"),
        completion_tokens=_usage(response, "completion_tokens"),
        cost_usd=_cost(response),
    )
