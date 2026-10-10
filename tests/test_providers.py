from types import SimpleNamespace

from rag_eval_studio import providers
from rag_eval_studio.models import ProviderConfig


def test_provider_config_hides_credentials():
    config = ProviderConfig("openai", "small", "secret-key", "http://example")
    assert config.model_id == "openai/small"
    assert "secret-key" not in repr(config)
    assert "example" not in repr(config)


def test_provider_calls_pass_credentials_without_logging(monkeypatch):
    calls = []

    def embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[{"embedding": [0.1, 0.2]}], usage={"total_tokens": 3}
        )

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )

    monkeypatch.setattr(providers.litellm, "embedding", embedding)
    monkeypatch.setattr(providers.litellm, "completion", completion)
    monkeypatch.setattr(providers.litellm, "completion_cost", lambda **_: 0.01)
    config = ProviderConfig("openai", "model", "secret-key", "https://api.example")
    embedded = providers.embed_query("Question", config)
    generated = providers.generate([{"role": "user", "content": "Q"}], config)
    assert embedded.vector == [0.1, 0.2] and embedded.tokens == 3
    assert generated.text == "Answer" and generated.completion_tokens == 2
    assert all(call["api_key"] == "secret-key" for call in calls)
    assert all(call["model"] == "openai/model" for call in calls)


def test_unknown_model_price_does_not_fail_generation(monkeypatch):
    monkeypatch.setattr(
        providers.litellm,
        "completion",
        lambda **_: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
            usage={"prompt_tokens": 2, "completion_tokens": 1},
        ),
    )

    def unknown_price(**_):
        raise RuntimeError("no price")

    monkeypatch.setattr(providers.litellm, "completion_cost", unknown_price)
    result = providers.generate([], ProviderConfig("openai", "unpriced"))
    assert result.text == "Answer"
    assert result.cost_usd is None


def test_ollama_generation_disables_thinking_without_api_key(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage={"prompt_tokens": 4, "completion_tokens": 6},
        )

    monkeypatch.setattr(providers.litellm, "completion", completion)
    monkeypatch.setattr(providers.litellm, "completion_cost", lambda **_: None)
    config = ProviderConfig("ollama", "qwen3.5:4b", base_url="http://127.0.0.1:11435")
    result = providers.generate(
        [{"role": "user", "content": "Return JSON"}],
        config,
        response_format={"type": "json_object"},
    )
    assert result.text == '{"ok": true}'
    assert calls == [
        {
            "messages": [{"role": "user", "content": "Return JSON"}],
            "model": "ollama/qwen3.5:4b",
            "api_base": "http://127.0.0.1:11435",
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        }
    ]
