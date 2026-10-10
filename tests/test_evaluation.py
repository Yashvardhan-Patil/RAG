from rag_eval_studio import judges
from rag_eval_studio.dataset import parse_case
from rag_eval_studio.evaluation import evaluate_quick, evaluate_standard
from rag_eval_studio.judges import JudgeOutcome
from rag_eval_studio.models import MetricValue, Observation, ProviderConfig


def test_precomputed_quick_preserves_skips_and_available_metadata():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "response": "A",
            "retrieved_doc_ids": ["doc"],
            "relevant_doc_ids": ["doc"],
            "retrieval_latency_ms": 4.5,
            "generation_prompt_tokens": 10,
        }
    )
    result = evaluate_quick(case, top_k=1)
    assert result.metrics["hit_rate"].value == 1
    assert result.metrics["retrieval_latency_ms"].value == 4.5
    assert result.metrics["generation_prompt_tokens"].value == 10
    assert result.metrics["total_latency_ms"].skip_reason
    assert result.metrics["total_tokens"].skip_reason
    assert result.metrics["total_cost_usd"].skip_reason


def test_live_observation_overrides_precomputed_fields():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "retrieved_doc_ids": ["wrong"],
            "relevant_doc_ids": ["right"],
        }
    )
    observed = Observation(retrieved_doc_ids=("right",), response="A")
    result = evaluate_quick(case, top_k=1, observation=observed)
    assert result.metrics["hit_rate"].value == 1
    assert result.observation.response == "A"


def test_standard_totals_include_judge_usage_when_all_stages_are_known(monkeypatch):
    case = parse_case({"question": "Q", "reference_answer": "A"})
    observed = Observation(
        embedding_tokens=3,
        generation_prompt_tokens=4,
        generation_completion_tokens=5,
        retrieval_cost_usd=0.2,
        generation_cost_usd=0.3,
    )
    monkeypatch.setattr(
        judges,
        "judge_standard",
        lambda *_: JudgeOutcome(
            {
                "judge_prompt_tokens": MetricValue(value=2),
                "judge_completion_tokens": MetricValue(value=1),
                "judge_cost_usd": MetricValue(value=0.1),
            },
            {},
        ),
    )
    result = evaluate_standard(
        case,
        top_k=1,
        judge=ProviderConfig("openai", "judge"),
        observation=observed,
    )
    assert result.metrics["total_tokens"].value == 15
    assert result.metrics["total_cost_usd"].value == 0.6
