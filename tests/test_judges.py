import json

import pytest

from rag_eval_studio import judges
from rag_eval_studio.dataset import parse_case
from rag_eval_studio.evaluation import evaluate_deep, evaluate_standard
from rag_eval_studio.models import Observation, ProviderConfig
from rag_eval_studio.providers import GenerationResult


def _mock_calls(monkeypatch, outputs):
    pending = iter(outputs)
    calls = []

    def fake_generate(messages, _config, *, response_format):
        calls.append((messages, response_format))
        return GenerationResult(next(pending), 10, 5, 0.01)

    monkeypatch.setattr(judges, "generate", fake_generate)
    return calls


def test_standard_scores_claim_evidence_and_tracks_judge_usage(monkeypatch):
    calls = _mock_calls(
        monkeypatch,
        [
            json.dumps(
                {
                    "reference_claims": ["Paris is in France"],
                    "response_claims": ["Paris is in France", "Paris is on Mars"],
                }
            ),
            json.dumps(
                {
                    "response_checks": [
                        {
                            "context_status": "supported",
                            "reference_status": "supported",
                            "supporting_context_ids": ["good"],
                        },
                        {
                            "context_status": "unsupported",
                            "reference_status": "unsupported",
                            "supporting_context_ids": [],
                        },
                    ],
                    "reference_checks": [
                        {"response_status": "supported", "context_status": "supported"}
                    ],
                    "chunk_checks": [
                        {"id": "good", "relevant": True},
                        {"id": "noise", "relevant": False},
                    ],
                    "answer_relevance": 0.8,
                }
            ),
        ],
    )
    case = parse_case({"question": "Where is Paris?", "reference_answer": "France"})
    observed = Observation(
        response="Paris is in France and on Mars",
        retrieved_doc_ids=("good", "noise"),
        retrieved_contexts=("Paris is in France", "unrelated"),
    )
    result = evaluate_standard(
        case,
        top_k=2,
        judge=ProviderConfig("openai", "judge"),
        observation=observed,
    )
    assert result.metrics["faithfulness"].value == 0.5
    assert result.metrics["factual_correctness"].value == pytest.approx(2 / 3)
    assert result.metrics["context_precision"].value == 1
    assert result.metrics["claim_recall"].value == 1
    assert result.metrics["answer_relevance"].value == 0.8
    assert result.metrics["judge_prompt_tokens"].value == 20
    assert result.metrics["judge_cost_usd"].value == 0.02
    assert result.judge_details["prompt_version"] == judges.PROMPT_VERSION
    assert all(format == {"type": "json_object"} for _, format in calls)


def test_standard_scores_available_dimensions_without_contexts(monkeypatch):
    _mock_calls(
        monkeypatch,
        [
            json.dumps({"reference_claims": ["A"], "response_claims": ["A"]}),
            json.dumps(
                {
                    "response_checks": [
                        {
                            "context_status": "uncertain",
                            "reference_status": "supported",
                            "supporting_context_ids": [],
                        }
                    ],
                    "reference_checks": [
                        {"response_status": "supported", "context_status": "uncertain"}
                    ],
                    "chunk_checks": [],
                    "answer_relevance": 1,
                }
            ),
        ],
    )
    case = parse_case({"question": "Q", "reference_answer": "A", "response": "A"})
    result = evaluate_standard(case, top_k=1, judge=ProviderConfig("openai", "judge"))
    assert result.metrics["factual_correctness"].value == 1
    assert result.metrics["faithfulness"].skip_reason
    assert result.metrics["claim_recall"].skip_reason


def test_malformed_judge_output_is_retried_then_skipped(monkeypatch):
    calls = _mock_calls(monkeypatch, ["{}", "not JSON"])
    case = parse_case({"question": "Q", "reference_answer": "A", "response": "A"})
    result = evaluate_standard(case, top_k=1, judge=ProviderConfig("openai", "judge"))
    assert len(calls) == 2
    assert result.metrics["faithfulness"].skip_reason == "judge output was invalid"
    assert result.metrics["judge_prompt_tokens"].value == 20


def test_missing_chunk_relevance_is_retried_then_skipped(monkeypatch):
    assessment = json.dumps(
        {
            "response_checks": [
                {
                    "context_status": "supported",
                    "reference_status": "supported",
                    "supporting_context_ids": ["doc"],
                }
            ],
            "reference_checks": [
                {"response_status": "supported", "context_status": "supported"}
            ],
            "chunk_checks": [{"id": "doc"}],
            "answer_relevance": 1,
        }
    )
    calls = _mock_calls(
        monkeypatch,
        [
            json.dumps({"reference_claims": ["A"], "response_claims": ["A"]}),
            assessment,
            assessment,
        ],
    )
    case = parse_case({"question": "Q", "reference_answer": "A"})
    observed = Observation(
        response="A", retrieved_doc_ids=("doc",), retrieved_contexts=("A",)
    )
    result = evaluate_standard(
        case,
        top_k=1,
        judge=ProviderConfig("openai", "judge"),
        observation=observed,
    )
    assert len(calls) == 3
    assert result.metrics["context_precision"].skip_reason == "judge output was invalid"
    assert result.metrics["judge_prompt_tokens"].value == 30


def test_ollama_assessment_uses_schema_for_required_fields(monkeypatch):
    calls = _mock_calls(
        monkeypatch,
        [
            json.dumps({"reference_claims": ["A"], "response_claims": ["A"]}),
            json.dumps(
                {
                    "response_checks": [
                        {
                            "context_status": "supported",
                            "reference_status": "supported",
                            "supporting_context_ids": ["doc"],
                        }
                    ],
                    "reference_checks": [
                        {"response_status": "supported", "context_status": "supported"}
                    ],
                    "chunk_checks": [{"id": "doc", "relevant": True}],
                    "answer_relevance": 1,
                }
            ),
        ],
    )
    case = parse_case({"question": "Q", "reference_answer": "A"})
    observed = Observation(
        response="A", retrieved_doc_ids=("doc",), retrieved_contexts=("A",)
    )
    result = evaluate_standard(
        case,
        top_k=1,
        judge=ProviderConfig("ollama", "local"),
        observation=observed,
    )
    assert result.metrics["faithfulness"].value == 1
    assert calls[0][1] == {"type": "json_object"}
    schema_format = calls[1][1]
    assert schema_format["type"] == "json_schema"
    schema = schema_format["json_schema"]["schema"]
    assert schema["properties"]["chunk_checks"]["minItems"] == 1
    assert "relevant" in schema["properties"]["chunk_checks"]["items"]["required"]


def test_deep_attributes_wrong_claim_to_noise_and_checks_citations(monkeypatch):
    _mock_calls(
        monkeypatch,
        [
            json.dumps(
                {
                    "reference_claims": ["Paris is in France"],
                    "response_claims": ["Paris is in France", "Paris is on Mars"],
                }
            ),
            json.dumps(
                {
                    "response_checks": [
                        {
                            "context_status": "supported",
                            "reference_status": "supported",
                            "supporting_context_ids": ["good"],
                        },
                        {
                            "context_status": "supported",
                            "reference_status": "unsupported",
                            "supporting_context_ids": ["noise"],
                        },
                    ],
                    "reference_checks": [
                        {"response_status": "supported", "context_status": "supported"}
                    ],
                    "chunk_checks": [
                        {"id": "good", "relevant": True},
                        {"id": "noise", "relevant": False},
                    ],
                    "answer_relevance": 0.8,
                }
            ),
            json.dumps(
                {
                    "citation_checks": [
                        {"cited_doc_ids": ["good"], "supported": True},
                        {"cited_doc_ids": ["noise"], "supported": False},
                    ]
                }
            ),
        ],
    )
    case = parse_case(
        {
            "question": "Where is Paris?",
            "reference_answer": "Paris is in France",
            "citations": [
                {"claim": "Paris is in France", "doc_ids": ["good"]},
                {"claim": "Paris is on Mars", "doc_ids": ["noise"]},
            ],
        }
    )
    observed = Observation(
        response="Paris is in France and on Mars",
        retrieved_doc_ids=("good", "noise"),
        retrieved_contexts=("Paris is in France", "Paris is on Mars"),
    )
    result = evaluate_deep(
        case,
        top_k=2,
        judge=ProviderConfig("openai", "judge"),
        observation=observed,
    )
    assert result.metrics["noise_sensitivity"].value == 0.5
    assert result.metrics["citation_support"].value == 0.5
    assert result.metrics["citation_coverage"].value == 0.5
    assert result.metrics["citation_unresolved_ids"].value == 0
    assert result.metrics["judge_cost_usd"].value == 0.03
    assert len(result.judge_details["citation_checks"]) == 2


def _audit_standard(
    monkeypatch,
    *,
    response_context=("supported",),
    response_reference=None,
    reference_response=None,
    reference_context=None,
    chunk_relevance=(True,),
    answer_relevance=1,
):
    response_reference = response_reference or ("supported",) * len(response_context)
    reference_response = reference_response or ("supported",) * len(response_context)
    reference_context = reference_context or ("supported",) * len(reference_response)
    response_claims = [f"response {i}" for i in range(len(response_context))]
    reference_claims = [f"reference {i}" for i in range(len(reference_response))]
    ids = [f"doc-{i}" for i in range(len(chunk_relevance))]
    extraction = {
        "response_claims": response_claims,
        "reference_claims": reference_claims,
    }
    assessment = {
        "response_checks": [
            {
                "context_status": context,
                "reference_status": reference,
                "supporting_context_ids": [ids[0]] if context == "supported" else [],
            }
            for context, reference in zip(
                response_context, response_reference, strict=True
            )
        ],
        "reference_checks": [
            {"response_status": response, "context_status": context}
            for response, context in zip(
                reference_response, reference_context, strict=True
            )
        ],
        "chunk_checks": [
            {"id": doc_id, "relevant": relevant}
            for doc_id, relevant in zip(ids, chunk_relevance, strict=True)
        ],
        "answer_relevance": answer_relevance,
        "reasoning": "private deliberation must not be exported",
    }
    _mock_calls(monkeypatch, [json.dumps(extraction), json.dumps(assessment)])
    case = parse_case({"question": "Q", "reference_answer": "Reference answer"})
    observed = Observation(
        response="Answer", retrieved_doc_ids=tuple(ids), retrieved_contexts=tuple(ids)
    )
    return evaluate_standard(
        case,
        top_k=len(ids),
        judge=ProviderConfig("openai", "judge"),
        observation=observed,
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("supported", "supported"), 1),
        (("supported", "unsupported"), 0.5),
        (("unsupported", "unsupported"), 0),
    ],
)
def test_audit_faithfulness_direction(monkeypatch, statuses, expected):
    result = _audit_standard(monkeypatch, response_context=statuses)
    assert result.metrics["faithfulness"].value == expected
    assert "private deliberation" not in json.dumps(result.judge_details)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("supported", "supported"), 1),
        (("supported", "unsupported"), 0.5),
        (("unsupported", "unsupported"), 0),
    ],
)
def test_audit_claim_recall_direction(monkeypatch, statuses, expected):
    result = _audit_standard(
        monkeypatch,
        response_context=("supported", "supported"),
        reference_context=statuses,
    )
    assert result.metrics["claim_recall"].value == expected


@pytest.mark.parametrize(
    ("response_reference", "reference_response", "expected"),
    [
        (("supported",), ("supported",), 1),
        (("supported", "unsupported"), ("supported",), 2 / 3),
        (("supported",), ("supported", "unsupported"), 2 / 3),
    ],
)
def test_audit_factual_correctness_direction(
    monkeypatch, response_reference, reference_response, expected
):
    result = _audit_standard(
        monkeypatch,
        response_context=("supported",) * len(response_reference),
        response_reference=response_reference,
        reference_response=reference_response,
    )
    assert result.metrics["factual_correctness"].value == pytest.approx(expected)


def test_audit_context_precision_rewards_earlier_relevance(monkeypatch):
    first = _audit_standard(monkeypatch, chunk_relevance=(True, False, False))
    last = _audit_standard(monkeypatch, chunk_relevance=(False, False, True))
    assert first.metrics["context_precision"].value == 1
    assert last.metrics["context_precision"].value == pytest.approx(1 / 3)


def test_audit_uncertain_chunk_relevance_skips_precision(monkeypatch):
    result = _audit_standard(monkeypatch, chunk_relevance=(None,))
    assert result.metrics["context_precision"].value is None
    assert (
        result.metrics["context_precision"].skip_reason
        == "chunk relevance is uncertain"
    )


def test_audit_missing_citation_verdict_skips_scores(monkeypatch):
    missing = json.dumps({"citation_checks": [{"cited_doc_ids": ["doc"]}]})
    _mock_calls(monkeypatch, [missing, missing])
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "citations": [{"claim": "A", "doc_ids": ["doc"]}],
        }
    )
    observed = Observation(
        response="A", retrieved_doc_ids=("doc",), retrieved_contexts=("A",)
    )
    details = {"response_claims": ["A"], "response_checks": [], "chunk_checks": []}
    result = judges.judge_deep(
        case, observed, ProviderConfig("openai", "judge"), details
    )
    assert (
        result.metrics["citation_support"].skip_reason
        == "judge citation output was invalid"
    )
    assert (
        result.metrics["citation_coverage"].skip_reason
        == "judge citation output was invalid"
    )
    assert result.metrics["citation_unresolved_ids"].value == 0


def test_audit_missing_citations_skip_all_citation_fields():
    case = parse_case({"question": "Q", "reference_answer": "A"})
    result = judges.judge_deep(
        case,
        Observation(response="A"),
        ProviderConfig("openai", "judge"),
        {"response_claims": ["A"], "response_checks": [], "chunk_checks": []},
    )
    assert all(
        result.metrics[name].skip_reason == "structured citations are unavailable"
        for name in ("citation_support", "citation_coverage", "citation_unresolved_ids")
    )


def test_audit_no_judge_call_marks_operational_fields_skipped():
    case = parse_case({"question": "Q", "reference_answer": "A"})
    result = evaluate_standard(case, top_k=1, judge=ProviderConfig("openai", "judge"))
    assert all(
        result.metrics[name].skip_reason
        == "response and retrieved contexts are unavailable"
        for name in (
            "judge_latency_ms",
            "judge_prompt_tokens",
            "judge_completion_tokens",
            "judge_cost_usd",
        )
    )


def test_audit_zero_response_claims_do_not_create_faithfulness_score(monkeypatch):
    _mock_calls(
        monkeypatch,
        [
            json.dumps({"response_claims": [], "reference_claims": ["A"]}),
            json.dumps(
                {
                    "response_checks": [],
                    "reference_checks": [
                        {
                            "response_status": "unsupported",
                            "context_status": "supported",
                        }
                    ],
                    "chunk_checks": [{"id": "doc", "relevant": True}],
                    "answer_relevance": 0.5,
                }
            ),
        ],
    )
    case = parse_case({"question": "Q", "reference_answer": "A"})
    observed = Observation(
        response="I do not know", retrieved_doc_ids=("doc",), retrieved_contexts=("A",)
    )
    result = evaluate_standard(
        case, top_k=1, judge=ProviderConfig("openai", "judge"), observation=observed
    )
    assert (
        result.metrics["faithfulness"].skip_reason == "response has no factual claims"
    )
    assert result.metrics["factual_correctness"].value == 0
    assert result.metrics["claim_recall"].value == 1


def test_audit_out_of_range_relevance_skips_judge_metrics(monkeypatch):
    extraction = json.dumps({"response_claims": ["A"], "reference_claims": ["A"]})
    invalid = json.dumps(
        {
            "response_checks": [
                {
                    "context_status": "supported",
                    "reference_status": "supported",
                    "supporting_context_ids": ["doc"],
                }
            ],
            "reference_checks": [
                {"response_status": "supported", "context_status": "supported"}
            ],
            "chunk_checks": [{"id": "doc", "relevant": True}],
            "answer_relevance": 1.2,
        }
    )
    _mock_calls(monkeypatch, [extraction, invalid, invalid])
    case = parse_case({"question": "Q", "reference_answer": "A"})
    observed = Observation(
        response="A", retrieved_doc_ids=("doc",), retrieved_contexts=("A",)
    )
    result = evaluate_standard(
        case, top_k=1, judge=ProviderConfig("openai", "judge"), observation=observed
    )
    assert all(result.metrics[name].value is None for name in judges.STANDARD_METRICS)
    assert result.metrics["judge_prompt_tokens"].value == 30
