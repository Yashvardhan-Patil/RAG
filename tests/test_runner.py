import json
from pathlib import Path

import gradio as gr
import pytest

from rag_eval_studio import app, judges, runner
from rag_eval_studio.dataset import parse_case
from rag_eval_studio.models import ProviderConfig
from rag_eval_studio.providers import GenerationResult
from rag_eval_studio.report import diagnostic_rows, export_report
from rag_eval_studio.runner import RunSettings, run_cases


def test_precomputed_quick_runs_without_providers(tmp_path):
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "response": "A",
            "relevant_doc_ids": ["doc"],
            "retrieved_doc_ids": ["doc"],
        }
    )
    path = tmp_path / "input.jsonl"
    path.write_text('{"question":"Q","reference_answer":"A"}\n', encoding="utf-8")
    settings = RunSettings("Precomputed Results", "Quick", 1)
    results = run_cases([case], settings)
    assert results[0].metrics["hit_rate"].value == 1
    assert results[0].metrics["retrieval_latency_ms"].skip_reason
    assert (
        export_report([case], results, settings, path)["cases"][0]["input"]["question"]
        == "Q"
    )


def test_case_failure_is_sanitized_and_later_cases_continue(monkeypatch):
    cases = [
        parse_case({"id": "first", "question": "Q1", "reference_answer": "A"}),
        parse_case({"id": "second", "question": "Q2", "reference_answer": "A"}),
    ]

    original = runner.evaluate_quick

    def fake(case, **kwargs):
        if case.id == "first":
            raise RuntimeError("secret-key in provider error")
        return original(case, **kwargs)

    monkeypatch.setattr(runner, "evaluate_quick", fake)
    results = run_cases(cases, RunSettings("Precomputed Results", "Quick", 1))
    assert "secret-key" not in str(results)
    assert results[0].metrics["hit_rate"].skip_reason == "case failed (RuntimeError)"
    assert results[1].case_id == "second"


def test_judge_failure_keeps_retrieval_scores(monkeypatch):
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "relevant_doc_ids": ["doc"],
            "retrieved_doc_ids": ["doc"],
        }
    )

    def fail_judge(*_args, **_kwargs):
        raise RuntimeError("secret-key in judge error")

    monkeypatch.setattr(runner, "evaluate_standard", fail_judge)
    settings = RunSettings(
        "Precomputed Results", "Standard", 1, judge=ProviderConfig("openai", "judge")
    )
    result = run_cases([case], settings)[0]
    assert result.metrics["hit_rate"].value == 1
    assert result.metrics["faithfulness"].skip_reason == "judge failed (RuntimeError)"
    assert "secret-key" not in str(result)
    assert diagnostic_rows([result])[0][-1] == "judge failed (RuntimeError)"


def test_report_excludes_provider_credentials(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"question":"Q","reference_answer":"A"}\n', encoding="utf-8")
    case = parse_case({"question": "Q", "reference_answer": "A"})
    config = ProviderConfig("openai", "model", "secret-key", "https://api.example")
    settings = RunSettings(
        "Precomputed Results", "Standard", 1, generation=config, judge=config
    )
    results = run_cases([case], RunSettings("Precomputed Results", "Quick", 1))
    report = export_report([case], results, settings, path)
    serialized = json.dumps(report)
    assert "secret-key" not in serialized
    assert "api.example" not in serialized
    assert report["judge_reuses_generation"]


def test_gradio_builds_without_launching():
    config = app.build_app().get_config_file()
    components = config["components"]
    assert any(
        item["type"] == "sidebar" and item["props"]["open"] is False
        for item in components
    )
    assert any(
        item["type"] == "markdown"
        and item["props"].get("value", "").startswith("# Documentation")
        for item in components
    )
    assert any(
        item["type"] == "markdown"
        and item["props"].get("value", "").startswith("# Settings")
        for item in components
    )
    assert app._view("Documentation")[1]["visible"]
    assert app._view("Settings")[3]["visible"]
    assert not app._mode_visibility("Precomputed Results", "Quick", True)[-1]["visible"]
    assert app._preset_visibility("Standard", "Precomputed Results")[0]["visible"]


def test_precomputed_quick_dashboard_callback_exports_diagnostics(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Q",
                "reference_answer": "A",
                "response": "A",
                "relevant_doc_ids": ["doc"],
                "retrieved_doc_ids": ["doc"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    form = ("Precomputed Results", "Quick", 1, *([None] * 15))
    settings = app._settings(*form)
    assert settings.embedding is settings.generation is settings.judge is None
    (
        status,
        overview,
        _,
        diagnostics,
        skips,
        _,
        entries,
        export_path,
        detail,
        retrieval,
        generation,
        empty_state,
        results_view,
        generation_tab,
    ) = app._run(str(dataset), *form, progress=lambda *_args, **_kwargs: None)
    assert "1 cases evaluated" in status
    assert any(row[1] == "Hit Rate@K" and row[2] == "1.000" for row in overview)
    assert diagnostics[0][-1] == "OK"
    assert "retrieval_latency_ms" in skips
    assert entries[0] == detail
    assert retrieval[0][1] == "Hit Rate@K"
    assert not generation
    assert not empty_state["visible"] and results_view["visible"]
    assert not generation_tab["visible"]
    exported = json.loads(Path(export_path).read_text())
    assert exported["cases"][0]["result"]["metrics"]["hit_rate"]["value"] == 1


@pytest.mark.parametrize("preset", ["Standard", "Deep"])
def test_precomputed_judge_presets_ignore_hidden_provider_fields(preset):
    form = ["Precomputed Results", preset, 1, *([None] * 15)]
    form[14:18] = ["openai", "judge", None, None]
    settings = app._settings(*form)
    settings.validate()
    assert settings.embedding is settings.generation is None
    assert settings.judge.model_id == "openai/judge"


@pytest.mark.parametrize("preset", ["Standard", "Deep"])
def test_missing_required_judge_is_a_validation_error(tmp_path, preset):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"question":"Q","reference_answer":"A"}\n')
    form = ("Precomputed Results", preset, 1, *([None] * 15))
    with pytest.raises(gr.Error, match="judge provider and model"):
        app._run(str(dataset), *form, progress=lambda *_args, **_kwargs: None)


def test_chroma_requires_only_its_active_provider_config():
    form = ["Existing Chroma", "Quick", 1, "db", "collection", *([None] * 13)]
    with pytest.raises(ValueError, match="embedding provider and model"):
        app._settings(*form).validate()
    form[5:7] = ["openai", "embed"]
    with pytest.raises(ValueError, match="generation provider and model"):
        app._settings(*form).validate()
    form[9:11] = ["openai", "generate"]
    settings = app._settings(*form)
    settings.validate()
    assert settings.judge is None


def test_chroma_standard_reuses_generation_without_judge_fields():
    form = ["Existing Chroma", "Standard", 1, "db", "collection", *([None] * 13)]
    form[5:7] = ["openai", "embed"]
    form[9:11] = ["openai", "generate"]
    form[13] = True
    settings = app._settings(*form)
    settings.validate()
    assert settings.judge is settings.generation

    form[13] = False
    with pytest.raises(ValueError, match="judge provider and model"):
        app._settings(*form).validate()


def test_connection_check_only_calls_judge_for_standard(monkeypatch):
    form = ["Precomputed Results", "Quick", 1, *([""] * 15)]
    assert "No external connection" in app._test_connection(*form)

    called = []

    def fake_generate(messages, config, *, response_format):
        called.append((config.model_id, response_format))
        return GenerationResult('{"ok": true}', 1, 1, 0.0)

    monkeypatch.setattr(app, "generate", fake_generate)
    form[1] = "Standard"
    form[14] = "openai"
    form[15] = "judge"
    assert "judge JSON output succeeded" in app._test_connection(*form)
    assert called == [("openai/judge", {"type": "json_object"})]


def test_precomputed_standard_dashboard_uses_mock_judge_and_hides_keys(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "standard.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Where is Paris?",
                "reference_answer": "Paris is in France.",
                "response": "Paris is in France.",
                "relevant_doc_ids": ["doc"],
                "retrieved_doc_ids": ["doc"],
                "retrieved_contexts": ["Paris is in France."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = iter(
        [
            {
                "reference_claims": ["Paris is in France"],
                "response_claims": ["Paris is in France"],
            },
            {
                "response_checks": [
                    {
                        "context_status": "supported",
                        "reference_status": "supported",
                        "supporting_context_ids": ["doc"],
                    }
                ],
                "reference_checks": [
                    {
                        "response_status": "supported",
                        "context_status": "supported",
                    }
                ],
                "chunk_checks": [{"id": "doc", "relevant": True}],
                "answer_relevance": 1.0,
            },
        ]
    )

    def fake_generate(_messages, _config, *, response_format):
        assert response_format == {"type": "json_object"}
        return GenerationResult(json.dumps(next(outputs)), 10, 5, 0.01)

    monkeypatch.setattr(judges, "generate", fake_generate)
    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    form = (
        "Precomputed Results",
        "Standard",
        1,
        "",
        "",
        "openai",
        "",
        "",
        "",
        "openai",
        "",
        "generation-secret",
        "",
        True,
        "openai",
        "judge",
        "judge-secret",
        "",
    )
    output = app._run(str(dataset), *form, progress=lambda *_args, **_kwargs: None)
    status, overview, export_path = output[0], output[1], output[7]
    assert "1 cases evaluated" in status
    assert any(row[1] == "Faithfulness" and row[2] == "1.000" for row in overview)
    assert any(row[1] == "Faithfulness" for row in output[10])
    assert output[13]["visible"]
    exported = Path(export_path).read_text(encoding="utf-8")
    assert "generation-secret" not in exported
    assert "judge-secret" not in exported
