"""Local Gradio dashboard for RAG evaluation."""

import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from rag_eval_studio.dataset import DatasetError, load_cases, parse_case
from rag_eval_studio.models import ProviderConfig
from rag_eval_studio.pipeline import run_existing_chroma
from rag_eval_studio.providers import generate
from rag_eval_studio.report import (
    category_rows,
    diagnostic_rows,
    export_report,
    skip_summary,
    summary_rows,
    write_report,
)
from rag_eval_studio.runner import MODES, PRESETS, RunSettings, run_cases


def _text(value: str | None) -> str:
    return (value or "").strip()


def _provider(
    provider: str | None,
    model: str | None,
    key: str | None,
    base_url: str | None,
) -> ProviderConfig | None:
    if not _text(provider) or not _text(model):
        return None
    return ProviderConfig(
        _text(provider), _text(model), _text(key) or None, _text(base_url) or None
    )


def _settings(
    mode: str,
    preset: str,
    top_k: int,
    chroma_path: str | None,
    collection: str | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_key: str | None,
    embedding_base_url: str | None,
    generation_provider: str | None,
    generation_model: str | None,
    generation_key: str | None,
    generation_base_url: str | None,
    reuse_judge: bool,
    judge_provider: str | None,
    judge_model: str | None,
    judge_key: str | None,
    judge_base_url: str | None,
) -> RunSettings:
    chroma = mode == "Existing Chroma"
    embedding = (
        _provider(
            embedding_provider, embedding_model, embedding_key, embedding_base_url
        )
        if chroma
        else None
    )
    generation = (
        _provider(
            generation_provider, generation_model, generation_key, generation_base_url
        )
        if chroma
        else None
    )
    judge = None
    if preset != "Quick":
        judge = (
            generation
            if chroma and reuse_judge
            else _provider(judge_provider, judge_model, judge_key, judge_base_url)
        )
    return RunSettings(
        mode=mode,
        preset=preset,
        top_k=int(top_k),
        chroma_path=(_text(chroma_path) or None) if chroma else None,
        collection_name=(_text(collection) or None) if chroma else None,
        embedding=embedding,
        generation=generation,
        judge=judge,
    )


def _run(dataset_path: str, *form_values, progress=gr.Progress()):  # noqa: B008
    if not dataset_path:
        raise gr.Error("Upload a JSONL or CSV dataset first.")
    try:
        cases = load_cases(dataset_path)
        settings = _settings(*form_values)
        settings.validate()
    except (DatasetError, ValueError) as exc:
        raise gr.Error(str(exc)) from None

    results = run_cases(
        cases,
        settings,
        progress=lambda index, total: progress(
            index / total, desc=f"Evaluating {index}/{total}"
        ),
    )
    report = export_report(cases, results, settings, dataset_path)
    with tempfile.NamedTemporaryFile(
        prefix="rag-eval-report-", suffix=".json", delete=False
    ) as file:
        export_path = write_report(report, file.name)
    entries = report["cases"]
    failed = sum("error" in (result.judge_details or {}) for result in results)
    status = (
        f"**{len(results)} cases evaluated · {failed} case failures.** "
        "Scores show eligible cases only. Bootstrap intervals represent "
        "case sampling, not judge accuracy."
    )
    if settings.preset == "Quick":
        status += " Generation quality needs Standard or Deep and a judge."
    if report["judge_reuses_generation"]:
        status += (
            " The generation model was reused as judge; self-evaluation bias may apply."
        )
    if any(result.metrics["total_cost_usd"].value is not None for result in results):
        status += " Provider costs are estimates when inferred from a model price map."
    first_id = results[0].case_id
    overview = summary_rows(results)
    return (
        status,
        overview,
        category_rows(results),
        diagnostic_rows(results),
        skip_summary(results),
        gr.Dropdown(choices=[result.case_id for result in results], value=first_id),
        entries,
        str(export_path),
        entries[0],
        [row for row in overview if row[0] == "Retrieval"],
        [row for row in overview if row[0] in {"Generation", "Robustness / citations"}],
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=settings.preset != "Quick"),
    )


def _select_case(case_id: str, entries: list[dict]) -> dict:
    return next(
        (entry for entry in entries if entry["result"]["case_id"] == case_id), {}
    )


def _test_connection(*form_values) -> str:
    try:
        settings = _settings(*form_values)
        settings.validate()
    except ValueError as exc:
        return f"Complete the configuration first: {exc}"
    try:
        checks = []
        if settings.mode == "Existing Chroma":
            sample = parse_case(
                {"question": "Connection test", "reference_answer": "Connection test"}
            )
            run_existing_chroma(
                sample,
                chroma_path=settings.chroma_path,
                collection_name=settings.collection_name,
                embedding=settings.embedding,
                generation=settings.generation,
                top_k=1,
            )
            checks.append("Chroma, embedding, and generation succeeded")
        if settings.preset != "Quick":
            response = generate(
                [
                    {"role": "system", "content": "Return a JSON object."},
                    {"role": "user", "content": 'Return {"ok": true}.'},
                ],
                settings.judge,
                response_format={"type": "json_object"},
            )
            if not isinstance(json.loads(response.text), dict):
                raise ValueError("judge did not return a JSON object")
            checks.append("judge JSON output succeeded")
        return (
            "; ".join(checks)
            if checks
            else "No external connection needed in Quick precomputed mode."
        )
    except Exception as exc:
        return (
            f"Connection test failed ({type(exc).__name__}). "
            "Check model, key, base URL, and collection."
        )


def _view(name: str) -> tuple[dict, ...]:
    return tuple(
        gr.update(visible=name == page)
        for page in ("Evaluation", "Documentation", "API Docs", "Settings", "About")
    )


def _mode_visibility(mode: str, preset: str, reuse_judge: bool) -> tuple[dict, ...]:
    chroma = mode == "Existing Chroma"
    return (
        gr.update(visible=chroma),
        gr.update(visible=not chroma),
        gr.update(visible=chroma),
        gr.update(visible=not chroma or not reuse_judge),
        gr.update(visible=chroma),
        gr.update(visible=chroma or preset != "Quick"),
    )


def _preset_visibility(preset: str, mode: str) -> tuple[dict, dict]:
    return (
        gr.update(visible=preset != "Quick"),
        gr.update(visible=mode == "Existing Chroma" or preset != "Quick"),
    )


CUSTOM_CSS = """
/* RAG Eval Studio — Clean, Restrained Developer Interface */
:root {
  --res-bg: #fcfcfd;
  --res-surface: #ffffff;
  --res-border: #e2e8f0;
  --res-border-focus: #3b82f6;
  --res-text-main: #0f172a;
  --res-text-muted: #64748b;
  --res-accent: #2563eb;
  --res-accent-hover: #1d4ed8;
}

body, .gradio-container {
  background-color: var(--res-bg) !important;
  color: var(--res-text-main) !important;
}

/* Sidebar styling */
.sidebar, [data-testid="sidebar"] {
  background-color: #f8fafc !important;
  border-right: 1px solid var(--res-border) !important;
}

.sidebar h2 {
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  color: #0f172a !important;
  letter-spacing: -0.02em !important;
  padding-bottom: 8px !important;
  border-bottom: 1px solid var(--res-border) !important;
  margin-bottom: 12px !important;
}

.sidebar button {
  text-align: left !important;
  justify-content: flex-start !important;
  border-radius: 6px !important;
  font-size: 0.885rem !important;
  font-weight: 500 !important;
  color: #334155 !important;
  padding: 8px 12px !important;
  margin-bottom: 4px !important;
  transition: all 0.15s ease !important;
}

.sidebar button:hover {
  background-color: #f1f5f9 !important;
  color: #0f172a !important;
}

.sidebar button.primary {
  background-color: #eff6ff !important;
  color: #1d4ed8 !important;
  border: 1px solid #bfdbfe !important;
  font-weight: 600 !important;
}

/* Form groups & clean containers */
.gr-group {
  border: 1px solid var(--res-border) !important;
  background-color: var(--res-surface) !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
  padding: 16px !important;
  margin-bottom: 14px !important;
}

/* Headings */
h1 {
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em !important;
  color: #0f172a !important;
}

h2 {
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  letter-spacing: -0.015em !important;
  color: #1e293b !important;
  margin-top: 1rem !important;
  margin-bottom: 0.35rem !important;
}

/* Input fields */
input[type="text"], input[type="password"], textarea, select {
  border: 1px solid #cbd5e1 !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  font-size: 0.9rem !important;
  color: #0f172a !important;
}

input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
  border-color: var(--res-border-focus) !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
  outline: none !important;
}

/* Action button */
button.primary:not(.sidebar button), .gr-button.primary:not(.sidebar button) {
  background-color: var(--res-accent) !important;
  color: #ffffff !important;
  border: 1px solid var(--res-accent) !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  box-shadow: 0 1px 2px rgba(37, 99, 235, 0.18) !important;
  transition: background-color 0.15s ease !important;
}

button.primary:not(.sidebar button):hover,
.gr-button.primary:not(.sidebar button):hover {
  background-color: var(--res-accent-hover) !important;
}

/* Tables and Dataframes */
.gr-dataframe {
  border: 1px solid var(--res-border) !important;
  border-radius: 6px !important;
}

.gr-dataframe th {
  background-color: #f8fafc !important;
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.03em !important;
  border-bottom: 1px solid var(--res-border) !important;
  padding: 10px 14px !important;
}

.gr-dataframe td {
  padding: 10px 14px !important;
  font-size: 0.885rem !important;
  color: #1e293b !important;
  border-bottom: 1px solid #f1f5f9 !important;
}

/* Tab bar */
button.tab-nav {
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  color: #64748b !important;
  border-bottom: 2px solid transparent !important;
}

button.tab-nav.selected {
  color: var(--res-accent) !important;
  border-bottom: 2px solid var(--res-accent) !important;
  font-weight: 600 !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Eval Studio") as app:
        with gr.Sidebar(open=False, width=250):
            gr.Markdown("## RAG Eval Studio")
            evaluation_nav = gr.Button("Evaluation", variant="primary")
            documentation_nav = gr.Button("Documentation")
            api_nav = gr.Button("API Docs")
            settings_nav = gr.Button("Settings")
            about_nav = gr.Button("About")

        with gr.Column() as evaluation_view:
            gr.Markdown(
                "# New evaluation\nEvaluate retrieval and answer quality "
                "as separate, explainable results."
            )
            gr.Markdown("## 1. Choose a source")
            mode = gr.Radio(
                MODES,
                value="Precomputed Results",
                label="Source",
            )
            gr.Markdown(
                "**Existing Chroma:** Evaluate a RAG pipeline using an existing "
                "Chroma collection.  \n**Precomputed Results:** Evaluate responses "
                "and retrieved contexts produced by another RAG pipeline."
            )

            gr.Markdown("## 2. Configure")
            with gr.Group(visible=False) as chroma_group:
                with gr.Row():
                    chroma_path = gr.Textbox(label="Chroma path")
                    collection = gr.Textbox(label="Collection")
                gr.Markdown(
                    "Use the same embedding model and vector dimensions that "
                    "built this collection."
                )
                with gr.Row():
                    embedding_provider = gr.Textbox(
                        value="openai", label="Embedding provider"
                    )
                    embedding_model = gr.Textbox(label="Embedding model")
                embedding_key = gr.Textbox(label="Embedding API key", type="password")
                with gr.Accordion("Advanced embedding options", open=False):
                    embedding_base = gr.Textbox(label="Embedding base URL")

            with gr.Group(visible=False) as generation_group:
                gr.Markdown("### Generation model")
                with gr.Row():
                    generation_provider = gr.Textbox(value="openai", label="Provider")
                    generation_model = gr.Textbox(label="Model")
                generation_key = gr.Textbox(label="API key", type="password")
                with gr.Accordion("Advanced generation options", open=False):
                    generation_base = gr.Textbox(label="Generation base URL")

            with gr.Group() as precomputed_note:
                gr.Markdown(
                    "Your uploaded records supply the responses and retrieval "
                    "results. Quick needs no provider configuration."
                )

            gr.Markdown("## 3. Golden dataset")
            dataset = gr.File(
                label="Upload JSONL or CSV",
                file_types=[".jsonl", ".csv"],
                type="filepath",
            )
            gr.Markdown(
                "One record per line. `question` and `reference_answer` are "
                "required; IDs and retrieved results are optional."
            )
            dataset_help = gr.Button("View format and example", size="sm")

            gr.Markdown("## 4. Evaluation")
            preset = gr.Radio(PRESETS, value="Quick", label="Preset")
            gr.Markdown(
                "**Quick:** labeled retrieval and supplied timing/usage, no judge. "
                "**Standard:** adds answer and context checks (judge calls). "
                "**Deep:** adds noise and citation checks when inputs exist "
                "(more judge calls)."
            )
            top_k = gr.Slider(
                1, 100, value=10, step=1, label="Top K (retrieval cutoff)"
            )
            with gr.Group(visible=False) as judge_group:
                gr.Markdown("### Judge model · Standard and Deep")
                reuse = gr.Checkbox(
                    value=True, label="Reuse generation model as judge", visible=False
                )
                with gr.Group() as judge_fields:
                    with gr.Row():
                        judge_provider = gr.Textbox(
                            value="openai", label="Judge provider"
                        )
                        judge_model = gr.Textbox(label="Judge model")
                    judge_key = gr.Textbox(label="Judge API key", type="password")
                    with gr.Accordion("Advanced judge options", open=False):
                        judge_base = gr.Textbox(label="Judge base URL")
                gr.Markdown(
                    "Judge calls may incur cost. A separate judge avoids "
                    "self-evaluation by the generation model."
                )

            form = [
                mode,
                preset,
                top_k,
                chroma_path,
                collection,
                embedding_provider,
                embedding_model,
                embedding_key,
                embedding_base,
                generation_provider,
                generation_model,
                generation_key,
                generation_base,
                reuse,
                judge_provider,
                judge_model,
                judge_key,
                judge_base,
            ]
            run_button = gr.Button("Run Evaluation", variant="primary", size="lg")
            test_button = gr.Button("Test connections", visible=False)
            connection_status = gr.Markdown()
            mode.change(
                _mode_visibility,
                inputs=[mode, preset, reuse],
                outputs=[
                    chroma_group,
                    precomputed_note,
                    generation_group,
                    judge_fields,
                    reuse,
                    test_button,
                ],
                api_visibility="private",
            )
            preset.change(
                _preset_visibility,
                inputs=[preset, mode],
                outputs=[judge_group, test_button],
                api_visibility="private",
            )
            reuse.change(
                lambda mode_value, reuse_value: gr.update(
                    visible=mode_value == "Precomputed Results" or not reuse_value
                ),
                inputs=[mode, reuse],
                outputs=judge_fields,
                api_visibility="private",
            )
            test_button.click(
                _test_connection,
                inputs=form,
                outputs=connection_status,
                api_name="test_connections",
            )

            empty_state = gr.Markdown(
                "No evaluation yet. Choose a source and upload a golden dataset "
                "to begin."
            )
            with gr.Group(visible=False) as results_view:
                gr.Markdown("## Results")
                run_status = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        overview = gr.Dataframe(
                            headers=[
                                "Dimension",
                                "Metric",
                                "Mean",
                                "95% CI",
                                "Eligible",
                            ],
                            interactive=False,
                            wrap=True,
                        )
                        gr.Markdown("### By category")
                        categories = gr.Dataframe(
                            headers=[
                                "Category",
                                "Metric",
                                "Mean",
                                "95% CI",
                                "Eligible",
                            ],
                            interactive=False,
                        )
                    with gr.Tab("Retrieval"):
                        gr.Markdown(
                            "Classical retrieval scores need explicit relevant "
                            "document IDs. See Diagnostics for skips."
                        )
                        retrieval = gr.Dataframe(
                            headers=[
                                "Dimension",
                                "Metric",
                                "Mean",
                                "95% CI",
                                "Eligible",
                            ],
                            interactive=False,
                        )
                    with gr.Tab("Generation", visible=False) as generation_tab:
                        generation = gr.Dataframe(
                            headers=[
                                "Dimension",
                                "Metric",
                                "Mean",
                                "95% CI",
                                "Eligible",
                            ],
                            interactive=False,
                        )
                    with gr.Tab("Cases"):
                        diagnostics = gr.Dataframe(
                            headers=[
                                "Case ID",
                                "Category",
                                "Response",
                                "Skipped",
                                "Status",
                            ],
                            interactive=False,
                            wrap=True,
                        )
                        case_selector = gr.Dropdown(label="Inspect case", choices=[])
                        case_detail = gr.JSON(label="Scores, skips, and judge evidence")
                    with gr.Tab("Diagnostics"):
                        gr.Markdown("### Skipped metrics and failure reasons")
                        skips = gr.Markdown()
                        export_file = gr.File(
                            label="Download JSON report", interactive=False
                        )
                entries = gr.State([])
                case_selector.change(
                    _select_case,
                    inputs=[case_selector, entries],
                    outputs=case_detail,
                    api_visibility="private",
                )
            run_button.click(
                _run,
                inputs=[dataset, *form],
                outputs=[
                    run_status,
                    overview,
                    categories,
                    diagnostics,
                    skips,
                    case_selector,
                    entries,
                    export_file,
                    case_detail,
                    retrieval,
                    generation,
                    empty_state,
                    results_view,
                    generation_tab,
                ],
                api_name="run_evaluation",
            )

        with gr.Column(visible=False) as documentation_view:
            gr.Markdown(Path(__file__).with_name("HELP.md").read_text(encoding="utf-8"))

        with gr.Column(visible=False) as api_view:
            gr.Markdown(
                "# API Docs\nThe local Gradio API can run this dashboard from a "
                "Python client. Use the address printed when the app starts:\n\n"
                "```python\nfrom gradio_client import Client\n"
                "client = Client('http://127.0.0.1:7860')\n"
                "client.view_api()  # Lists run_evaluation and test_connections\n"
                "```\nThe server may choose another port if 7860 is busy. "
                "See [Gradio's API guide](https://gradio.app/guides/view-api-page) "
                "for client upload and prediction examples."
            )

        with gr.Column(visible=False) as settings_view:
            gr.Markdown(
                "# Settings\nChoose the source, models, judge, and retrieval cutoff "
                "on the Evaluation page for each run. API keys are used only "
                "during that run and are excluded from exported reports. "
                "The dashboard does not save provider settings."
            )

        with gr.Column(visible=False) as about_view:
            gr.Markdown(
                "# About RAG Eval Studio\nA small, local tool for evaluating "
                "retrieval and answer quality without a universal score. "
                "Built with [Gradio](https://gradio.app/). The full README "
                "is available in the repository checkout."
            )

        views = [
            evaluation_view,
            documentation_view,
            api_view,
            settings_view,
            about_view,
        ]
        evaluation_nav.click(
            lambda: _view("Evaluation"), outputs=views, api_visibility="private"
        )
        documentation_nav.click(
            lambda: _view("Documentation"), outputs=views, api_visibility="private"
        )
        api_nav.click(
            lambda: _view("API Docs"), outputs=views, api_visibility="private"
        )
        settings_nav.click(
            lambda: _view("Settings"), outputs=views, api_visibility="private"
        )
        about_nav.click(lambda: _view("About"), outputs=views, api_visibility="private")
        dataset_help.click(
            lambda: _view("Documentation"), outputs=views, api_visibility="private"
        )

    @app.app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    port_env = os.environ.get("PORT") or os.environ.get("RAG_EVAL_PORT") or "7860"
    port = int(port_env)
    host = os.environ.get("RAG_EVAL_HOST", "0.0.0.0")
    os.environ["GRADIO_SERVER_NAME"] = host
    os.environ["GRADIO_SERVER_PORT"] = str(port)
    print(f"Starting RAG Eval Studio on {host}:{port}...")
    # Gradio checks its own localhost route; bypass ambient HTTP proxies for it.
    for name in ("NO_PROXY", "no_proxy"):
        entries = [
            entry.strip() for entry in os.environ.get(name, "").split(",") if entry
        ]
        if "127.0.0.1" not in entries:
            entries.append("127.0.0.1")
        if "0.0.0.0" not in entries:
            entries.append("0.0.0.0")
        os.environ[name] = ",".join(entries)
    app = build_app()
    app.queue().launch(
        _app=app.app,
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=False,
        footer_links=[],
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=[
                "system-ui",
                "Segoe UI",
                "Roboto",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
            font_mono=["ui-monospace", "Consolas", "monospace"],
        ),
    )


if __name__ == "__main__":
    main()
