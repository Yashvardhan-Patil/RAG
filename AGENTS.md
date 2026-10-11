# AGENTS.md — Development Guidelines for AI Agents

## Overview
**RAG Eval Studio** is a lightweight local dashboard for evaluating retrieval and generation quality in Retrieval-Augmented Generation (RAG) pipelines. It keeps retrieval metrics, answer generation, robustness, latency, and cost strictly separated, providing per-case evidence and explicit skip reasons when metrics cannot be computed.

## Core Architecture
- **Language & Runtime**: Python >=3.12 (managed via `uv`).
- **Web UI**: Gradio (`src/rag_eval_studio/app.py`), running locally on `127.0.0.1:7860`.
- **Vector Storage**: ChromaDB (`chromadb.PersistentClient`) for direct collection evaluation.
- **Model Interoperability**: LiteLLM for routing queries, generations, and judge calls to local Ollama or cloud providers.
- **Data Schemas**: Canonical JSONL and CSV support via dataclasses in `models.py` and parsers in `dataset.py`.
- **Metrics Engine**:
  - `retrieval.py`: Classical IR metrics (Hit Rate@K, Precision@K, Recall@K, MRR@K, nDCG@K).
  - `judges.py`: LLM-as-a-judge claim extraction and entailment verdicts (Faithfulness, Answer Relevance, Factual Correctness, Context Precision, Claim Recall, Noise Sensitivity, Citations).
  - `statistics.py`: Percentile bootstrap confidence intervals (n >= 5) and categorical aggregations.

## Rules of Engagement (Ponytail Mode)
Follow the principles defined in `.agents/rules/ponytail.md`:
1. **YAGNI & Simplicity**: Never build unrequested abstractions or introduce unneeded dependencies.
2. **Standard Library & Existing Helpers First**: Reuse existing dataclasses, utilities, and typing patterns.
3. **Root Cause over Symptom**: When fixing an issue, address the shared logic rather than patching individual callers.
4. **Credential Isolation**: Never persist API keys, tokens, or raw provider exceptions into logs, reports, or committed files.
5. **Executable Verification**: Ensure changes pass `uv run pytest` and `uv run ruff check .`.

## Useful Commands
- Sync dependencies: `uv sync --group dev`
- Run test suite: `uv run pytest`
- Lint checks: `uv run ruff check .`
- Formatting check: `uv run ruff format --check .`
- Launch dashboard: `uv run rag-eval-studio`
