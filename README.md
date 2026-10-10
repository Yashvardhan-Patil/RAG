# RAG Eval Studio

A local dashboard for evaluating retrieval and answer quality in RAG systems.
It keeps retrieval, generation, robustness, latency, and cost separate, with
per-case evidence and explicit reasons when a metric cannot be scored.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run rag-eval-studio
```

Open the local address printed by Gradio. The app binds to `127.0.0.1` by
default. API keys entered in the form stay in memory for the current run and
are omitted from exports. Text sent for embedding, generation, or judging is
transmitted to the configured provider; use local providers for local text
processing.

The left sidebar opens **Evaluation**, in-app **Documentation**, **API Docs**,
**Settings**, and **About**. Settings explains how per-run configuration and
API keys are handled. The evaluation screen walks through source,
configuration, dataset, and preset. Results appear after a run under Overview,
Retrieval, Generation (for judge presets), Cases, and Diagnostics.

To try the dashboard without API calls, keep the default **Precomputed Results**
source and **Quick** preset, then upload
[`examples/golden.jsonl`](examples/golden.jsonl). For Standard or Deep, supply a
judge model; these presets make provider calls.

## Input modes

**Existing Chroma** reads a persisted Chroma collection, embeds each question
with the configured provider/model, retrieves `top_k` chunks, and calls the
configured generation model once. Use the same embedding space and dimension
that built the collection. This mode evaluates the app's simple answer prompt
over your collection. If your application has custom retrieval, reranking, or
prompting, capture its outputs and use **Precomputed Results**.

**Precomputed Results** reads response and retrieval fields from the dataset.
It does not rerun your RAG pipeline. Latency, tokens, and cost appear only if
the record contains those measurements. Use **Test connections** to make small
provider calls before a Standard or Deep evaluation.

The canonical dataset is JSONL, one object per line. `question` and
`reference_answer` are required. `id` is optional and generated from their
contents if missing. Optional fields include `category`, `relevant_doc_ids`,
`reference_contexts`, `response`, `retrieved_contexts`, `retrieved_doc_ids`,
and structured `citations`. CSV is also accepted; array-valued cells must
contain JSON arrays. See [the input contract](docs/ARCHITECTURE.md) for timing,
usage, cost, and citation fields.

For local Ollama, set the provider to `ollama`, use an embedding model such as
`nomic-embed-text`, and use a generation/judge model such as `qwen3.5:4b`.
Set the matching base URL for each active provider in Advanced options (for
example, `http://127.0.0.1:11434` for a standard local Ollama installation).
An Existing Chroma collection must have been built with a compatible embedding
model. Model availability and speed depend on your machine. Standard and Deep
make judge calls even when generation is local.

Classical Hit Rate@K, Precision@K, Recall@K, MRR@K, and nDCG@K require a
complete, nonempty set of explicit `relevant_doc_ids` at the same document or
chunk granularity as `retrieved_doc_ids`. Without it, those metrics are skipped.
An empty retrieved list with valid labels scores zero. Context Precision and
Claim Recall are separate judge-based measures; they do not substitute for
document-label metrics. Duplicate retrieved IDs are invalid: dataset import
rejects them, and direct scoring skips the classical metrics with a reason.

## Presets and results

| Preset | Adds |
| --- | --- |
| Quick | Labeled-ID retrieval metrics and available latency, usage, cost |
| Standard | Faithfulness, answer relevance, factual correctness, context precision, claim recall |
| Deep | Noise sensitivity and citation support/coverage when structured citations exist |

Every aggregate shows its eligible case count. A fixed-seed case bootstrap
provides a 95% interval when five or more cases are eligible. Smaller samples
show the mean and count without an interval. The interval
describes variation across sampled cases; it cannot validate an LLM judge.
The JSON download includes source records, scores, skips, judge verdicts, and
run metadata without credentials.

Judge metrics use claim extraction and structured entailment verdicts. They
can be biased or inconsistent. Review per-case verdicts and calibrate the
judge against human-labeled examples before relying on them for decisions.
Reusing the generation model as judge is allowed but labeled in the report.
Answer Relevance is a direct judge rubric, not Ragas's embedding-based response
relevancy. Noise sensitivity and citation diagnostics are experimental; see the
[metric audit](docs/METRIC_AUDIT.md) for exact formulas, variants, and skip rules.
See [research and metric definitions](docs/RESEARCH.md) and
[architecture and limitations](docs/ARCHITECTURE.md).

RAG Eval Studio was inspired by the evaluation dashboard in Ed Donner's
*LLM Engineering* Week 5 course. It is an independent project and is not
affiliated with or endorsed by the course or its author.

## Develop

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests use temporary Chroma data and mocked provider calls. They require no
API keys or GPU.
