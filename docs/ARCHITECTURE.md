# v1 architecture

## Scope and flow

Python 3.12, uv, Gradio, Chroma, LiteLLM, pytest, and ruff. Keep Ragas,
RAGChecker, DeepEval, TruLens, LangChain, pandas, and SciPy out of the runtime
dependency set. Use the standard library for validation, JSONL/CSV, and the
fixed-seed percentile bootstrap. Gradio's own dependencies can serve tables
and plots. The app binds to `127.0.0.1` by default.

`dataset` loads canonical JSONL into typed records. Each row requires a
nonblank `question` and `reference_answer`; absent IDs are generated
deterministically and must be unique. Optional: `category`,
`relevant_doc_ids`, `reference_contexts`, `response`, `retrieved_contexts`,
`retrieved_doc_ids`. Optional imported timing/usage/citation fields must have
documented schemas and remain separate from the canonical minimum. Arrays of
retrieved contexts and IDs must align when both appear. A row with an empty
response is incomplete, not a zero-scoring answer. CSV import accepts JSON
arrays in list-valued cells, without guessing separators. Optional imported
operational fields are `retrieval_latency_ms`, `generation_latency_ms`,
`total_latency_ms`, `embedding_tokens`, `generation_prompt_tokens`,
`generation_completion_tokens`, `retrieval_cost_usd`, and
`generation_cost_usd`; absent values remain skipped.
Structured `citations` is an optional JSON array of objects with nonblank
`claim` text and a nonempty `doc_ids` array. IDs refer to retrieved chunks;
unknown IDs are counted as unresolved and cannot support a claim.

Mode 1 opens an **existing** named collection via Chroma `PersistentClient`
and `get_collection`; never creates or mutates it. LiteLLM embeds the query
using the user's chosen model, then Chroma queries `query_embeddings` for K
ranked IDs and texts. The embedding model must match the indexed vector space
and dimension. The application sends those contexts to the configured
generation model with a documented prompt. This evaluates the app's simple
pipeline over the existing Chroma collection, not an arbitrary user's custom
RAG orchestration. For that, use Mode 2.

Mode 2 evaluates precomputed `response`, `retrieved_contexts`, and
`retrieved_doc_ids` without retrieval or generation calls. Partial rows remain
valid; unavailable metrics get explicit skip reasons. Imported stage timings,
usage, and citations are optional. Neither cost nor latency is inferred from
text alone.

Provider configurations each hold model, API key, and optional base URL in
memory only. LiteLLM accepts provider-qualified model names; expose an optional
provider field in UI and form the model identifier once. Embedding, generation,
and optional judge configurations are independent. Reuse generation as judge
is explicit and marked in the report. Check a connection only on user action;
never call external APIs on file upload. Catch and display provider errors
without exposing keys. No automatic global LiteLLM logging/callbacks.

The evaluator assembles one immutable per-case result with observed outputs,
deterministic scores, judge scores, skips, evidence, timings, and usage. Judge
metrics operate on the *same* captured response and contexts as retrieval
metrics. The aggregator computes dimension-specific means, counts, category
breakdowns and CIs. Export JSON/JSONL with run metadata and per-case results,
excluding credentials and raw provider exceptions that might contain secrets.
`total_latency_ms` covers retrieval and generation only. `judge_latency_ms`
is separate. `total_tokens` and `total_cost_usd` include the judge stage in
Standard/Deep only when every required stage is known; otherwise they skip.

## Presets

| Preset | Evaluations | Expected incremental judge work |
| --- | --- | --- |
| Quick | Five labeled-ID retrieval metrics, latency, token usage, cost | none |
| Standard | Quick + faithfulness, answer relevance, factual correctness, context precision, claim recall | claim extraction/checks and one relevance call per eligible case |
| Deep | Standard + noise sensitivity and citation support/coverage when structured citations exist | additional chunk attribution and citation checks; highest cost |

All presets report available measurements and skip unavailable metrics.
Standard and Deep require a judge configuration. A dataset can contain cases
with different supervision; report eligibility per metric. No universal score.

Standard uses two JSON-mode judge calls per eligible case: atomic claim
extraction, then verdicts for each claim and retrieved chunk plus a direct
answer-relevance rubric. It validates verdict count and chunk IDs, retries
malformed structured output once, and retains the prompt version and verdicts
for inspection. A provider that cannot return JSON objects cannot run this
preset; affected judge metrics show a skip rather than a fabricated score.
Deep derives noise sensitivity from those verdicts only when an irrelevant
chunk and assessable claim correctness exist. Structured citations trigger an
additional JSON-mode call that maps each response claim to cited chunk IDs and
checks support. Deep counts unresolved IDs separately. The extra call is
skipped when no structured citations were supplied.

## Planned small modules

- `models.py`: dataset/config/result dataclasses and field contracts.
- `dataset.py`: JSONL/CSV import and validation.
- `retrieval.py`: labeled-ID ranking metrics.
- `statistics.py`: aggregation and bootstrap.
- `providers.py`: thin LiteLLM calls, usage and cost capture.
- `pipeline.py`: read-only Chroma retrieval and simple generation prompt.
- `judges.py`: versioned prompts and validated structured judgments.
- `evaluation.py`: case orchestration and explicit skips.
- `app.py`: Gradio setup, progress, diagnostics, and export.

Implementation order: foundation and deterministic metrics; adapters and
provider calls; judge metrics; dashboard; release documentation. Avoid adding
abstractions until a second implementation needs them.

## Primary API references

- [Chroma Python client](https://docs.trychroma.com/reference/python)
- [LiteLLM SDK](https://docs.litellm.ai/docs/)
- [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)
