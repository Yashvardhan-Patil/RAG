# Documentation

## Quick Start

Choose **Precomputed Results**, upload a JSONL or CSV dataset, keep **Quick**,
and select **Run Evaluation**. The included `examples/golden.jsonl` works
without API keys. Open **Overview** for scores, **Cases** for individual runs,
and **Diagnostics** for skipped metrics and the JSON download.

## Evaluation Modes

**Existing Chroma** runs the studio's simple retrieval-and-answer pipeline on
your collection. **Precomputed Results** scores outputs produced by your own
pipeline; it makes no generation or embedding calls.

## Existing Chroma

Supply the persistence directory, collection name, embedding provider/model,
and generation provider/model. The query embedding model must match the
collection's indexed dimensions and embedding space. Use **Test connections**
before a full run. The collection is read, not rebuilt.

## Precomputed Results

Put `response`, `retrieved_contexts`, and `retrieved_doc_ids` in each record
when available. Quick needs no provider. Standard and Deep need a judge model.
Imported latency, token, and cost fields are shown only when supplied.

## Golden Dataset Format

JSONL contains one JSON object per line. `question` and `reference_answer`
are required. `id`, `category`, `relevant_doc_ids`, `reference_contexts`,
`response`, `retrieved_contexts`, and `retrieved_doc_ids` are optional. If
retrieved IDs and contexts are both present, their arrays must align. CSV is
also accepted; array-valued cells contain JSON arrays.

```jsonl
{"id":"capital-1","question":"What is the capital of France?","reference_answer":"Paris is the capital of France.","category":"direct_fact","relevant_doc_ids":["france-capital"],"response":"Paris is the capital of France.","retrieved_doc_ids":["france-capital"],"retrieved_contexts":["Paris is the capital of France."]}
```

## Metrics

Hit Rate@K, Precision@K, Recall@K, MRR@K, and nDCG@K require explicit,
complete `relevant_doc_ids` and ranked `retrieved_doc_ids` at the same
document or chunk granularity. Without those labels, they are skipped, not
estimated from keywords. Standard adds judge-based faithfulness, answer
relevance, factual correctness, context precision, and claim recall. Deep
adds noise sensitivity and citation checks when applicable. Retrieval,
generation, robustness, latency, and cost remain separate dimensions.

## Quick / Standard / Deep

- **Quick:** deterministic retrieval and supplied operational measurements;
  no judge calls.
- **Standard:** Quick plus answer and context quality checks; judge calls incur
  provider cost.
- **Deep:** Standard plus noise and citation diagnostics when inputs allow;
  may make extra judge calls.

## LLM Judge Configuration

For Existing Chroma, you may reuse the generation model or supply a separate
judge. Precomputed Results requires a judge provider/model for Standard and
Deep. Judge verdicts can be biased or inconsistent; review the case evidence
and calibrate against human-labeled examples before relying on them.

## API Keys and Privacy

Keys are used for the current run and are never written to JSON exports.
The app binds to localhost by default and does not share publicly. A provider
receives the text sent to it for embedding, generation, or judging.

## Common Errors

- **Dataset missing or invalid:** upload JSONL/CSV with a valid object per row
  and both required fields.
- **IR metrics skipped:** add complete `relevant_doc_ids` and aligned ranked
  `retrieved_doc_ids`.
- **Judge required:** configure one for Standard or Deep, or choose Quick.
- **Chroma connection failed:** check the persistence path, collection name,
  and embedding model compatibility.

## Result Export

Use **Diagnostics → Download JSON report** after a run. The report includes
inputs, per-case scores, skip reasons, judge evidence, run metadata, and a
dataset hash. It excludes provider API keys. Aggregate confidence intervals
describe variation across cases, not judge validity.
