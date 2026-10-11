# RAG evaluation research (2026-09-25)

## Reference implementation

The read-only Week 5 project uses markdown files split into 500-character chunks
with 200-character overlap, OpenAI `text-embedding-3-large`, a persisted Chroma
store, and `gpt-4.1-nano` via LangChain for answers. Its retriever defaults to
10 results; `fetch_context(question)` passes `k` to `invoke` even though the
retriever is constructed without search kwargs, so the requested cutoff may
not be applied as intended. A second LiteLLM call judges each answer.

`evaluation/tests.jsonl` holds `question`, `keywords`, `reference_answer`, and
`category`. The evaluator separately shows average retrieval and answer scores
and a category bar chart. Preserving this separation and category view is useful.

The reported MRR is the mean reciprocal rank of *keywords found as substrings*
in retrieved chunks, not reciprocal rank of a relevant document. Its nDCG uses
the same keyword proxy and calculates the ideal ordering only from retrieved
items; it cannot penalize a missed relevant item. Keyword coverage is a useful
informal diagnostic but cannot establish relevance. Retrieval and answer runs
call the pipeline separately, so they may assess different contexts or outputs.
The answer judge reports opaque 1–5 ratings; per-case evidence, missing-input
states, usage, uncertainty, and latency are absent. Both UIs assume the single
course pipeline. Neither the course dependencies nor LangChain are needed in
the new core.

## Metric contract

All rates are per case in [0, 1], then averaged across *eligible* cases. A
skipped metric has a reason and is excluded from its mean; it is never zero.
`K` means the first `min(K, available)` ranked results, but Precision@K uses
the configured K as denominator. Document IDs must use the same stable chunk
or document granularity on both sides. Explicit `relevant_doc_ids` is assumed
to be a complete set for the case; incomplete judgments make recall/nDCG
optimistic or misleading. Duplicate retrieved IDs are invalid. Empty explicit
relevance sets are marked ineligible for the five classical metrics.

| Metric | Meaning/formula | Required inputs; skip condition | Type; incremental cost |
| --- | --- | --- | --- |
| Hit Rate@K | 1 if any top-K ID is relevant, else 0 | ranked IDs, nonempty complete relevant ID set; skip if absent | deterministic; none |
| Precision@K | relevant top-K IDs / K | same; skip if absent | deterministic; none |
| Recall@K | relevant top-K IDs / all relevant IDs | same; skip if absent | deterministic; none |
| MRR@K | 1 / rank of first relevant ID in top K, else 0 | same; skip if absent | deterministic; none |
| nDCG@K | DCG/ideal DCG, binary gain `rel_i/log2(i+1)` and ideal has `min(K, relevant count)` hits | same; skip if absent | deterministic; none |
| Context Precision | average precision at ranks judged useful for answering the *reference* answer | question, reference answer, ranked context texts, judge; skip if unavailable/empty | judge-based; one or more calls per case, often grows with chunks |
| Context Recall / Claim Recall | reference-answer claims entailed by retrieved contexts / all reference claims | reference answer, context texts, judge; skip if unavailable/empty or no factual claims | judge-based; claim extraction + entailment calls |
| Faithfulness / Groundedness | response claims entailed by retrieved contexts / all response claims | response, context texts, judge; skip if unavailable/empty or no factual claims | judge-based; claim extraction + entailment calls |
| Answer Relevance | rubric score for whether response addresses question, independent of factual truth | question, response, judge; skip if unavailable/empty | judge-based; one call or more |
| Factual Correctness | claim precision/recall/F1 of response against reference answer, including contradictions | response, reference answer, judge; skip if unavailable/empty or no factual claims | judge-based; extraction + entailment calls |
| Noise Sensitivity | fraction of response claims judged incorrect against reference and traceable to irrelevant retrieved chunks (lower is better) | reference, response, contexts, judge and chunk relevance judgments; skip if unavailable/empty or no assessable claims | judge-based; expensive, Deep only |
| Citation support | fraction of cited factual claims entailed by their cited chunks; show unresolved citation IDs separately | structured claim-to-chunk citations and cited chunk texts, judge; skip if no citations/schema or evidence | judge-based; extraction + per-citation entailment calls |
| Citation coverage | factual claims with a valid supporting citation / all factual claims | same; skip if absent/no claims | judge-based; same calls as citation support |
| Retrieval latency | elapsed embedding + query time in ms | measured live retrieval or imported timing; skip otherwise | measured; no judge cost |
| Generation latency | elapsed answer-model call time in ms | measured live generation or imported timing; skip otherwise | measured; no judge cost |
| Total latency | elapsed retrieval + generation in ms, excluding evaluation | both stage times or imported end-to-end timing; skip otherwise | measured; no judge cost |
| Token usage | prompt, completion, total tokens by stage | provider metadata or imported usage; skip missing fields | observed; no extra call |
| Cost | USD by embedding, generation, judge and total where known | provider-reported cost or known model price + usage; skip unknown components, never treat as zero | estimated or reported; no extra call |

Context Precision here is *judge-assessed ranking precision*, separate from
ID-based Precision@K; it should never silently substitute for that metric.
Ragas also offers reference-free context utilization; it tests usefulness to
the produced response and is not equivalent to relevance to the ground truth.
Likewise, claim recall is evidence coverage, not document recall. We use a
direct relevance rubric for Answer Relevance; it does not reproduce Ragas's
reverse-question embedding measure, and the UI must identify the method.

For statistics, show case count and eligible count with every aggregate.
Calculate category means only from eligible cases. Use a fixed-seed,
case-resampled percentile bootstrap (e.g. 2,000 resamples) for 95% CIs of
means when at least five eligible cases exist. Smaller samples show no CI.
Intervals quantify sampling variation across cases, not judge validity or
provider drift. For later side-by-side comparisons, resample paired case IDs.
The reference answer may be incomplete or allow valid alternatives, so judge
correctness and claim recall inherit the quality of that supervision.

## Judge design and limitations

Decompose factual sentences into atomic, independently checkable claims.
Require structured outputs containing each claim, a supported/unsupported/
uncertain verdict and evidence chunk IDs. The current protocol does not ask
for or store free-form rationales or chain-of-thought. Validate the
schema and IDs; retry malformed outputs once, then mark the metric skipped with
an error reason. `uncertain` must not be counted as supported. Avoid a single
undifferentiated 1–5 score. Fix prompt version, model, temperature where
supported, top K, dataset hash, and run time in the result. Repeated calls may
still vary; calibrate judges against a small human-labeled set and inspect
disagreements. A separate judge model is preferable when practical; reusing
the generator is convenient but exposes self-preference bias. Pairwise judges
can show order bias; absolute, claim-level checks reduce that exposure but do
not remove judge bias. Long and fluent answers can receive undue credit.

Citation evaluation requires an explicit, documented citation-to-chunk ID
mapping. A citation marker alone is not evidence that the cited text supports
its associated claim. Missing or unresolved citations are reported separately.
Noise Sensitivity needs the reference answer to identify wrong claims and
chunk relevance to attribute them; an arbitrary noise perturbation would be a
different robustness experiment and must be named separately.

## Sources

- [NIST TREC relevance judgments](https://trec.nist.gov/data/reljudge_eng.html), [TREC 2021 precision and recall definitions](https://trec.nist.gov/pubs/trec30/papers/Overview-2021.pdf), [TREC graded nDCG](https://trec.nist.gov/pubs/trec19/papers/WEB.OVERVIEW.pdf)
- [Ragas context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/), [faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/), [factual correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/), [noise sensitivity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/), [response relevancy](https://docs.ragas.io/en/v0.2.7/concepts/metrics/available_metrics/answer_relevance/)
- [RAGChecker paper and implementation](https://github.com/amazon-science/RAGChecker), [claim-level methodology](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md)
- [DeepEval metric documentation](https://docs.confident-ai.com/docs/metrics-introduction), [TruLens RAG triad](https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/)
- [ALCE citation evaluation](https://github.com/princeton-nlp/ALCE), [ALCE paper](https://aclanthology.org/2023.emnlp-main.398.pdf)
- [LLM judge position bias](https://arxiv.org/abs/2406.07791), [self-preference bias](https://arxiv.org/abs/2410.21819), [verbosity bias](https://arxiv.org/abs/2409.16788)
- [SciPy bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
