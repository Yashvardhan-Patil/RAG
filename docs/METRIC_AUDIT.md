# Metric and release audit for v0.1.0

Audited 2026-09-26. This is a code and semantic review, backed by hand-calculated
fixtures and mocked judge outputs. The two-case live Ollama check in
`docs/agent/HANDOFF.md` confirms wiring, not metric validity. A judge score is
an estimate conditional on its extracted claims and verdicts; it is not a human
annotation. No Ragas or RAGChecker API compatibility is claimed.

**Classification.** VERIFIED means the implemented arithmetic or measurement
matches the stated contract in the tested conditions. VERIFIED WITH DOCUMENTED
VARIANT means the calculation is internally sound but its rubric, decomposition,
or empty-input convention differs from a named reference. EXPERIMENTAL means
that evidence attribution or pricing has not been calibrated against an
independent ground truth. BROKEN metrics must be fixed or disabled before
release. No BROKEN metric remains enabled after this audit.

## Independent retrieval checks

For relevant IDs `{A,C,D}`, ranked IDs `[A,B,C]`, and `K=3`, hand calculation
gives Hit Rate `1`, Precision `2/3`, Recall `2/3`, MRR `1`, and binary
`DCG = 1/log2(2) + 1/log2(4) = 1.5`. The ideal has all three relevant IDs
at ranks 1–3: `IDCG = 1 + 1/log2(3) + 1/log2(4) = 2.1309297536`.
Thus `nDCG = 1.5/2.1309297536 = 0.7039180890`. A temporary scikit-learn
`ndcg_score([[1,0,1,1]], [[3,2,1,0]], k=3)` independently returned
`0.7039180890341348`; scikit-learn was **not** added to project dependencies.
For a single relevant item at rank 3, hand calculation and scikit-learn both
give nDCG `0.5`.

Other fixed expectations in `tests/test_retrieval.py`: first hit at rank 3 has
MRR `1/3`; no hits produce five zeros; three relevant IDs ranked first produce
five ones; `K=1` ignores later hits; `K=3` with only one returned relevant ID
uses `1/3` for Precision and `1/3` for Recall; empty retrieval with complete
nonempty labels produces zeros. Absent or empty labels skip all five metrics.
Duplicate ranked IDs are invalid: `parse_case` rejects them at import, and
`score_retrieval` now skips all five metrics if passed a directly constructed
case with duplicates. Counting repeated IDs as separate hits previously could
make Recall or nDCG exceed 1.

All five use `retrieval.score_retrieval` in `src/rag_eval_studio/retrieval.py`,
require complete nonempty `relevant_doc_ids`, aligned ranked
`retrieved_doc_ids`, and `K >= 1`, and are deterministic with range `[0,1]`.
Missing IDs or labels skip; `K < 1` raises `ValueError`. The labels must be at
the same document/chunk granularity as returned IDs. Incomplete relevance
judgments can overstate Recall and nDCG. Definitions follow
[TREC relevance judgments](https://trec.nist.gov/data/reljudge_eng.html),
[TREC nDCG](https://trec.nist.gov/pubs/trec19/papers/WEB.OVERVIEW.pdf), and
[scikit-learn nDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).

| Metric | Exact per-case rule | Empty valid retrieval | Classification and result |
| --- | --- | --- | --- |
| Hit Rate@K | `1` if any top-K ID is relevant; else `0` | `0` | VERIFIED; hand fixtures pass |
| Precision@K | Number of relevant top-K IDs divided by configured `K`, including unfilled slots | `0` | VERIFIED; hand fixtures pass |
| Recall@K | Number of distinct relevant top-K IDs divided by all labeled relevant IDs | `0` | VERIFIED; hand fixtures pass |
| MRR@K | Reciprocal of the first relevant top-K rank; `0` if none | `0` | VERIFIED; hand fixtures pass |
| Binary nDCG@K | `sum(rel_i/log2(i+1)) / sum(1/log2(i+1), i=1..min(K, relevant count))` | `0` | VERIFIED; hand and scikit-learn checks pass |

The implementation counts hits rather than creating a second set in its
numerator; the duplicate-ID guard makes these rules equivalent for valid
input. Precision uses configured K even when fewer than K results arrive.

## Judge protocol and limits

`judges.judge_standard` first asks the judge to extract atomic factual claims
from the response and reference, then asks for claim entailment against the
reference and retrieved chunks, ordered chunk usefulness, and a direct answer
relevance score. It validates claim arrays, verdict vocabulary, expected
counts, exact chunk IDs, evidence IDs, required fields, and the `[0,1]` direct
score. Malformed JSON or schema is retried once; if still invalid, judge
metrics are skipped with a concise reason, never changed to zero. Provider
exceptions are sanitized by `runner.run_cases`; deterministic retrieval and
operational results already obtained remain available. `uncertain` does not
receive support credit. Free-form chain-of-thought is neither requested nor
persisted: exported details contain claims, verdicts, evidence IDs, model, and
prompt version. There is no per-verdict prose rationale; evidence IDs support
manual review, but cannot prove the judge's semantic decision. Reuse of the
generation model as judge creates self-evaluation bias.

The mocked fixtures in `tests/test_judges.py` verify all/half/none directions
for Faithfulness and Claim Recall (`1`, `0.5`, `0`), exact/extra/missing facts
for Factual Correctness (`1`, `2/3`, `2/3`), and Context Precision `1` when the
only useful chunk is first versus `1/3` when it is third. They also check
zero response claims, malformed output, missing fields, out-of-range relevance,
uncertain chunk labels, citation verdict omission, and non-persistence of an
extra `reasoning` field. These fixtures test arithmetic and error semantics;
they cannot validate a real judge's claim extraction accuracy.

| Metric | Source and exact rule | Required inputs; undefined/empty behavior | Range; type | Reference and classification |
| --- | --- | --- | --- | --- |
| Faithfulness / Groundedness | `judges.judge_standard`: context-supported response claims / all response claims | Response, retrieved context texts, judge. Missing response/context or zero response claims: skip. Empty supplied context with claims can score 0. | `[0,1]`; LLM-judged | [Ragas Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/) uses the same ratio; claim extraction and entailment prompts differ. VERIFIED WITH DOCUMENTED VARIANT; directional fixtures pass. |
| Answer Relevance | `judges.judge_standard`: one direct judge rubric value for whether the answer addresses the question, without factual correctness | Question, nonempty response, judge. Missing response or malformed/out-of-range value: skip. | `[0,1]`; LLM-judged | [Ragas Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/) instead averages embedding similarities of generated questions. Our direct rubric is **not equivalent**. VERIFIED WITH DOCUMENTED VARIANT; bounds and live result checked. |
| Factual Correctness | `judges.judge_standard`: response-claim precision against reference and reference-claim recall against response, combined as `2PR/(P+R)` or 0 when `P+R=0` | Response, reference answer, judge. Missing response or zero reference claims: skip; zero response claims with reference claims: 0. | `[0,1]`; LLM-judged | [Ragas Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/) and [RAGChecker](https://arxiv.org/abs/2408.08067) use claim-level overlap/F1 concepts, but extraction/checking differ. VERIFIED WITH DOCUMENTED VARIANT; directional fixtures pass. |
| Context Precision | `judges.judge_standard`: average of precision at each chunk judged useful to the reference answer, divided by number of useful chunks; 0 if no chunk is useful | Ranked context texts, reference claims, judge. Missing/empty contexts, zero reference claims, or any uncertain chunk relevance: skip. | `[0,1]`; LLM-judged | [Ragas Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) uses the same ranking formula; chunk usefulness rubric differs. VERIFIED WITH DOCUMENTED VARIANT; first-versus-last fixture passes. |
| Context Recall / Claim Recall | `judges.judge_standard` key `claim_recall`: reference claims supported by retrieved contexts / all reference claims | Reference answer, supplied retrieved context texts, judge. Missing contexts or zero reference claims: skip; empty supplied context can score 0. | `[0,1]`; LLM-judged | [Ragas Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) and [RAGChecker claim recall](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md) use this evidence-coverage concept; judge protocol differs. VERIFIED WITH DOCUMENTED VARIANT; directional fixtures pass. |
| Noise Sensitivity | `judges.judge_deep`: response claims unsupported by reference but supported by at least one chunk judged irrelevant / all response claims | Standard checks, nonempty response claims and contexts, definitive correctness/chunk labels, at least one irrelevant chunk. Otherwise skip. | `[0,1]`, lower better; LLM-judged | Related to [RAGChecker irrelevant-noise sensitivity](https://arxiv.org/abs/2408.08067) and [Ragas Noise Sensitivity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/), but our single attribution rule and uncertainty policy are distinct. EXPERIMENTAL; mock attribution fixture passes, no human calibration. |
| Citation Support | `judges.judge_deep`: cited response claims judged supported by their cited retrieved text / response claims matched to at least one citation | Structured citations, extracted response claims, retrieved IDs/text, judge. No citations, claims, evidence, or matched cited claims: skip; malformed verdict: skip. | `[0,1]`; LLM-judged | Claim-to-evidence goal resembles [ALCE](https://github.com/princeton-nlp/ALCE), but matching and entailment protocol are custom. EXPERIMENTAL; mock support/failure fixtures pass. |
| Citation Coverage | `judges.judge_deep`: response claims with a supporting citation / all response claims | Same inputs; missing citations/claims/evidence or malformed verdict: skip. No matched citation with valid inputs: 0. | `[0,1]`; LLM-judged | ALCE-inspired diagnostic, not ALCE's metric. EXPERIMENTAL; mock support/failure fixtures pass. |
| Citation Unresolved IDs | `judges.judge_deep`: cardinality of cited IDs absent from retrieved IDs | Structured citations and retrieved IDs/text plus extracted claims under the Deep path; if prerequisites are absent, skip explicitly. | Nonnegative integer-valued count; deterministic after judge extraction | Direct set-difference count. VERIFIED; known and missing-input fixtures pass. |

A missing `citation_checks[].supported` field now invalidates the citation
judge response instead of silently counting it as unsupported. Citation Support
and Coverage remain visible but explicitly experimental. Their zero means a
valid assessment with no credited support; a failed or unavailable assessment
is a skip.

## Operational metrics and summaries

Live measurements come from `pipeline.run_existing_chroma`; imported values
come from validated dataset fields via `evaluation.precomputed_observation`.
`evaluation.evaluate_quick` and `judges._operational` build per-case values.
Missing source fields skip without imputation. Imported numbers must be finite
and nonnegative. Provider token counts are accepted only as nonnegative
integers by `providers._usage`. Provider costs use `providers._cost` and may
be estimates; they are not bills. All operational values are deterministic
calculations over clocks or metadata, not LLM verdicts, although provider calls
supply the metadata. The measurement/reference contracts are Python's
[`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)
for elapsed durations and LiteLLM's
[usage/cost interface](https://docs.litellm.ai/docs/) for provider metadata;
neither establishes billed cost for every provider.

| Metric | Source and exact rule | Required inputs / empty behavior | Range | Classification and verification |
| --- | --- | --- | --- | --- |
| Retrieval latency (ms) | `pipeline.run_existing_chroma`: elapsed wall clock around query embedding plus Chroma search | Live run or imported `retrieval_latency_ms`; otherwise skip | `[0,∞)` | VERIFIED; live timing populated, import/skip path tested |
| Generation latency (ms) | `pipeline.run_existing_chroma`: elapsed wall clock around answer-model call | Live run or imported `generation_latency_ms`; otherwise skip | `[0,∞)` | VERIFIED; live timing populated, import/skip path tested |
| Total latency (ms) | `pipeline.run_existing_chroma`: elapsed embedding + search + answer call, excluding judge | Live run or imported `total_latency_ms`; otherwise skip; imported total is not recomputed | `[0,∞)` | VERIFIED WITH DOCUMENTED VARIANT; stage meaning explicit, live timing populated |
| Judge latency (ms) | `judges._operational`: elapsed judge call sequence, including invalid responses; Deep adds Standard and citation-call times when available | Judge call sequence; absent judge stage: skip | `[0,∞)` | VERIFIED; live value and mocked retry path checked |
| Embedding tokens | `evaluation.evaluate_quick`: provider `total_tokens` for query embedding or imported value | Provider metadata or imported `embedding_tokens`; otherwise skip | Nonnegative integer-valued | VERIFIED; live value and import/skip path checked |
| Generation prompt tokens | `evaluation.evaluate_quick`: provider `prompt_tokens` or imported value | Provider metadata or imported `generation_prompt_tokens`; otherwise skip | Nonnegative integer-valued | VERIFIED; live value and import/skip path checked |
| Generation completion tokens | `evaluation.evaluate_quick`: provider `completion_tokens` or imported value | Provider metadata or imported `generation_completion_tokens`; otherwise skip | Nonnegative integer-valued | VERIFIED; live value and import/skip path checked |
| Judge prompt tokens | `judges._operational`: sum of prompt tokens across judge calls, including invalid retries | Complete metadata for all judge calls; otherwise skip | Nonnegative integer-valued | VERIFIED; mocked retry and live paths checked |
| Judge completion tokens | `judges._operational`: sum of completion tokens across judge calls | Complete metadata for all judge calls; otherwise skip | Nonnegative integer-valued | VERIFIED; mocked and live paths checked |
| Total tokens | `evaluation.evaluate_quick` and `_include_judge_totals`: sum of embedding, generation prompt/completion, and, for Standard/Deep, judge prompt/completion tokens | Every included component must be known; otherwise skip | Nonnegative integer-valued | VERIFIED; additive rule and missing-component behavior inspected |
| Retrieval cost (USD) | `evaluation.evaluate_quick`: embedding-call price estimate or imported retrieval cost | Provider price/usage metadata or imported value; otherwise skip | `[0,∞)` | EXPERIMENTAL as a monetary estimate; arithmetic/skip rule inspected |
| Generation cost (USD) | `evaluation.evaluate_quick`: generation-call price estimate or imported generation cost | Provider price/usage metadata or imported value; otherwise skip | `[0,∞)` | EXPERIMENTAL as a monetary estimate; arithmetic/skip rule inspected |
| Judge cost (USD) | `judges._operational`: sum of price estimates for every judge call | Price metadata for all judge calls; otherwise skip | `[0,∞)` | EXPERIMENTAL as a monetary estimate; mocked retry sum checked |
| Total cost (USD) | `evaluation.evaluate_quick` and `_include_judge_totals`: sum of all applicable stage costs | Every included stage cost must be known; otherwise skip | `[0,∞)` | EXPERIMENTAL as a monetary estimate; additive rule inspected |

`statistics.summarize` reports the arithmetic mean of **eligible** values,
`eligible` count, and total count; skipped/failed observations are never zero.
`statistics.summarize_by_category` applies the same rule per category. Missing
metric keys in a case are now counted as ineligible, so totals match case
counts. With at least five eligible cases, a fixed-seed (`0`) percentile
bootstrap resamples eligible cases with replacement 2,000 times and reports
order statistics at indices `int(0.025*(B-1))` and `int(0.975*(B-1))` as a
nominal 95% interval for the mean. Fewer than five eligible cases show no
interval. This small-sample gate prevents a two-case interval from looking
precise; even at five cases, the interval describes case sampling only, not
judge validity or systematic bias. Known fixtures verify mean `0.5` for
`[0,1,skip]`, mean/CI `2` for five constant values, category counts, and
seed reproducibility. See [SciPy's bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
for the reference resampling method; this project implements a simple
percentile interval rather than SciPy's BCa default. **Mean and category mean:
VERIFIED. Interval: VERIFIED WITH DOCUMENTED VARIANT** (fixed-seed percentile
method and `n>=5` display rule).

## Release audit

| Check | Result |
| --- | --- |
| New-user README, uv install, Quick quickstart, both modes, JSONL schema, presets | Pass after clarification; `examples/golden.jsonl` is valid and Quick needs no provider call. |
| Judge definitions, limitations, variants, privacy | Pass; README links here, explains direct Answer Relevance, experimental Deep diagnostics, provider text transfer, and in-memory keys. |
| Local Ollama example | Added provider/model/base-URL example; a matching Chroma embedding space is required. |
| Ed Donner Week 5 inspiration | Acknowledged as inspiration with explicit independence and no affiliation. `docs/COURSE_REUSE_AUDIT.md` records the read-only comparison; no course source file is imported or copied. |
| LICENSE, package metadata | MIT license exists; `rag-eval-studio` version `0.1.0`, Python `>=3.12`, CLI entry point, and core/dev dependencies are sensible. |
| Tracked artifacts and secrets | `git ls-files` has no local Chroma DB, `/tmp` artifact, `.env`, Codex auth/config, or copied course source. A filename-only scan of tracked text found no obvious API-key/private-key patterns; this cannot prove absence of every secret. |
| CI | Added minimal GitHub Actions Python 3.12 + uv locked sync + Ruff + pytest workflow. It does not publish. |
| Publication | Deliberately not performed. Public release remains a separate user-authorized action. |
