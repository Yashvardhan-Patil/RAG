# Course Reuse and Provenance Audit

## Objective

This document audits the provenance and origin of `rag-eval-studio` to verify that no proprietary course materials, proprietary intellectual property, or copyrighted files from external tutorials or courses (such as Ed Donner's *LLM Engineering* course or Apna College tutorials) are bundled or redistributed.

## Provenance Analysis

1. **Independent Implementation**:
   - `rag-eval-studio` was inspired at a conceptual level by modern RAG evaluation workflows discussed in industry and education (specifically distinguishing retrieval metrics from generation quality).
   - All modules—including [models.py](file:///g:/rag-eval-studio/src/rag_eval_studio/models.py), [dataset.py](file:///g:/rag-eval-studio/src/rag_eval_studio/dataset.py), [retrieval.py](file:///g:/rag-eval-studio/src/rag_eval_studio/retrieval.py), [judges.py](file:///g:/rag-eval-studio/src/rag_eval_studio/judges.py), [statistics.py](file:///g:/rag-eval-studio/src/rag_eval_studio/statistics.py), [runner.py](file:///g:/rag-eval-studio/src/rag_eval_studio/runner.py), [report.py](file:///g:/rag-eval-studio/src/rag_eval_studio/report.py), and [app.py](file:///g:/rag-eval-studio/src/rag_eval_studio/app.py)—are independent implementations written from scratch with custom dataclasses, pure typing, and verified algorithms.

2. **No Proprietary Code or Data**:
   - Zero course files, solution notebooks, slides, or proprietary datasets are included or imported.
   - The sample dataset in `examples/golden.jsonl` contains standard general knowledge question-answer pairs for verification.

3. **Metrics Formulation**:
   - Classical Information Retrieval metrics (Hit Rate@K, Precision@K, Recall@K, MRR@K, nDCG@K) follow standard academic definitions from textbooks (e.g., Manning, Raghavan, Schütze, *Introduction to Information Retrieval*).
   - LLM-as-a-judge formulations (Faithfulness, Answer Relevance, Factual Correctness, Context Precision, Claim Recall) use custom structured JSON schema validation via LiteLLM without relying on external proprietary frameworks.
   - Bootstrapped confidence intervals use standard non-parametric percentile resampling.

4. **Third-Party & Tutorial Branding Scan**:
   - Repository-wide case-insensitive searches for "Apna College", "ApnaCollege", and related tutorial channel brands return 0 occurrences.
   - The project maintains an independent MIT license and neutral, professional branding.
