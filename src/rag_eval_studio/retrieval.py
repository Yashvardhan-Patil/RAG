"""Classical retrieval scores from complete, explicit relevance labels."""

import math

from rag_eval_studio.models import GoldenCase, MetricValue

METRICS = ("hit_rate", "precision", "recall", "mrr", "ndcg")


def score_retrieval(case: GoldenCase, k: int) -> dict[str, MetricValue]:
    """Score a ranked result list at K, or explain why every score is skipped."""
    if k < 1:
        raise ValueError("k must be positive")

    if case.relevant_doc_ids is None:
        reason = "relevant_doc_ids are unavailable"
    elif not case.relevant_doc_ids:
        reason = "relevant_doc_ids is empty"
    elif case.retrieved_doc_ids is None:
        reason = "retrieved_doc_ids are unavailable"
    elif len(set(case.retrieved_doc_ids)) != len(case.retrieved_doc_ids):
        reason = "retrieved_doc_ids contains duplicate IDs"
    else:
        reason = None
    if reason:
        return {name: MetricValue(skip_reason=reason) for name in METRICS}

    assert case.relevant_doc_ids is not None
    assert case.retrieved_doc_ids is not None
    relevant = set(case.relevant_doc_ids)
    ranked = case.retrieved_doc_ids[:k]
    hits = [int(doc_id in relevant) for doc_id in ranked]
    count = sum(hits)
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))

    return {
        "hit_rate": MetricValue(value=float(count > 0)),
        "precision": MetricValue(value=count / k),
        "recall": MetricValue(value=count / len(relevant)),
        "mrr": MetricValue(value=1 / first if first else 0.0),
        "ndcg": MetricValue(value=dcg / ideal),
    }
