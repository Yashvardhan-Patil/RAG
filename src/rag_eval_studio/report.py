"""Summaries and credential-free result serialization."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_studio.models import CaseResult, GoldenCase, MetricValue
from rag_eval_studio.runner import RunSettings
from rag_eval_studio.statistics import summarize, summarize_by_category

_RETRIEVAL_LABELS = {
    "hit_rate": "Hit Rate@K",
    "precision": "Precision@K",
    "recall": "Recall@K",
    "mrr": "MRR@K",
    "ndcg": "nDCG@K",
}


def _metric_label(name: str) -> str:
    if name in _RETRIEVAL_LABELS:
        return _RETRIEVAL_LABELS[name]
    if name.endswith("_ms"):
        return f"{name[:-3].replace('_', ' ').title()} (ms)"
    if name.endswith("_cost_usd"):
        return f"{name[:-9].replace('_', ' ').title()} Cost (USD)"
    return name.replace("_", " ").title().replace(" Ids", " IDs")


def summary_rows(results: list[CaseResult]) -> list[list[str]]:
    rows = []
    for name in dict.fromkeys(name for result in results for name in result.metrics):
        values = [
            result.metrics.get(name, MetricValue(skip_reason="metric was not recorded"))
            for result in results
        ]
        summary = summarize(values)
        interval = (
            f"{_format(name, summary.ci_low)}–{_format(name, summary.ci_high)}"
            if summary.ci_low is not None
            else "—"
        )
        rows.append(
            [
                _dimension(name),
                _metric_label(name),
                _format(name, summary.mean) if summary.mean is not None else "—",
                interval,
                f"{summary.eligible}/{summary.total}",
            ]
        )
    return rows


def category_rows(results: list[CaseResult]) -> list[list[str]]:
    groups = summarize_by_category(
        [(result.category, result.metrics) for result in results]
    )
    return [
        [
            category,
            _metric_label(name),
            _format(name, summary.mean) if summary.mean is not None else "—",
            (
                f"{_format(name, summary.ci_low)}–{_format(name, summary.ci_high)}"
                if summary.ci_low is not None
                else "—"
            ),
            f"{summary.eligible}/{summary.total}",
        ]
        for category, metrics in groups.items()
        for name, summary in metrics.items()
    ]


def diagnostic_rows(results: list[CaseResult]) -> list[list[str]]:
    return [
        [
            result.case_id,
            result.category or "Uncategorized",
            result.observation.response or "—",
            str(sum(value.value is None for value in result.metrics.values())),
            (result.judge_details or {}).get("error", "OK"),
        ]
        for result in results
    ]


def skip_summary(results: list[CaseResult]) -> str:
    reasons = Counter(
        (name, value.skip_reason)
        for result in results
        for name, value in result.metrics.items()
        if value.skip_reason
    )
    if not reasons:
        return "All requested metrics were scored."
    lines = ["| Metric | Reason | Cases |", "| --- | --- | ---: |"]
    lines.extend(
        f"| {_metric_label(name)} | {reason} | {count} |"
        for (name, reason), count in sorted(reasons.items())
    )
    return "\n".join(lines)


def _dimension(name: str) -> str:
    if name in {
        "hit_rate",
        "precision",
        "recall",
        "mrr",
        "ndcg",
        "context_precision",
        "claim_recall",
    }:
        return "Retrieval"
    if name in {"faithfulness", "answer_relevance", "factual_correctness"}:
        return "Generation"
    if name in {
        "noise_sensitivity",
        "citation_support",
        "citation_coverage",
        "citation_unresolved_ids",
    }:
        return "Robustness / citations"
    return "Operations"


def _format(name: str, value: float) -> str:
    if name.endswith("_cost_usd"):
        return f"{value:.6f}"
    if name.endswith("_ms") or name.endswith("_tokens"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def export_report(
    cases: list[GoldenCase],
    results: list[CaseResult],
    settings: RunSettings,
    dataset_path: str | Path,
) -> dict:
    """Build an export with no API keys or raw provider exceptions."""
    fingerprint = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": fingerprint,
        "mode": settings.mode,
        "preset": settings.preset,
        "top_k": settings.top_k,
        "judge_model": settings.judge.model_id if settings.judge else None,
        "judge_reuses_generation": bool(
            settings.judge
            and settings.generation
            and settings.judge == settings.generation
        ),
        "cases": [
            {"input": asdict(case), "result": asdict(result)}
            for case, result in zip(cases, results, strict=True)
        ],
    }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
