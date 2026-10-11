"""Load and validate golden cases without guessing missing supervision."""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rag_eval_studio.models import Citation, GoldenCase


class DatasetError(ValueError):
    """A dataset row does not satisfy the canonical schema."""


_FIELDS = set(GoldenCase.__dataclass_fields__)
_LIST_FIELDS = {
    "relevant_doc_ids",
    "reference_contexts",
    "retrieved_contexts",
    "retrieved_doc_ids",
}
_LATENCY_FIELDS = {"retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"}
_COST_FIELDS = {"retrieval_cost_usd", "generation_cost_usd"}
_TOKEN_FIELDS = {
    "embedding_tokens",
    "generation_prompt_tokens",
    "generation_completion_tokens",
}


def _text(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise DatasetError(f"{field} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{field} must be a nonblank string")
    return value.strip()


def _items(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DatasetError(f"{field} must be an array of nonblank strings")
    items = tuple(item.strip() for item in value)
    if field.endswith("_doc_ids") and len(set(items)) != len(items):
        raise DatasetError(f"{field} contains duplicate IDs")
    return items


def _number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{field} must be a nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DatasetError(f"{field} must be a nonnegative finite number")
    return result


def _tokens(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"{field} must be a nonnegative integer")
    return value


def _citations(value: Any) -> tuple[Citation, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DatasetError("citations must be an array")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"claim", "doc_ids"}:
            raise DatasetError("each citation needs claim and doc_ids")
        claim = _text(item["claim"], "citation claim", required=True)
        doc_ids = _items(item["doc_ids"], "citation doc_ids")
        if not doc_ids:
            raise DatasetError("citation doc_ids must be nonempty")
        assert claim is not None
        result.append(Citation(claim, doc_ids))
    return tuple(result)


def parse_case(data: dict[str, Any]) -> GoldenCase:
    """Validate one canonical JSON object and generate a stable ID if needed."""
    unknown = set(data) - _FIELDS
    if unknown:
        raise DatasetError(f"unknown fields: {', '.join(sorted(unknown))}")

    question = _text(data.get("question"), "question", required=True)
    reference = _text(data.get("reference_answer"), "reference_answer", required=True)
    assert question is not None and reference is not None
    case_id = _text(data.get("id"), "id")
    if case_id is None:
        digest = hashlib.sha256(
            json.dumps([question, reference], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        case_id = f"case-{digest}"

    lists = {field: _items(data.get(field), field) for field in _LIST_FIELDS}
    numbers = {
        field: _number(data.get(field), field)
        for field in _LATENCY_FIELDS | _COST_FIELDS
    }
    tokens = {field: _tokens(data.get(field), field) for field in _TOKEN_FIELDS}
    contexts = lists["retrieved_contexts"]
    ids = lists["retrieved_doc_ids"]
    if contexts is not None and ids is not None and len(contexts) != len(ids):
        raise DatasetError("retrieved_contexts and retrieved_doc_ids must align")

    response = data.get("response")
    if isinstance(response, str) and not response.strip():
        response = None

    return GoldenCase(
        id=case_id,
        question=question,
        reference_answer=reference,
        category=_text(data.get("category"), "category"),
        response=_text(response, "response"),
        citations=_citations(data.get("citations")),
        **lists,
        **numbers,
        **tokens,
    )


def _csv_row(row: dict[str, str | None]) -> dict[str, Any]:
    if None in row:
        raise DatasetError("CSV row has more values than headers")
    data: dict[str, Any] = {key: value for key, value in row.items() if value != ""}
    for field in _LIST_FIELDS & data.keys():
        try:
            data[field] = json.loads(data[field])
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{field} must contain a JSON array") from exc
    if "citations" in data:
        try:
            data["citations"] = json.loads(data["citations"])
        except json.JSONDecodeError as exc:
            raise DatasetError("citations must contain a JSON array") from exc
    for field in (_LATENCY_FIELDS | _COST_FIELDS | _TOKEN_FIELDS) & data.keys():
        try:
            data[field] = json.loads(data[field])
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{field} must contain a JSON number") from exc
    return data


def load_cases(path: str | Path) -> list[GoldenCase]:
    """Read JSONL or CSV; errors identify the source row."""
    path = Path(path)
    if path.suffix.lower() not in {".jsonl", ".csv"}:
        raise DatasetError("dataset must be a .jsonl or .csv file")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        if path.suffix.lower() == ".jsonl":
            rows = ((number, line) for number, line in enumerate(source, 1))
        else:
            rows = enumerate(csv.DictReader(source), 2)

        for number, raw in rows:
            if isinstance(raw, str):
                if not raw.strip():
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"line {number}: invalid JSON: {exc.msg}"
                    ) from exc
            else:
                try:
                    data = _csv_row(raw)
                except DatasetError as exc:
                    raise DatasetError(f"line {number}: {exc}") from exc
            if not isinstance(data, dict):
                raise DatasetError(f"line {number}: expected a JSON object")
            try:
                case = parse_case(data)
            except DatasetError as exc:
                raise DatasetError(f"line {number}: {exc}") from exc
            if case.id in seen:
                raise DatasetError(f"line {number}: duplicate case ID {case.id}")
            seen.add(case.id)
            cases.append(case)

    if not cases:
        raise DatasetError("dataset contains no cases")
    return cases
