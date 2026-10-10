import csv
import json

import pytest

from rag_eval_studio.dataset import DatasetError, load_cases, parse_case


def test_generated_id_is_stable_and_optional_lists_remain_missing():
    data = {"question": "What happened?", "reference_answer": "An event."}
    first = parse_case(data)
    assert first.id == parse_case(data).id
    assert first.relevant_doc_ids is None
    assert first.retrieved_doc_ids is None
    assert parse_case({**data, "response": " "}).response is None


def test_jsonl_loads_precomputed_record(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Q",
                "reference_answer": "A",
                "response": "B",
                "relevant_doc_ids": ["doc-1"],
                "retrieved_doc_ids": ["doc-2", "doc-1"],
                "retrieved_contexts": ["irrelevant", "evidence"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].retrieved_doc_ids == ("doc-2", "doc-1")


def test_csv_requires_json_arrays(tmp_path):
    path = tmp_path / "golden.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["question", "reference_answer", "relevant_doc_ids"]
        )
        writer.writeheader()
        writer.writerow(
            {"question": "Q", "reference_answer": "A", "relevant_doc_ids": '["1"]'}
        )
    assert load_cases(path)[0].relevant_doc_ids == ("1",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"question": " "}, "question must be a nonblank string"),
        ({"relevant_doc_ids": ["x", "x"]}, "duplicate IDs"),
        (
            {"retrieved_doc_ids": ["a"], "retrieved_contexts": ["a", "b"]},
            "must align",
        ),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_invalid_records_fail(change, message):
    with pytest.raises(DatasetError, match=message):
        parse_case({"question": "Q", "reference_answer": "A", **change})


def test_duplicate_generated_ids_are_rejected(tmp_path):
    path = tmp_path / "golden.jsonl"
    row = json.dumps({"question": "Q", "reference_answer": "A"})
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="line 2: duplicate case ID"):
        load_cases(path)


def test_imported_operational_values_and_citations_are_validated():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "total_latency_ms": 5.5,
            "embedding_tokens": 3,
            "citations": [{"claim": "A", "doc_ids": ["doc"]}],
        }
    )
    assert case.total_latency_ms == 5.5
    assert case.citations[0].doc_ids == ("doc",)
    with pytest.raises(DatasetError, match="nonnegative"):
        parse_case({"question": "Q", "reference_answer": "A", "total_latency_ms": -1})
    with pytest.raises(DatasetError, match="nonnegative integer"):
        parse_case({"question": "Q", "reference_answer": "A", "embedding_tokens": 1.5})


def test_csv_extra_values_report_line_number(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("question,reference_answer\nQ,A,extra\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="line 2: CSV row has more values"):
        load_cases(path)
