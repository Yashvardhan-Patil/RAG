import math

import pytest

from rag_eval_studio.dataset import parse_case
from rag_eval_studio.models import GoldenCase
from rag_eval_studio.retrieval import score_retrieval


def test_scores_use_explicit_labels_and_full_ideal_ranking():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "relevant_doc_ids": ["a", "b"],
            "retrieved_doc_ids": ["noise", "a", "other"],
        }
    )
    scores = score_retrieval(case, 3)
    assert scores["hit_rate"].value == 1
    assert scores["precision"].value == pytest.approx(1 / 3)
    assert scores["recall"].value == 0.5
    assert scores["mrr"].value == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert scores["ndcg"].value == pytest.approx((1 / math.log2(3)) / ideal)


def test_valid_empty_retrieval_is_zero_not_skipped():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "relevant_doc_ids": ["a"],
            "retrieved_doc_ids": [],
        }
    )
    assert all(value.value == 0 for value in score_retrieval(case, 5).values())


@pytest.mark.parametrize("labels", [None, []])
def test_missing_or_empty_labels_skip_all_metrics(labels):
    data = {"question": "Q", "reference_answer": "A", "retrieved_doc_ids": ["a"]}
    if labels is not None:
        data["relevant_doc_ids"] = labels
    assert all(
        value.value is None and value.skip_reason
        for value in score_retrieval(parse_case(data), 2).values()
    )


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        score_retrieval(parse_case({"question": "Q", "reference_answer": "A"}), 0)


def test_hand_calculated_multiple_relevant_fixture():
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "relevant_doc_ids": ["A", "C", "D"],
            "retrieved_doc_ids": ["A", "B", "C"],
        }
    )
    scores = score_retrieval(case, 3)
    assert scores["hit_rate"].value == 1
    assert scores["precision"].value == pytest.approx(2 / 3)
    assert scores["recall"].value == pytest.approx(2 / 3)
    assert scores["mrr"].value == 1
    # Binary gains at ranks 1 and 3; ideal has gains at ranks 1, 2, and 3.
    assert scores["ndcg"].value == pytest.approx(1.5 / (1 + 1 / math.log2(3) + 0.5))


@pytest.mark.parametrize(
    ("relevant", "retrieved", "k", "expected"),
    [
        (["D"], ["A", "B", "D"], 3, (1, 1 / 3, 1, 1 / 3, 0.5)),
        (["D"], ["A", "B", "C"], 3, (0, 0, 0, 0, 0)),
        (["A", "B", "C"], ["A", "B", "C"], 3, (1, 1, 1, 1, 1)),
        (["A", "C", "D"], ["B", "A", "C"], 1, (0, 0, 0, 0, 0)),
        (
            ["A", "C", "D"],
            ["A"],
            3,
            (1, 1 / 3, 1 / 3, 1, 1 / (1 + 1 / math.log2(3) + 0.5)),
        ),
    ],
)
def test_hand_calculated_retrieval_edges(relevant, retrieved, k, expected):
    case = parse_case(
        {
            "question": "Q",
            "reference_answer": "A",
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved,
        }
    )
    scores = score_retrieval(case, k)
    assert [
        scores[name].value
        for name in ("hit_rate", "precision", "recall", "mrr", "ndcg")
    ] == pytest.approx(expected)


def test_duplicate_ranked_ids_are_ineligible_even_when_case_bypasses_loader():
    case = GoldenCase(
        "q", "Q", "A", relevant_doc_ids=("A",), retrieved_doc_ids=("A", "A")
    )
    scores = score_retrieval(case, 2)
    assert all(score.value is None for score in scores.values())
    assert all(
        score.skip_reason == "retrieved_doc_ids contains duplicate IDs"
        for score in scores.values()
    )
