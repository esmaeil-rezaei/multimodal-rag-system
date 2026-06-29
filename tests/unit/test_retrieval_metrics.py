from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers", reason="full ML dependency stack not installed")

from src.evaluation.retrieval_eval import (  # noqa: E402
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = ["b", "d", "z"]


def test_precision_at_k_basic():
    # top-3 = [a, b, c] -> 1 of 3 relevant
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)


def test_precision_at_k_zero_k_returns_zero():
    assert precision_at_k(RETRIEVED, RELEVANT, 0) == 0.0


def test_precision_at_k_no_relevant_items():
    assert precision_at_k(RETRIEVED, [], 3) == 0.0


def test_precision_at_k_all_relevant():
    assert precision_at_k(["x", "y"], ["x", "y"], 2) == 1.0


def test_recall_at_k_basic():
    # top-5 = all of RETRIEVED -> hits b, d (2 of 3 relevant)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant_returns_zero():
    assert recall_at_k(RETRIEVED, [], 5) == 0.0


def test_recall_at_k_k_smaller_than_relevant_positions():
    # top-1 = [a] -> none of the relevant items found
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.0


def test_mrr_first_result_relevant():
    assert mean_reciprocal_rank(["b", "a"], ["b"]) == 1.0


def test_mrr_second_result_relevant():
    assert mean_reciprocal_rank(["a", "b"], ["b"]) == pytest.approx(0.5)


def test_mrr_no_relevant_results_returns_zero():
    assert mean_reciprocal_rank(["a", "c"], ["b"]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    # All relevant items at the top -> NDCG == 1.0
    retrieved = ["x", "y", "z"]
    relevant = ["x", "y"]
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_items_in_results_is_zero():
    assert ndcg_at_k(["a", "b", "c"], ["z"], 3) == 0.0


def test_ndcg_at_k_empty_relevant_set_is_zero():
    assert ndcg_at_k(["a", "b"], [], 2) == 0.0


def test_ndcg_at_k_order_matters():
    relevant = ["x"]
    score_first = ndcg_at_k(["x", "a", "b"], relevant, 3)
    score_last = ndcg_at_k(["a", "b", "x"], relevant, 3)
    assert score_first > score_last
