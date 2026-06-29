from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("sentence_transformers", reason="full ML dependency stack not installed")

from src.evaluation.fairness_eval import (  # noqa: E402
    _pairwise_keys,
    cosine_similarity,
    jaccard_similarity,
    load_golden_fairness_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FAIRNESS_PAIRS = PROJECT_ROOT / "tests" / "golden_fairness_pairs.json"




def test_jaccard_similarity_identical_sets():
    assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0


def test_jaccard_similarity_disjoint_sets():
    assert jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_similarity_partial_overlap():
    # intersection={b}, union={a,b,c} -> 1/3
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_jaccard_similarity_both_empty_is_one():
    assert jaccard_similarity([], []) == 1.0




def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0




def test_pairwise_keys_two_labels():
    assert _pairwise_keys(["a", "b"]) == [("a", "b")]


def test_pairwise_keys_three_labels():
    pairs = _pairwise_keys(["a", "b", "c"])
    assert pairs == [("a", "b"), ("a", "c"), ("b", "c")]


def test_pairwise_keys_single_label_has_no_pairs():
    assert _pairwise_keys(["a"]) == []




def test_load_golden_fairness_pairs_returns_dataclasses():
    pairs = load_golden_fairness_pairs(str(GOLDEN_FAIRNESS_PAIRS))
    assert len(pairs) > 0
    pair = pairs[0]
    assert pair.pair_id
    assert pair.dimension
    assert len(pair.variants) >= 2
    for variant in pair.variants:
        assert variant.label
        assert variant.query
