from __future__ import annotations

import pytest
from src.rlhf.reward_functions import (
    citation_score,
    composite_reward,
    faithfulness_score,
    length_penalty,
)



def test_citation_score_present():
    assert citation_score("The sky is blue [1].") == 1.0


def test_citation_score_absent():
    assert citation_score("The sky is blue.") == 0.0




def test_faithfulness_score_no_context_returns_one():
    assert faithfulness_score("anything goes here", None) == 1.0


def test_faithfulness_score_empty_response_returns_zero():
    assert faithfulness_score("   ", "some context") == 0.0


def test_faithfulness_score_full_overlap():
    context = "the cat sat on the mat"
    response = "the cat sat"
    assert faithfulness_score(response, context) == 1.0


def test_faithfulness_score_partial_overlap():
    context = "the cat sat on the mat"
    response = "the dog sat"
    # "the" and "sat" overlap, "dog" does not -> 2/3
    assert faithfulness_score(response, context) == pytest.approx(2 / 3)




def test_length_penalty_empty_response_is_zero():
    assert length_penalty("", target_length_tokens=10, max_length_tokens=20) == 0.0


def test_length_penalty_at_target_is_one():
    response = " ".join(["word"] * 10)
    assert length_penalty(response, target_length_tokens=10, max_length_tokens=20) == 1.0


def test_length_penalty_below_target_scales_linearly():
    response = " ".join(["word"] * 5)
    assert length_penalty(response, target_length_tokens=10, max_length_tokens=20) == pytest.approx(
        0.5
    )


def test_length_penalty_at_or_above_max_is_zero():
    response = " ".join(["word"] * 20)
    assert length_penalty(response, target_length_tokens=10, max_length_tokens=20) == 0.0


def test_length_penalty_between_target_and_max_decays():
    response = " ".join(["word"] * 15)
    # Halfway between target (10) and max (20) -> 0.5
    assert length_penalty(response, target_length_tokens=10, max_length_tokens=20) == pytest.approx(
        0.5
    )




_CFG = {
    "weights": {
        "learned_reward": 0.7,
        "citation_score": 0.15,
        "faithfulness": 0.1,
        "length_penalty": 0.05,
    },
    "citation_pattern": r"\[\d+\]",
    "target_length_tokens": 4,
    "max_length_tokens": 8,
}


def test_composite_reward_without_reward_model_redistributes_weight():
    breakdown = composite_reward(
        prompt="What is X?",
        response="X is the answer [1].",
        context="X is the answer.",
        cfg=dict(_CFG),
    )

    assert breakdown.learned_reward == 0.0
    # learned_reward weight (0.7) is redistributed; total weight excluding it stays ~1.0
    assert breakdown.weights["learned_reward"] == 0.0
    assert sum(breakdown.weights.values()) == pytest.approx(1.0)
    assert breakdown.citation_score == 1.0
    assert 0.0 <= breakdown.total <= 1.0


def test_composite_reward_total_is_weighted_sum():
    breakdown = composite_reward(
        prompt="What is X?",
        response="no citation here",
        context=None,
        cfg=dict(_CFG),
    )

    expected = (
        breakdown.weights["citation_score"] * breakdown.citation_score
        + breakdown.weights["faithfulness"] * breakdown.faithfulness_score
        + breakdown.weights["length_penalty"] * breakdown.length_penalty
    )
    assert breakdown.total == pytest.approx(expected)
