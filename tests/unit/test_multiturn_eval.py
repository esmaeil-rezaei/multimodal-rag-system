from __future__ import annotations

from pathlib import Path
import pytest

pytest.importorskip("sentence_transformers", reason="full ML dependency stack not installed")

from src.evaluation.multiturn_eval import (  # noqa: E402
    _check_condensation,
    _check_keywords,
    _word_present,
    load_golden_conversations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CONVERSATIONS = PROJECT_ROOT / "tests" / "golden_conversations.json"




def test_word_present_basic_match():
    assert _word_present("The phospho-tau-181 level was elevated.", "phospho-tau-181") is True


def test_word_present_is_case_insensitive():
    assert _word_present("the phospho-tau-181 level was elevated.", "Phospho-Tau-181") is True


def test_word_present_requires_whole_word_boundary():
    # "it" should not match inside "digital"
    assert _word_present("the digital assessment", "it") is False


def test_word_present_phrase_with_multiple_words():
    assert _word_present("area under the curve was computed", "area under the curve") is True


def test_word_present_not_found():
    assert _word_present("completely unrelated text", "phospho-tau-181") is False




def test_check_condensation_passes_when_required_terms_present_and_pronouns_absent():
    result = _check_condensation(
        "Was the Ab(1-42) level reduced in CSF?",
        must_contain=["Ab(1-42)", "CSF"],
        must_not_contain=["its"],
    )
    assert result is True


def test_check_condensation_fails_when_required_term_missing():
    result = _check_condensation(
        "Was its level reduced?",
        must_contain=["Ab(1-42)"],
        must_not_contain=[],
    )
    assert result is False


def test_check_condensation_fails_when_forbidden_pronoun_present():
    result = _check_condensation(
        "Was its level reduced in CSF?",
        must_contain=["Ab(1-42)"],
        must_not_contain=["its"],
    )
    assert result is False


def test_check_condensation_passes_with_no_constraints():
    assert _check_condensation("anything at all", [], []) is True




def test_check_keywords_full_coverage():
    answer = "The Ab(1-42) level was reduced due to plaque deposition and reduced clearance."
    keywords = ["Ab(1-42)", "plaque deposition", "clearance"]
    assert _check_keywords(answer, keywords) == pytest.approx(1.0)


def test_check_keywords_partial_coverage():
    answer = "The Ab(1-42) was reduced in CSF samples."
    keywords = ["Ab(1-42)", "phospho-tau-181"]
    assert _check_keywords(answer, keywords) == pytest.approx(0.5)


def test_check_keywords_no_expected_keywords_returns_one():
    assert _check_keywords("any answer text", []) == 1.0


def test_check_keywords_is_case_insensitive():
    answer = "the ab(1-42) level was reduced"
    assert _check_keywords(answer, ["Ab(1-42)"]) == pytest.approx(1.0)




def test_load_golden_conversations_returns_dataclasses():
    conversations = load_golden_conversations(str(GOLDEN_CONVERSATIONS))
    assert len(conversations) > 0
    convo = conversations[0]
    assert convo.conversation_id
    assert len(convo.turns) > 0
    turn = convo.turns[0]
    assert turn.query
    assert isinstance(turn.expects_condensation, bool)
