from __future__ import annotations

import json
from pathlib import Path
from src.rlhf.feedback_dataset import (
    FeedbackRecord,
    PreferencePair,
    build_preference_pairs,
    load_feedback_records,
    save_preference_pairs,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")




def test_load_feedback_records_missing_dir_returns_empty(tmp_path):
    assert load_feedback_records(tmp_path / "does_not_exist") == []


def test_load_feedback_records_reads_all_subdirs(tmp_path):
    feedback_dir = tmp_path / "feedback"
    _write_jsonl(
        feedback_dir / "liked_responses" / "a.jsonl",
        [
            {
                "query_id": "q1",
                "question": "What is X?",
                "answer": "X is the answer.",
                "rating": "like",
                "session_id": "s1",
            }
        ],
    )
    _write_jsonl(
        feedback_dir / "disliked_responses" / "b.jsonl",
        [
            {
                "query_id": "q2",
                "question": "What is X?",
                "answer": "I cannot answer this from the available context.",
                "rating": "dislike",
                "session_id": "s2",
            }
        ],
    )

    records = load_feedback_records(feedback_dir)

    assert len(records) == 2
    assert all(isinstance(r, FeedbackRecord) for r in records)
    ratings = {r.rating for r in records}
    assert ratings == {"like", "dislike"}


def test_load_feedback_records_skips_records_missing_question_or_answer(tmp_path):
    feedback_dir = tmp_path / "feedback"
    _write_jsonl(
        feedback_dir / "unknown_queries" / "c.jsonl",
        [
            {"query_id": "q1", "question": "", "answer": "something", "rating": "unknown"},
            {"query_id": "q2", "question": "Has no answer field", "rating": "unknown"},
            {"query_id": "q3", "question": "Valid", "answer": "ok", "rating": "unknown"},
        ],
    )

    records = load_feedback_records(feedback_dir)

    assert len(records) == 1
    assert records[0].question == "Valid"


def test_load_feedback_records_handles_malformed_json(tmp_path):
    feedback_dir = tmp_path / "feedback" / "liked_responses"
    feedback_dir.mkdir(parents=True)
    (feedback_dir / "broken.jsonl").write_text(
        '{"query_id": "q1", "question": "Q", "answer": "A", "rating": "like"}\n' "not valid json\n",
        encoding="utf-8",
    )

    records = load_feedback_records(feedback_dir.parent)

    assert len(records) == 1
    assert records[0].question == "Q"




def test_build_preference_pairs_pairs_liked_with_disliked():
    records = [
        FeedbackRecord(
            query_id="q1",
            question="What is X?",
            answer="X is the correct answer.",
            rating="like",
            session_id="s1",
        ),
        FeedbackRecord(
            query_id="q2",
            question="What is X?",
            answer="X is something else entirely.",
            rating="dislike",
            session_id="s2",
        ),
    ]

    pairs = build_preference_pairs(records)

    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, PreferencePair)
    assert pair.prompt == "What is X?"
    assert pair.chosen == "X is the correct answer."
    assert pair.rejected == "X is something else entirely."
    assert pair.meta["rejected_rating"] == "dislike"


def test_build_preference_pairs_excludes_non_answers_as_chosen():
    records = [
        FeedbackRecord(
            query_id="q1",
            question="What is X?",
            answer="I cannot answer this from the available context.",
            rating="like",
        ),
        FeedbackRecord(
            query_id="q2",
            question="What is X?",
            answer="X is something else.",
            rating="dislike",
        ),
    ]

    assert build_preference_pairs(records) == []


def test_build_preference_pairs_skips_identical_chosen_and_rejected():
    records = [
        FeedbackRecord(query_id="q1", question="What is X?", answer="Same answer.", rating="like"),
        FeedbackRecord(
            query_id="q2", question="What is X?", answer="Same answer.", rating="dislike"
        ),
    ]

    assert build_preference_pairs(records) == []


def test_build_preference_pairs_returns_empty_when_no_counterpart():
    records = [
        FeedbackRecord(
            query_id="q1", question="What is X?", answer="X is the answer.", rating="like"
        ),
        FeedbackRecord(
            query_id="q2", question="What is Y?", answer="Y is unclear.", rating="dislike"
        ),
    ]

    # No question has both a liked and disliked/unknown answer.
    assert build_preference_pairs(records) == []


def test_build_preference_pairs_dedupes_duplicate_triples():
    records = [
        FeedbackRecord(query_id="q1", question="What is X?", answer="Good answer.", rating="like"),
        FeedbackRecord(
            query_id="q2", question="What is X?", answer="Bad answer.", rating="dislike"
        ),
        FeedbackRecord(query_id="q3", question="What is X?", answer="Good answer.", rating="like"),
        FeedbackRecord(
            query_id="q4", question="What is X?", answer="Bad answer.", rating="unknown"
        ),
    ]

    pairs = build_preference_pairs(records)

    triples = {(p.prompt, p.chosen, p.rejected) for p in pairs}
    assert len(pairs) == len(triples)




def test_save_preference_pairs_writes_jsonl(tmp_path):
    pairs = [
        PreferencePair(prompt="Q", chosen="good", rejected="bad", meta={"k": "v"}),
    ]
    output_path = tmp_path / "data" / "rlhf" / "preference_pairs.jsonl"

    result = save_preference_pairs(pairs, output_path)

    assert result == output_path
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"prompt": "Q", "chosen": "good", "rejected": "bad", "meta": {"k": "v"}}


def test_save_preference_pairs_empty_list_writes_empty_file(tmp_path):
    output_path = tmp_path / "preference_pairs.jsonl"

    save_preference_pairs([], output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""
