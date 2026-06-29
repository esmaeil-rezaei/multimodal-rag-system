"""Builds RLHF (chosen, rejected) preference pairs from operational feedback logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FEEDBACK_DIR = Path("logs/feedback")
DEFAULT_OUTPUT_PATH = Path("data/rlhf/preference_pairs.jsonl")

_NON_ANSWER_MARKERS = (
    "i cannot answer this from the available context",
    "i've handed off your question to another agent",
    "i don't know",
    "i do not know",
)

_REJECTED_RATINGS = ("dislike", "unknown")


@dataclass
class FeedbackRecord:
    """A single row from `logs/feedback/*`."""

    query_id: str
    question: str
    answer: str
    rating: str
    timestamp: str = ""
    session_id: str = ""
    comment: str = ""
    source_file: str = ""


@dataclass
class PreferencePair:
    """A `(prompt, chosen, rejected)` triple for reward-model / DPO training."""

    prompt: str
    chosen: str
    rejected: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "meta": self.meta,
        }


def _is_non_answer(text: str) -> bool:
    """True if `text` is a generic fallback / non-answer."""
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _NON_ANSWER_MARKERS)


def _iter_json_records(path: Path) -> list[dict[str, Any]]:
    """
    Parse a feedback file into a list of dict records.

    Supports both `.jsonl` (one JSON object per line) and `.json`
    (a single object, or a list of objects).
    """
    records: list[dict[str, Any]] = []

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {path}")
        return records

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"Skipping malformed JSON file {path}")
        return records

    if isinstance(data, list):
        records.extend(data)
    else:
        records.append(data)
    return records


def load_feedback_records(feedback_dir: Path | str = DEFAULT_FEEDBACK_DIR) -> list[FeedbackRecord]:
    """Load every feedback record under `logs/feedback/`; returns [] if directory is missing."""
    feedback_dir = Path(feedback_dir)
    if not feedback_dir.exists():
        logger.warning(f"Feedback directory not found: {feedback_dir}")
        return []

    records: list[FeedbackRecord] = []

    for sub_dir in sorted(feedback_dir.glob("*")):
        if not sub_dir.is_dir():
            continue
        for path in sorted(sub_dir.glob("*.json*")):
            for raw in _iter_json_records(path):
                question = raw.get("question")
                answer = raw.get("answer")
                if not question or answer is None:
                    continue
                records.append(
                    FeedbackRecord(
                        query_id=raw.get("query_id", ""),
                        question=question,
                        answer=answer,
                        rating=raw.get("rating", "unknown"),
                        timestamp=raw.get("timestamp", ""),
                        session_id=raw.get("session_id", ""),
                        comment=raw.get("comment", ""),
                        source_file=str(path),
                    )
                )

    logger.info(f"Loaded {len(records)} feedback records from {feedback_dir}")
    return records


def build_preference_pairs(records: list[FeedbackRecord]) -> list[PreferencePair]:
    """Build (prompt, chosen, rejected) preference pairs from feedback records."""
    by_question: dict[str, list[FeedbackRecord]] = {}
    for rec in records:
        by_question.setdefault(rec.question, []).append(rec)

    pairs: list[PreferencePair] = []
    seen: set[tuple[str, str, str]] = set()

    for question, group in by_question.items():
        chosen_candidates = [
            r for r in group if r.rating == "like" and not _is_non_answer(r.answer)
        ]
        rejected_candidates = [
            r for r in group if r.rating in _REJECTED_RATINGS and r.answer.strip()
        ]

        for chosen_rec in chosen_candidates:
            for rejected_rec in rejected_candidates:
                if chosen_rec.answer == rejected_rec.answer:
                    continue
                key = (question, chosen_rec.answer, rejected_rec.answer)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    PreferencePair(
                        prompt=question,
                        chosen=chosen_rec.answer,
                        rejected=rejected_rec.answer,
                        meta={
                            "chosen_session": chosen_rec.session_id,
                            "rejected_session": rejected_rec.session_id,
                            "rejected_rating": rejected_rec.rating,
                        },
                    )
                )

    logger.info(f"Built {len(pairs)} preference pairs from {len(by_question)} unique questions")
    return pairs


def save_preference_pairs(
    pairs: list[PreferencePair], output_path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    """Write preference pairs to a JSONL file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(pairs)} preference pairs to {output_path}")
    return output_path


def build_and_save(
    feedback_dir: Path | str = DEFAULT_FEEDBACK_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Convenience entry point: load feedback -> build pairs -> write JSONL."""
    records = load_feedback_records(feedback_dir)
    pairs = build_preference_pairs(records)
    return save_preference_pairs(pairs, output_path)


if __name__ == "__main__":
    build_and_save()
