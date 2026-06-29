"""Build preference_pairs.jsonl and ppo_prompts.jsonl from operational feedback logs. See docs/rlhf.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rlhf import feedback_dataset
from src.rlhf.feedback_dataset import DEFAULT_FEEDBACK_DIR
from src.rlhf.feedback_dataset import DEFAULT_OUTPUT_PATH as DEFAULT_PREFERENCE_PAIRS_PATH
from src.rlhf.ppo_trainer import DEFAULT_PPO_PROMPTS_PATH
from src.tuning.reranker_dataset import DEFAULT_GOLDEN_QUERIES_PATH, load_golden_queries
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_ppo_prompts(
    feedback_dir: Path | str = DEFAULT_FEEDBACK_DIR,
    golden_queries_path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH,
) -> list[dict[str, str | None]]:
    """Build the PPO rollout prompt pool from feedback logs and golden queries (context set to None for live retrieval)."""
    seen: set[str] = set()
    prompts: list[dict[str, str | None]] = []

    for record in feedback_dataset.load_feedback_records(feedback_dir):
        if record.question not in seen:
            seen.add(record.question)
            prompts.append({"prompt": record.question, "context": None})

    for item in load_golden_queries(golden_queries_path):
        query = item.get("query")
        if query and query not in seen:
            seen.add(query)
            prompts.append({"prompt": query, "context": None})

    return prompts


def save_ppo_prompts(prompts: list[dict[str, str | None]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            fh.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(prompts)} PPO prompts to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RLHF preference-pair and PPO-prompt datasets from feedback logs"
    )
    parser.add_argument("--feedback-dir", default=str(DEFAULT_FEEDBACK_DIR))
    parser.add_argument("--golden-queries", default=str(DEFAULT_GOLDEN_QUERIES_PATH))
    parser.add_argument("--preference-pairs-output", default=str(DEFAULT_PREFERENCE_PAIRS_PATH))
    parser.add_argument("--ppo-prompts-output", default=str(DEFAULT_PPO_PROMPTS_PATH))
    args = parser.parse_args()

    records = feedback_dataset.load_feedback_records(args.feedback_dir)
    pairs = feedback_dataset.build_preference_pairs(records)
    feedback_dataset.save_preference_pairs(pairs, args.preference_pairs_output)

    if not pairs:
        logger.warning(
            "No preference pairs were built — reward model / DPO training need at least "
            "one question with both a liked and a disliked/unknown answer. "
            "Collect more feedback via the /feedback endpoint and re-run this script."
        )

    prompts = build_ppo_prompts(args.feedback_dir, args.golden_queries)
    save_ppo_prompts(prompts, args.ppo_prompts_output)


if __name__ == "__main__":
    main()
