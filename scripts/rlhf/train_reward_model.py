"""Train the RLHF reward model on preference_pairs.jsonl. See docs/rlhf.md."""

from __future__ import annotations

import argparse

from src.rlhf.reward_model import DEFAULT_PREFERENCE_PAIRS_PATH, train_reward_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RLHF reward model")
    parser.add_argument("--preference-pairs", default=str(DEFAULT_PREFERENCE_PAIRS_PATH))
    args = parser.parse_args()

    output_dir = train_reward_model(preference_pairs_path=args.preference_pairs)
    logger.info(f"Done. Reward model adapter: {output_dir}")


if __name__ == "__main__":
    main()
