"""DPO fine-tuning of the RLHF policy model on preference_pairs.jsonl. See docs/rlhf.md."""

from __future__ import annotations

import argparse

from src.rlhf.dpo_trainer import train_dpo
from src.rlhf.reward_model import DEFAULT_PREFERENCE_PAIRS_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tune the RLHF policy model")
    parser.add_argument("--preference-pairs", default=str(DEFAULT_PREFERENCE_PAIRS_PATH))
    args = parser.parse_args()

    output_dir = train_dpo(preference_pairs_path=args.preference_pairs)
    logger.info(f"Done. DPO policy adapter: {output_dir}")


if __name__ == "__main__":
    main()
