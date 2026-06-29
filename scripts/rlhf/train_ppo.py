"""PPO fine-tuning of the RLHF policy model against the trained reward model. See docs/rlhf.md."""

from __future__ import annotations

import argparse

from src.rlhf.ppo_trainer import DEFAULT_PPO_PROMPTS_PATH, run_ppo_training
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO fine-tune the RLHF policy model")
    parser.add_argument("--ppo-prompts", default=str(DEFAULT_PPO_PROMPTS_PATH))
    args = parser.parse_args()

    output_dir = run_ppo_training(prompts_path=args.ppo_prompts)
    logger.info(f"Done. PPO policy adapter: {output_dir}")


if __name__ == "__main__":
    main()
