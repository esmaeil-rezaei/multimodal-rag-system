"""PPO fine-tuning of the policy LLM against the composite reward signal via TRL PPOTrainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import get_config
from src.rlhf.reward_functions import composite_reward
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PPO_PROMPTS_PATH = Path("data/rlhf/ppo_prompts.jsonl")


def load_ppo_prompts(path: Path | str = DEFAULT_PPO_PROMPTS_PATH) -> list[dict[str, Any]]:
    """Load PPO rollout prompts from a JSONL file; each record has {prompt, context}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No PPO prompts found at {path}. "
            "Run `python -m scripts.rlhf.build_preference_dataset` to generate it."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{path} is empty — no PPO prompts to train on.")

    return records


def build_policy_model(cfg: dict[str, Any] | None = None):
    """Build the LoRA-wrapped policy model (with value head) and tokenizer for PPO."""
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    rlhf_cfg = get_config().rlhf
    policy_model_name = rlhf_cfg["policy_model"]
    lora_cfg = (cfg or rlhf_cfg["ppo"])["lora"]

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        peft_config=peft_config,
    )
    ref_model = None

    return model, ref_model, tokenizer


def run_ppo_training(
    prompts_path: Path | str = DEFAULT_PPO_PROMPTS_PATH,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Run the end-to-end PPO training loop and return the output directory."""
    import torch
    from trl import PPOConfig, PPOTrainer

    from src.rlhf.reward_model import load_trained_reward_model

    rlhf_cfg = get_config().rlhf
    ppo_cfg = cfg or rlhf_cfg["ppo"]
    train_cfg = ppo_cfg["training"]

    prompts = load_ppo_prompts(prompts_path)
    logger.info(f"Loaded {len(prompts)} PPO prompts")

    model, ref_model, tokenizer = build_policy_model(ppo_cfg)
    reward_model, reward_tokenizer = load_trained_reward_model()

    ppo_config = PPOConfig(
        model_name=rlhf_cfg["policy_model"],
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
        mini_batch_size=train_cfg["mini_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        ppo_epochs=train_cfg["ppo_epochs"],
        init_kl_coef=train_cfg["init_kl_coef"],
        target=train_cfg["target_kl"],
    )

    def _collator(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {key: [d[key] for d in data] for key in data[0]}

    dataset = [
        {
            "input_ids": tokenizer(p["prompt"], return_tensors="pt").input_ids[0],
            "prompt": p["prompt"],
            "context": p.get("context"),
        }
        for p in prompts
    ]

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=_collator,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": train_cfg["max_new_tokens"],
    }

    total_episodes = train_cfg.get("total_episodes", len(dataset))
    episodes_done = 0

    for batch in ppo_trainer.dataloader:
        if episodes_done >= total_episodes:
            break

        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors
        ]

        rewards = [
            torch.tensor(
                composite_reward(
                    prompt=prompt,
                    response=response,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    context=context,
                ).total
            )
            for prompt, response, context in zip(
                batch["prompt"], batch["response"], batch["context"], strict=False
            )
        ]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        episodes_done += len(query_tensors)
        logger.info(f"PPO step done — {episodes_done}/{total_episodes} episodes")

    output_dir = Path(ppo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"PPO-tuned policy adapter saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_ppo_training()
