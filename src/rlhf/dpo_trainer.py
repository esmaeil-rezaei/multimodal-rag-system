"""DPO fine-tuning of the policy LLM on preference pairs via TRL DPOTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import get_config
from src.rlhf.reward_model import DEFAULT_PREFERENCE_PAIRS_PATH, load_preference_pairs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dpo_dataset(records: list[dict[str, Any]]):
    """Wrap preference-pair records in a HuggingFace Dataset for DPOTrainer."""
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in records
        ]
    )


def build_dpo_policy_model(cfg: dict[str, Any] | None = None):
    """Build the LoRA-wrapped causal LM and tokenizer for DPO training."""
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rlhf_cfg = get_config().rlhf
    policy_model_name = rlhf_cfg["policy_model"]
    lora_cfg = (cfg or rlhf_cfg["dpo"])["lora"]

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(policy_model_name)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    return model, tokenizer, peft_config


def train_dpo(
    preference_pairs_path: Path | str = DEFAULT_PREFERENCE_PAIRS_PATH,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Run end-to-end DPO training and return the output directory."""
    from trl import DPOConfig, DPOTrainer

    rlhf_cfg = get_config().rlhf
    dpo_cfg = cfg or rlhf_cfg["dpo"]
    train_cfg = dpo_cfg["training"]

    records = load_preference_pairs(preference_pairs_path)
    logger.info(f"Loaded {len(records)} preference pairs for DPO training")

    dataset = build_dpo_dataset(records)
    model, tokenizer, peft_config = build_dpo_policy_model(dpo_cfg)

    output_dir = Path(dpo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        beta=train_cfg["beta"],
        max_length=train_cfg["max_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"DPO-tuned policy adapter saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_dpo()
