"""Trains a scalar reward model on (prompt, chosen, rejected) preference pairs via TRL RewardTrainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PREFERENCE_PAIRS_PATH = Path("data/rlhf/preference_pairs.jsonl")


def load_preference_pairs(path: Path | str = DEFAULT_PREFERENCE_PAIRS_PATH) -> list[dict[str, Any]]:
    """
    Load `(prompt, chosen, rejected)` records written by
    `src.rlhf.feedback_dataset.save_preference_pairs`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No preference pairs found at {path}. "
            "Run `python -m src.rlhf.feedback_dataset` to build it from feedback logs."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{path} is empty — no preference pairs to train on.")

    return records


def build_reward_dataset(records: list[dict[str, Any]]):
    """Wrap preference-pair records in a HuggingFace `datasets.Dataset`."""
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


def tokenize_reward_dataset(dataset, tokenizer, max_length: int):
    """Tokenize preference pairs into the column layout expected by TRL's RewardTrainer."""

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[Any]]:
        chosen_full = [p + c for p, c in zip(batch["prompt"], batch["chosen"], strict=False)]
        rejected_full = [p + r for p, r in zip(batch["prompt"], batch["rejected"], strict=False)]

        chosen_enc = tokenizer(chosen_full, truncation=True, max_length=max_length)
        rejected_enc = tokenizer(rejected_full, truncation=True, max_length=max_length)

        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def build_reward_model(cfg: dict[str, Any] | None = None):
    """Build the LoRA-wrapped sequence-classification reward model and tokenizer."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = cfg or get_config().rlhf["reward_model"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"],
        num_labels=cfg.get("num_labels", 1),
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_reward_model(
    preference_pairs_path: Path | str = DEFAULT_PREFERENCE_PAIRS_PATH,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Train the reward model end-to-end and return the output directory."""
    from trl import RewardConfig, RewardTrainer

    rlhf_cfg = get_config().rlhf
    rm_cfg = cfg or rlhf_cfg["reward_model"]
    train_cfg = rm_cfg["training"]

    records = load_preference_pairs(preference_pairs_path)
    logger.info(f"Loaded {len(records)} preference pairs for reward model training")

    dataset = build_reward_dataset(records)
    model, tokenizer = build_reward_model(rm_cfg)
    dataset = tokenize_reward_dataset(dataset, tokenizer, max_length=train_cfg["max_length"])

    output_dir = Path(rm_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    args = RewardConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        max_length=train_cfg["max_length"],
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Reward model adapter saved to {output_dir}")
    return output_dir


def load_trained_reward_model(output_dir: Path | str | None = None):
    """Load a trained reward model (base + LoRA adapter) in eval mode."""
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rm_cfg = get_config().rlhf["reward_model"]
    output_dir = Path(output_dir or rm_cfg["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        rm_cfg["base_model"],
        num_labels=rm_cfg.get("num_labels", 1),
    )
    model = PeftModel.from_pretrained(base_model, str(output_dir))
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    train_reward_model()
