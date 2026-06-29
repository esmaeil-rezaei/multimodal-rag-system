"""Fine-tunes the cross-encoder reranker on (query, passage, label) triples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path("data/tuning/reranker_pairs.jsonl")


def load_reranker_records(path: Path | str = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load `(query, passage, label)` records written by `src.tuning.reranker_dataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No reranker training data at {path}. "
            "Run `python -m src.tuning.reranker_dataset` to build it."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{path} is empty — no reranker training examples.")

    return records


def build_input_examples(records: list[dict[str, Any]]):
    """Convert records into `sentence_transformers.InputExample`s for `CrossEncoder.fit`."""
    from sentence_transformers import InputExample

    return [
        InputExample(texts=[r["query"], r["passage"]], label=float(r["label"])) for r in records
    ]


def train_reranker(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Fine-tune the cross-encoder reranker and return the output directory."""
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    rr_cfg = cfg or get_config().tuning["reranker"]
    train_cfg = rr_cfg["training"]

    records = load_reranker_records(dataset_path)
    logger.info(f"Loaded {len(records)} reranker training examples")

    examples = build_input_examples(records)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=train_cfg["batch_size"])

    model = CrossEncoder(rr_cfg["base_model"], num_labels=1, max_length=train_cfg["max_length"])

    num_epochs = train_cfg["num_epochs"]
    warmup_steps = int(len(train_dataloader) * num_epochs * train_cfg["warmup_ratio"])

    output_dir = Path(rr_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": train_cfg["learning_rate"]},
        output_path=str(output_dir),
    )

    logger.info(f"Fine-tuned reranker saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_reranker()
