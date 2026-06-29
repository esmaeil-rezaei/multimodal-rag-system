"""Fine-tunes the bi-encoder embedding model on (anchor, positive, negative) triplets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path("data/tuning/embedding_triplets.jsonl")


def load_embedding_records(path: Path | str = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load `(anchor, positive, negative)` records written by `src.tuning.embedding_dataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No embedding training data at {path}. "
            "Run `python -m src.tuning.embedding_dataset` to build it."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{path} is empty — no embedding training triplets.")

    return records


def build_input_examples(records: list[dict[str, Any]]):
    """Convert records into `sentence_transformers.InputExample` triplets."""
    from sentence_transformers import InputExample

    return [InputExample(texts=[r["anchor"], r["positive"], r["negative"]]) for r in records]


def train_embedder(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Fine-tune the bi-encoder embedding model and return the output directory."""
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    emb_cfg = cfg or get_config().tuning["embedder"]
    train_cfg = emb_cfg["training"]

    records = load_embedding_records(dataset_path)
    logger.info(f"Loaded {len(records)} embedding training triplets")

    examples = build_input_examples(records)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=train_cfg["batch_size"])

    model = SentenceTransformer(emb_cfg["base_model"])
    train_loss = losses.MultipleNegativesRankingLoss(model)

    num_epochs = train_cfg["num_epochs"]
    warmup_steps = int(len(train_dataloader) * num_epochs * train_cfg["warmup_ratio"])

    output_dir = Path(emb_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": train_cfg["learning_rate"]},
        output_path=str(output_dir),
    )

    logger.info(f"Fine-tuned embedding model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_embedder()
