"""Fine-tune the cross-encoder reranker on triples from golden queries. See docs/tuning.md."""

from __future__ import annotations

import argparse

from src.tuning.reranker_dataset import DEFAULT_OUTPUT_PATH as DEFAULT_DATASET_PATH
from src.tuning.reranker_dataset import build_and_save as build_reranker_dataset
from src.tuning.reranker_trainer import train_reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the cross-encoder reranker")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Rebuild the dataset from tests/golden_queries.json before training "
        "(requires a live Qdrant connection)",
    )
    args = parser.parse_args()

    if args.build_dataset:
        build_reranker_dataset(output_path=args.dataset)

    output_dir = train_reranker(dataset_path=args.dataset)
    logger.info(f"Done. Fine-tuned reranker: {output_dir}")


if __name__ == "__main__":
    main()
