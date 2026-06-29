"""Builds (anchor, positive, negative) triplets for bi-encoder embedding fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tuning.reranker_dataset import (
    DEFAULT_GOLDEN_QUERIES_PATH,
    ChunkResolver,
    QdrantChunkResolver,
    load_golden_queries,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/tuning/embedding_triplets.jsonl")


@dataclass
class EmbeddingTriplet:
    """A single `(anchor, positive, negative)` training example."""

    anchor: str
    positive: str
    negative: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
            "meta": self.meta,
        }


def build_embedding_dataset(
    golden_queries: list[dict[str, Any]],
    resolver: ChunkResolver,
    negative_sampler: Any | None = None,
    seed: int = 42,
) -> list[EmbeddingTriplet]:
    """Build (anchor, positive, negative) triplets from golden queries."""
    rng = random.Random(seed)
    triplets: list[EmbeddingTriplet] = []

    all_positive_ids: list[tuple[str, int]] = [
        (cid, i) for i, q in enumerate(golden_queries) for cid in q.get("relevant_chunk_ids", [])
    ]

    for idx, item in enumerate(golden_queries):
        query_text = item.get("query")
        relevant_ids = item.get("relevant_chunk_ids", [])
        if not query_text or not relevant_ids:
            continue

        if negative_sampler is not None:
            negative_ids = negative_sampler(item, golden_queries, len(relevant_ids))
        else:
            candidates = [
                cid for cid, owner in all_positive_ids if owner != idx and cid not in relevant_ids
            ]
            rng.shuffle(candidates)
            negative_ids = candidates[: len(relevant_ids)]

        if not negative_ids:
            logger.debug(f"No negatives available for query={query_text!r} — skipping")
            continue

        for i, chunk_id in enumerate(relevant_ids):
            positive_text = resolver.resolve(chunk_id)
            if not positive_text:
                logger.debug(
                    f"Could not resolve positive chunk_id={chunk_id!r} for query={query_text!r}"
                )
                continue

            negative_id = negative_ids[i % len(negative_ids)]
            negative_text = resolver.resolve(negative_id)
            if not negative_text:
                logger.debug(
                    f"Could not resolve negative chunk_id={negative_id!r} for query={query_text!r}"
                )
                continue

            triplets.append(
                EmbeddingTriplet(
                    anchor=query_text,
                    positive=positive_text,
                    negative=negative_text,
                    meta={
                        "positive_chunk_id": chunk_id,
                        "negative_chunk_id": negative_id,
                        "namespace": item.get("namespace"),
                    },
                )
            )

    logger.info(
        f"Built {len(triplets)} embedding triplets from {len(golden_queries)} golden queries"
    )
    return triplets


def save_embedding_dataset(
    triplets: list[EmbeddingTriplet], output_path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    """Write embedding triplets to a JSONL file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for triplet in triplets:
            fh.write(json.dumps(triplet.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(triplets)} embedding triplets to {output_path}")
    return output_path


def build_and_save(
    golden_queries_path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Convenience entry point: load golden queries -> resolve chunks via Qdrant -> write JSONL."""
    from src.indexing.vector_store import DenseVectorStore

    golden_queries = load_golden_queries(golden_queries_path)
    resolver = QdrantChunkResolver(DenseVectorStore())
    triplets = build_embedding_dataset(golden_queries, resolver)
    return save_embedding_dataset(triplets, output_path)


if __name__ == "__main__":
    build_and_save()
