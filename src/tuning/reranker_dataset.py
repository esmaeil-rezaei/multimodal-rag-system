"""Builds (query, passage, label) triples for cross-encoder reranker fine-tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GOLDEN_QUERIES_PATH = Path("tests/golden_queries.json")
DEFAULT_OUTPUT_PATH = Path("data/tuning/reranker_pairs.jsonl")

DEFAULT_NEGATIVES_PER_QUERY = 2


class ChunkResolver(Protocol):
    """Resolves a `chunk_id` to its passage text."""

    def resolve(self, chunk_id: str) -> str | None: ...


class InMemoryChunkResolver:
    """A `ChunkResolver` backed by a plain `{chunk_id: text}` dict — useful for tests/offline runs."""

    def __init__(self, chunks: dict[str, str]) -> None:
        self._chunks = chunks

    def resolve(self, chunk_id: str) -> str | None:
        return self._chunks.get(chunk_id)


class QdrantChunkResolver:
    """A `ChunkResolver` backed by the live Qdrant `DenseVectorStore`."""

    def __init__(self, dense_store: Any) -> None:
        self._dense_store = dense_store

    def resolve(self, chunk_id: str) -> str | None:
        try:
            from src.indexing.vector_store import DenseVectorStore

            points = self._dense_store._client.retrieve(
                collection_name=self._dense_store._collection,
                ids=[DenseVectorStore._chunk_id_to_int(chunk_id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # pragma: no cover - network/Qdrant errors
            logger.warning(f"Chunk resolution failed for {chunk_id}: {exc}")
            return None

        if not points:
            return None
        payload = points[0].payload or {}
        return payload.get("text")


@dataclass
class RerankerExample:
    """A single `(query, passage, label)` training example."""

    query: str
    passage: str
    label: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "passage": self.passage,
            "label": self.label,
            "meta": self.meta,
        }


def load_golden_queries(path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH) -> list[dict[str, Any]]:
    """Load `tests/golden_queries.json`."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Golden queries file not found: {path}")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def build_reranker_dataset(
    golden_queries: list[dict[str, Any]],
    resolver: ChunkResolver,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
    negative_sampler: Any | None = None,
    seed: int = 42,
) -> list[RerankerExample]:
    """Build reranker training triples from golden queries (positives label=1.0, negatives label=0.0)."""
    rng = random.Random(seed)
    examples: list[RerankerExample] = []

    all_positive_ids: list[tuple[str, int]] = [
        (cid, i) for i, q in enumerate(golden_queries) for cid in q.get("relevant_chunk_ids", [])
    ]

    for idx, item in enumerate(golden_queries):
        query_text = item.get("query")
        relevant_ids = item.get("relevant_chunk_ids", [])
        if not query_text or not relevant_ids:
            continue

        for chunk_id in relevant_ids:
            passage = resolver.resolve(chunk_id)
            if not passage:
                logger.debug(f"Could not resolve chunk_id={chunk_id!r} for query={query_text!r}")
                continue
            examples.append(
                RerankerExample(
                    query=query_text,
                    passage=passage,
                    label=1.0,
                    meta={"chunk_id": chunk_id, "namespace": item.get("namespace")},
                )
            )

        if negative_sampler is not None:
            negative_ids = negative_sampler(item, golden_queries, negatives_per_query)
        else:
            candidates = [
                cid for cid, owner in all_positive_ids if owner != idx and cid not in relevant_ids
            ]
            rng.shuffle(candidates)
            negative_ids = candidates[:negatives_per_query]

        for chunk_id in negative_ids:
            passage = resolver.resolve(chunk_id)
            if not passage:
                continue
            examples.append(
                RerankerExample(
                    query=query_text,
                    passage=passage,
                    label=0.0,
                    meta={
                        "chunk_id": chunk_id,
                        "namespace": item.get("namespace"),
                        "negative": True,
                    },
                )
            )

    logger.info(
        f"Built {len(examples)} reranker examples "
        f"({sum(1 for e in examples if e.label == 1.0)} positive, "
        f"{sum(1 for e in examples if e.label == 0.0)} negative) "
        f"from {len(golden_queries)} golden queries"
    )
    return examples


def save_reranker_dataset(
    examples: list[RerankerExample], output_path: Path | str = DEFAULT_OUTPUT_PATH
) -> Path:
    """Write reranker examples to a JSONL file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} reranker examples to {output_path}")
    return output_path


def build_and_save(
    golden_queries_path: Path | str = DEFAULT_GOLDEN_QUERIES_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
) -> Path:
    """Convenience entry point: load golden queries -> resolve chunks via Qdrant -> write JSONL."""
    from src.indexing.vector_store import DenseVectorStore

    golden_queries = load_golden_queries(golden_queries_path)
    resolver = QdrantChunkResolver(DenseVectorStore())
    examples = build_reranker_dataset(
        golden_queries, resolver, negatives_per_query=negatives_per_query
    )
    return save_reranker_dataset(examples, output_path)


if __name__ == "__main__":
    build_and_save()
