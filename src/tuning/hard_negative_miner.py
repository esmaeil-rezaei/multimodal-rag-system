"""Mines hard-negative passages for reranker and embedding fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.config.settings import get_config
from src.indexing.vector_store import HybridSearchEngine, SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbedFn(Protocol):
    """Embeds a query string into a dense vector for retrieval."""

    def __call__(self, text: str) -> np.ndarray: ...


@dataclass
class HardNegative:
    """A mined hard-negative passage for a given query."""

    chunk_id: str
    text: str
    score: float


def mine_hard_negatives(
    query: str,
    query_vector: np.ndarray,
    positive_chunk_ids: set[str],
    search_engine: HybridSearchEngine,
    cfg: dict[str, Any] | None = None,
) -> list[HardNegative]:
    """Mine hard negatives for a single query, excluding known-positive chunk ids."""
    hn_cfg = cfg or get_config().tuning.get("hard_negatives", {})
    top_k_candidates = hn_cfg.get("top_k_candidates", 20)
    num_per_query = hn_cfg.get("num_per_query", 3)
    floor = hn_cfg.get("similarity_floor", 0.0)
    ceiling = hn_cfg.get("similarity_ceiling", 1.0)

    results: list[SearchResult] = search_engine.search(
        query=query,
        query_vector=query_vector,
        top_k=top_k_candidates,
    )

    candidates: list[HardNegative] = []
    for result in results:
        chunk_id = result.chunk.chunk_id
        if not chunk_id or chunk_id in positive_chunk_ids:
            continue
        if not (floor <= result.score <= ceiling):
            continue
        candidates.append(
            HardNegative(chunk_id=chunk_id, text=result.chunk.text, score=result.score)
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    selected = candidates[:num_per_query]

    logger.debug(
        f"Mined {len(selected)}/{len(candidates)} hard negatives for query={query!r} "
        f"(from {len(results)} candidates, {len(positive_chunk_ids)} positives excluded)"
    )
    return selected


def make_retrieval_negative_sampler(
    search_engine: HybridSearchEngine,
    embed_fn: EmbedFn,
    cfg: dict[str, Any] | None = None,
):
    """Build a negative_sampler callable for use with build_reranker_dataset / build_embedding_dataset."""

    def _sampler(
        query_item: dict[str, Any], _all_queries: list[dict[str, Any]], n: int
    ) -> list[str]:
        query_text = query_item.get("query", "")
        positive_ids = set(query_item.get("relevant_chunk_ids", []))
        query_vector = embed_fn(query_text)

        local_cfg = dict(cfg or get_config().tuning.get("hard_negatives", {}))
        local_cfg["num_per_query"] = n

        negatives = mine_hard_negatives(
            query=query_text,
            query_vector=query_vector,
            positive_chunk_ids=positive_ids,
            search_engine=search_engine,
            cfg=local_cfg,
        )
        return [neg.chunk_id for neg in negatives]

    return _sampler
