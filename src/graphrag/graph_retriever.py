
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.settings import get_config, get_secrets
from src.graphrag.neo4j_store import Neo4jGraphStore
from src.graphrag.schema import EntityType
from src.indexing.vector_store import DenseVectorStore, HybridSearchEngine
from src.ingestion.parser import ParsedChunk
from src.query.understanding import ProcessedQuery
from src.retrieval.retriever import ContextItem, Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

# RRF smoothing constant — same as the rest of the codebase.
_RRF_K = 60

class GraphRetriever:
    """
    Orchestrates graph-aware retrieval on top of the existing vector pipeline.

    Parameters are read from ``config.yaml`` under ``graphrag.retrieval``.
    """

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        vector_retriever: Retriever,
        search_engine: HybridSearchEngine,
        dense_store: DenseVectorStore,
    ) -> None:
        self._graph = graph_store
        self._vector_retriever = vector_retriever
        self._search_engine = search_engine
        self._dense_store = dense_store

        cfg = get_config()
        self._gr_cfg = cfg.get("graphrag", {})
        self._ret_cfg = self._gr_cfg.get("retrieval", {})
        self._global_top_k: int = self._ret_cfg.get("community_top_k", 5)
        self._local_hop_depth: int = self._ret_cfg.get("local_hop_depth", 2)
        self._max_entity_sources: int = self._ret_cfg.get("max_entity_source_chunks", 10)
        self._rrf_k: int = self._ret_cfg.get("rrf_k", _RRF_K)
        self._graph_available: bool = True  # set False on first connection failure

    def retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[ContextItem]:
        """
        Main graph-aware retrieval entry point.

        Args:
            pq:           Processed query from the query understanding stage.
            query_vector: Dense embedding of the final query.
            namespace:    Multi-tenant namespace (mirrors vector retriever).
            mode:         ``"local"`` | ``"global"`` | ``"hybrid"`` (default).
                          ``"local"``  — entity neighbourhood only.
                          ``"global"`` — community summaries only.
                          ``"hybrid"`` — local + global + vector, fused with RRF.

        Returns:
            List[ContextItem] ready for the generation stage.
        """
        if not self._graph_available:
            logger.warning(
                "Graph store unavailable — falling back to vector-only retrieval"
            )
            return self._vector_fallback(pq, query_vector, namespace)

        try:
            if mode == "local":
                return self._local_retrieve(pq, query_vector, namespace)
            elif mode == "global":
                return self._global_retrieve(pq, query_vector, namespace)
            else:  # "hybrid" — default
                return self._hybrid_retrieve(pq, query_vector, namespace)

        except Exception as exc:
            logger.error(
                "GraphRetriever.retrieve failed (mode=%s): %s — "
                "falling back to vector retrieval",
                mode, exc, exc_info=True,
            )
            self._graph_available = False
            return self._vector_fallback(pq, query_vector, namespace)

    def _local_retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str],
    ) -> List[ContextItem]:
        """
        Entity-centric local graph retrieval.

        Steps:
        1. Extract entity names from the processed query (spaCy entities + sub-questions).
        2. Look up those entities in Neo4j by name (exact + fuzzy).
        3. Traverse k-hop neighbourhood.
        4. Collect source chunk_ids from entity and relationship provenance.
        5. Fetch those chunks from the vector store.
        6. Wrap as ContextItems.
        """
        query_entities = self._extract_query_entities(pq)
        if not query_entities:
            logger.debug("No query entities extracted — local graph retrieval skipped")
            return []

        node_ids, entity_chunks = self._resolve_entities(query_entities)
        if not node_ids:
            logger.debug("No entity nodes matched in graph")
            return []

        # Graph traversal: collect neighbours and their chunk provenance
        neighbour_chunks: List[str] = []
        for node_id in node_ids:
            try:
                neighbours = self._graph.get_entity_neighbors(
                    node_id=node_id,
                    max_hops=self._local_hop_depth,
                    limit=30,
                )
                for nb in neighbours:
                    nb_chunks = self._graph.get_chunks_for_entity(
                        nb["entity"]["node_id"]
                    )
                    neighbour_chunks.extend(nb_chunks)
            except Exception as exc:
                logger.warning("Neighbour traversal failed for %s: %s", node_id, exc)

        all_chunk_ids = list(dict.fromkeys(entity_chunks + neighbour_chunks))
        all_chunk_ids = all_chunk_ids[: self._max_entity_sources]

        if not all_chunk_ids:
            return []

        context_items = self._fetch_chunks_as_context(all_chunk_ids, query_vector)
        logger.info(
            "Local graph retrieval: %d entities → %d source chunks → %d context items",
            len(node_ids), len(all_chunk_ids), len(context_items),
        )
        return context_items

    def _global_retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str],
    ) -> List[ContextItem]:
        """
        Community-level global retrieval.

        Embeds the query, runs ANN search over community summary embeddings,
        and returns community summaries as pseudo-chunks so the generator
        can synthesise a high-level answer.
        """
        try:
            hits = self._graph.vector_search_entities(
                query_embedding=query_vector.tolist(),
                top_k=self._global_top_k,
            )
        except Exception as exc:
            logger.warning("Community vector search failed: %s", exc)
            return []

        context_items: List[ContextItem] = []
        for rank, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            community_id = entity.get("community_id")
            if not community_id:
                continue
            summary = self._graph.get_community_summary(community_id)
            if not summary:
                continue

            pseudo_chunk = ParsedChunk(
                text=summary,
                source_name=f"community:{community_id[:8]}",
                modality="text",
                metadata={
                    "community_id": community_id,
                    "community_level": entity.get("level", 0),
                    "retrieval_source": "graph_community",
                },
            )
            context_items.append(
                ContextItem(chunk=pseudo_chunk, score=float(hit.get("score", 0.0)), rank=rank)
            )

        logger.info(
            "Global community retrieval: %d community summaries returned",
            len(context_items),
        )
        return context_items

    def _hybrid_retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str],
    ) -> List[ContextItem]:
        """
        Full hybrid retrieval:
        - Local graph context (entity neighbourhood)
        - Global community context (thematic summaries)
        - Vector retrieval (existing Retriever)
        All three are merged via Reciprocal Rank Fusion.
        """
        local_items = self._local_retrieve(pq, query_vector, namespace)
        global_items = self._global_retrieve(pq, query_vector, namespace)
        vector_items = self._vector_fallback(pq, query_vector, namespace)

        merged = self._rrf_merge([local_items, global_items, vector_items])
        logger.info(
            "Hybrid graph+vector retrieval: local=%d, global=%d, vector=%d → merged=%d",
            len(local_items), len(global_items), len(vector_items), len(merged),
        )
        return merged

    def _rrf_merge(self, lists: List[List[ContextItem]]) -> List[ContextItem]:
        """
        Merge multiple ContextItem lists with Reciprocal Rank Fusion.

        Items are keyed by (chunk_id or first-50-chars).  Score from each list
        is 1 / (rrf_k + rank).  The merged list is sorted descending by total
        RRF score and capped at the configured top_k_final.
        """
        cfg = get_config()
        top_k_final: int = cfg.retrieval.get("top_k_final", 5)
        scores: Dict[str, float] = {}
        item_map: Dict[str, ContextItem] = {}

        for result_list in lists:
            for rank, item in enumerate(result_list, start=1):
                key = item.chunk.chunk_id or item.chunk.text[:50]
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                item_map.setdefault(key, item)

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        merged = []
        for final_rank, key in enumerate(sorted_keys[:top_k_final], start=1):
            item = item_map[key]
            item.score = scores[key]
            item.rank = final_rank
            merged.append(item)

        return merged

    def _extract_query_entities(self, pq: ProcessedQuery) -> List[str]:
        """
        Extract potential entity names from the processed query.

        Sources (in priority order):
        1. Named entities from spaCy NER (stored in pq.metadata_filters["entities"])
        2. Capitalised phrases in the standalone query (simple heuristic)
        3. Sub-question text (extra entities may appear there)
        """
        entities: List[str] = []

        # 1. NER results from QueryUnderstanding
        ner_entities = pq.metadata_filters.get("entities", [])
        if isinstance(ner_entities, list):
            entities.extend(str(e) for e in ner_entities)

        # 2. Capitalised token heuristic (noun phrases)
        for text in [pq.standalone_query] + list(pq.sub_questions):
            tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
            entities.extend(tokens)

        # Deduplicate preserving order
        seen = set()
        unique: List[str] = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)

        return unique

    def _resolve_entities(
        self, entity_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Look up entity names in Neo4j.

        Returns:
            node_ids    — list of matched entity node_ids
            chunk_ids   — deduplicated chunk_ids from matched entities
        """
        node_ids: List[str] = []
        chunk_ids: List[str] = []

        for name in entity_names:
            try:
                entity = self._graph.get_entity_by_name(name)
                if entity and entity.get("node_id"):
                    node_ids.append(entity["node_id"])
                    chunks = self._graph.get_chunks_for_entity(entity["node_id"])
                    chunk_ids.extend(chunks)
            except Exception as exc:
                logger.debug("Entity lookup failed for '%s': %s", name, exc)

        node_ids = list(dict.fromkeys(node_ids))
        chunk_ids = list(dict.fromkeys(chunk_ids))
        return node_ids, chunk_ids

    def _fetch_chunks_as_context(
        self, chunk_ids: List[str], query_vector: np.ndarray
    ) -> List[ContextItem]:
        """
        Fetch ParsedChunks from the vector store by chunk_id and wrap them
        as ContextItems with a graph-provenance score.
        """
        context_items: List[ContextItem] = []
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            chunk = self._vector_retriever._fetch_chunk_by_id(chunk_id)
            if chunk:
                # Score = rank-based decay (1/rank) — we don't have a dot-product
                # score here since this is provenance-driven, not similarity-driven.
                score = 1.0 / rank
                context_items.append(
                    ContextItem(chunk=chunk, score=score, rank=rank)
                )
        return context_items

    def _vector_fallback(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str],
    ) -> List[ContextItem]:
        """Delegate to the existing vector Retriever as a fallback."""
        try:
            return self._vector_retriever.retrieve(
                pq=pq, query_vector=query_vector, namespace=namespace
            )
        except Exception as exc:
            logger.error("Vector retrieval fallback also failed: %s", exc)
            return []
