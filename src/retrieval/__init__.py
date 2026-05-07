# src/retrieval/retriever.py
# =============================================================================
# Stage 4 — Retrieval & Re-ranking.
# Covers:
#   Challenge 15 — Low recall at top-k (parent-child, sentence window)
#   Challenge 16 — Re-ranking quality (cross-encoder, ColBERT, LLM reranker)
#   Challenge 17 — Lost-in-the-middle (position-aware ordering, compression)
#   Challenge 18 — Context window overflow (LLMLingua, map-reduce)
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np                                  # Embedding vectors
import openai                                       # OpenAI client for LLM-based reranking
import cohere                                       # Cohere client for cross-encoder reranking
from sentence_transformers import CrossEncoder      # BGE / local cross-encoder reranker

from src.config.settings import get_config, get_secrets  # Configuration accessors
from src.indexing.vector_store import (
    HybridSearchEngine,
    DenseVectorStore,
    SparseIndex,
    SearchResult,
)
from src.ingestion.parser import ParsedChunk         # Data model
from src.query.understanding import ProcessedQuery   # Query data model
from src.utils.logger import get_logger              # Structured logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Context item — reranked and ready for the generator
# ---------------------------------------------------------------------------

class ContextItem:
    """
    A single piece of context to be passed to the generation stage.
    Carries the chunk, its final rank, and its source score.
    """
    def __init__(self, chunk: ParsedChunk, score: float, rank: int) -> None:
        self.chunk = chunk                          # The ParsedChunk with text + metadata
        self.score = score                          # Final relevance score after reranking
        self.rank = rank                            # 1-based position in the final context window


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Orchestrates the full retrieval pipeline:
      ANN/hybrid search → parent-child expansion → reranking → context ordering → compression
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        dense_store: DenseVectorStore,
    ) -> None:
        self._search_engine = search_engine         # Hybrid (dense + sparse) search engine
        self._dense_store = dense_store             # Direct Qdrant access for parent lookups
        cfg = get_config()
        sec = get_secrets()
        self._ret_cfg = cfg.retrieval               # Retrieval settings from YAML
        self._ctx_cfg = cfg.retrieval["context_management"]  # Context overflow settings
        self._openai = openai.OpenAI(api_key=sec.openai_api_key)
        self._cohere = cohere.Client(sec.cohere_api_key)    # Cohere reranker client

        # Lazy-load local cross-encoder if configured
        self._cross_encoder: Optional[CrossEncoder] = None

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        namespace: Optional[str] = None,
    ) -> List[ContextItem]:
        """
        Full retrieval pipeline for a processed query.

        Args:
            pq:           ProcessedQuery from the query understanding stage.
            query_vector: Dense embedding of the final query.
            namespace:    Tenant namespace for ACL filtering (Challenge 26).

        Returns:
            Ordered list of ContextItems ready for the generation stage.
        """
        top_k_initial = self._ret_cfg["top_k_initial"]   # Over-retrieve before reranking
        top_k_final = self._ret_cfg["top_k_final"]        # Final k after reranking

        # -- Step 1: Initial retrieval (hybrid ANN + BM25) --------------------
        # NER-derived metadata filters (Challenge 14) are only applied when Qdrant
        # payload indexes exist for those fields.  Pass None to avoid 400 errors
        # on collections that don't have dynamic entity indexes pre-built.
        raw_results: List[SearchResult] = self._search_engine.search(
            query=pq.final_query(),
            query_vector=query_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,                     # NER filters skipped — no payload indexes for entity_* fields
        )
        logger.info(f"Initial retrieval: {len(raw_results)} candidates")

        # -- Step 2: Parent-child expansion (Challenge 15) --------------------
        if self._ret_cfg["parent_child"]["enabled"]:
            raw_results = self._expand_to_parents(raw_results)

        # -- Step 3: Sentence window expansion (Challenge 15) -----------------
        if self._ret_cfg["sentence_window"]["enabled"]:
            raw_results = self._expand_sentence_window(raw_results)

        # -- Step 4: Reranking (Challenge 16) ---------------------------------
        if self._ret_cfg["reranking"]["enabled"]:
            raw_results = self._rerank(pq.final_query(), raw_results)

        # Trim to final top-k after reranking
        raw_results = raw_results[:top_k_final]

        # -- Step 5: Position-aware ordering (Challenge 17) -------------------
        ordered = self._apply_position_aware_ordering(raw_results)

        # -- Step 6: Context compression / overflow management (Challenge 18) -
        context_items = self._manage_context(ordered, pq.final_query())

        logger.info(f"Final context: {len(context_items)} items")
        return context_items

    # -------------------------------------------------------------------------
    # HyDE dual retrieval — merges query vector + hypothetical doc vector
    # -------------------------------------------------------------------------

    def retrieve_dual(
        self,
        pq: ProcessedQuery,
        query_vector: np.ndarray,
        hyde_vector: np.ndarray,
        namespace: Optional[str] = None,
    ) -> List[ContextItem]:
        """
        Run retrieval twice — once with the raw query vector and once with the
        HyDE hypothetical document vector — then merge both result lists via
        Reciprocal Rank Fusion before reranking.

        This solves the vocabulary mismatch problem: interrogative queries like
        "Who is X?" embed very differently from declarative answers like "X is a...".
        The HyDE vector sits close to the answer in vector space, so combining both
        retrieval passes dramatically improves recall for biographical and factual queries.
        """
        top_k_initial = self._ret_cfg["top_k_initial"]
        top_k_final   = self._ret_cfg["top_k_final"]
        rrf_k         = 60                              # Standard RRF smoothing constant

        # -- Retrieval pass 1: raw query vector -----------------------------------
        results_query = self._search_engine.search(
            query=pq.final_query(),
            query_vector=query_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,
        )

        # -- Retrieval pass 2: HyDE hypothetical document vector ------------------
        results_hyde = self._search_engine.search(
            query=pq.hypothetical_doc or pq.final_query(),
            query_vector=hyde_vector,
            top_k=top_k_initial,
            namespace=namespace,
            metadata_filter=None,
        )

        # -- Reciprocal Rank Fusion -----------------------------------------------
        scores: dict = {}
        chunk_map: dict = {}

        for rank, result in enumerate(results_query, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:50]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map.setdefault(cid, result)

        for rank, result in enumerate(results_hyde, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:50]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_map.setdefault(cid, result)

        # Sort by descending RRF score
        sorted_cids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k_initial]
        raw_results: List[SearchResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            r = chunk_map[cid]
            r.score = scores[cid]
            r.rank  = rank
            r.retrieval_method = "hyde_dual"
            raw_results.append(r)

        logger.info(
            f"Dual retrieval merged: {len(results_query)} query results + "
            f"{len(results_hyde)} HyDE results → {len(raw_results)} unique candidates"
        )

        # -- Rest of the pipeline (reranking, ordering, compression) --------------
        if self._ret_cfg["parent_child"]["enabled"]:
            raw_results = self._expand_to_parents(raw_results)
        if self._ret_cfg["sentence_window"]["enabled"]:
            raw_results = self._expand_sentence_window(raw_results)
        if self._ret_cfg["reranking"]["enabled"]:
            raw_results = self._rerank(pq.final_query(), raw_results)

        raw_results = raw_results[:top_k_final]
        ordered     = self._apply_position_aware_ordering(raw_results)
        context_items = self._manage_context(ordered, pq.final_query())

        logger.info(f"Final context (dual): {len(context_items)} items")
        return context_items

    # -------------------------------------------------------------------------
    # Challenge 15: Parent-child retrieval
    # -------------------------------------------------------------------------

    def _expand_to_parents(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Replace retrieved child chunks with their parent section chunks.
        This provides broader context while keeping retrieval focused on precise matches.
        The parent chunk has the same metadata but encompasses a larger text window.
        """
        expanded: List[SearchResult] = []
        seen_parent_ids = set()                     # Prevent duplicate parents

        for result in results:
            parent_id = result.chunk.metadata.get("parent_id")  # Set during hierarchical chunking
            if parent_id and parent_id not in seen_parent_ids:
                # Attempt to fetch the parent chunk from Qdrant
                parent_chunk = self._fetch_chunk_by_id(parent_id)
                if parent_chunk:
                    parent_result = SearchResult(
                        chunk=parent_chunk,
                        score=result.score,         # Inherit child's relevance score
                        retrieval_method=result.retrieval_method,
                        rank=result.rank,
                    )
                    expanded.append(parent_result)
                    seen_parent_ids.add(parent_id)
                    continue                        # Use parent instead of child
            expanded.append(result)                 # No parent found — keep original
        return expanded

    def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[ParsedChunk]:
        """
        Retrieve a single chunk from Qdrant by its point ID.
        Used for parent-child lookup.
        """
        try:
            points = self._dense_store._client.retrieve(
                collection_name=self._dense_store._collection,
                ids=[DenseVectorStore._chunk_id_to_int(chunk_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            payload = points[0].payload or {}
            chunk = ParsedChunk(
                text=payload.get("text", ""),
                source_file=payload.get("source_file"),
                source_name=payload.get("source_name"),
                modality=payload.get("modality", "text"),
                metadata=payload,
            )
            chunk.chunk_id = chunk_id
            return chunk
        except Exception as exc:
            logger.warning(f"Parent fetch failed for {chunk_id}: {exc}")
            return None

    # -------------------------------------------------------------------------
    # Challenge 15: Sentence window expansion
    # -------------------------------------------------------------------------

    def _expand_sentence_window(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Expand each retrieved chunk to include surrounding sentences from the original document.
        This gives the LLM more context around a precise match.
        In production, sentence window expansion requires the original document text to be
        accessible (stored in the payload or a document store).
        """
        window = self._ret_cfg["sentence_window"]["window_size"]  # Sentences before + after
        for result in results:
            # Retrieve extended context from payload's surrounding_text field (set at index time)
            surrounding = result.chunk.metadata.get("surrounding_text")
            if surrounding:
                result.chunk.text = surrounding     # Replace with windowed text
            # If surrounding_text not stored, the chunk text is used as-is
        return results

    # -------------------------------------------------------------------------
    # Challenge 16: Re-ranking
    # -------------------------------------------------------------------------

    def _rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply cross-encoder re-ranking to the initial retrieval candidates.
        Supports Cohere, ColBERT, BGE, and LLM-based rerankers.
        """
        model_choice = self._ret_cfg["reranking"]["model"]   # From YAML config
        documents = [r.chunk.text for r in results]           # Text of each candidate

        if model_choice == "cohere":
            return self._rerank_cohere(query, documents, results)
        elif model_choice == "bge":
            return self._rerank_cross_encoder(query, documents, results)
        elif model_choice == "llm_pointwise":
            return self._rerank_llm(query, documents, results)
        else:
            logger.warning(f"Unknown reranker '{model_choice}' — skipping reranking")
            return results

    def _rerank_cohere(
        self, query: str, documents: List[str], results: List[SearchResult]
    ) -> List[SearchResult]:
        """Re-rank using the Cohere cross-encoder reranker API."""
        model = self._ret_cfg["reranking"]["cohere_model"]    # e.g. "rerank-english-v3.0"
        response = self._cohere.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=len(documents),                   # Return all documents re-scored
        )
        # Map Cohere's reranked indices back to our SearchResult list
        reranked: List[SearchResult] = []
        for i, rerank_result in enumerate(response.results):
            original = results[rerank_result.index]  # Original SearchResult at this index
            original.score = rerank_result.relevance_score  # Replace score with Cohere's
            original.rank = i + 1                   # Update rank
            reranked.append(original)
        return reranked                              # Already sorted by Cohere

    def _rerank_cross_encoder(
        self, query: str, documents: List[str], results: List[SearchResult]
    ) -> List[SearchResult]:
        """Re-rank using a local BGE cross-encoder model."""
        if self._cross_encoder is None:
            model_name = self._ret_cfg["reranking"]["bge_model"]
            self._cross_encoder = CrossEncoder(model_name)  # Lazy-load the model
        pairs = [(query, doc) for doc in documents]         # Query-document pairs for cross-encoder
        scores = self._cross_encoder.predict(pairs)         # Relevance scores for each pair
        for result, score in zip(results, scores):
            result.score = float(score)             # Assign cross-encoder score
        results.sort(key=lambda r: r.score, reverse=True)   # Sort by descending score
        for rank, result in enumerate(results, start=1):
            result.rank = rank                      # Reassign ranks after sorting
        return results

    def _rerank_llm(
        self, query: str, documents: List[str], results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Re-rank using an LLM with a listwise scoring prompt.
        Most expensive but highest quality; use only for critical queries.
        """
        model = self._ret_cfg["reranking"]["llm_rerank_model"]
        doc_list_str = "\n".join(
            f"[{i+1}] {doc[:500]}" for i, doc in enumerate(documents)  # Truncate to 500 chars each
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list_str}\n\n"
            "Rank the documents by relevance to the query. "
            "Return ONLY a comma-separated list of document numbers in order from most to least relevant. "
            "Example: 3, 1, 5, 2, 4"
        )
        response = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
        # Parse comma-separated rank list: "3, 1, 5, 2, 4"
        try:
            order = [int(x.strip()) - 1 for x in raw.split(",")]  # Convert to 0-based indices
            reranked = [results[i] for i in order if 0 <= i < len(results)]
            # Append any results not mentioned (defensive)
            mentioned = set(order)
            reranked += [r for i, r in enumerate(results) if i not in mentioned]
            for rank, result in enumerate(reranked, start=1):
                result.rank = rank
            return reranked
        except (ValueError, IndexError):
            logger.warning("LLM reranker returned unparseable output — using original order")
            return results

    # -------------------------------------------------------------------------
    # Challenge 17: Lost-in-the-middle ordering
    # -------------------------------------------------------------------------

    def _apply_position_aware_ordering(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Reorder context chunks so the most relevant are at the beginning and end,
        with the least relevant placed in the middle of the context window.

        Research shows LLMs suffer from "lost-in-the-middle" degradation where
        information in the middle of a long context is underutilised.
        """
        ordering = self._ret_cfg["context_ordering"]["strategy"]  # From YAML

        if ordering != "position_aware" or len(results) < 3:
            return results                          # No reordering needed

        # Strategy: place best at position 0, second-best at last, rest in middle
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        if not sorted_results:
            return results

        best = sorted_results[0]                    # Highest-scoring chunk → position 0
        second_best = sorted_results[1] if len(sorted_results) > 1 else None
        middle = sorted_results[2:]                 # Remaining chunks fill the middle

        ordered = [best] + middle                   # Best first
        if second_best:
            ordered.append(second_best)             # Second-best last (minimises lost-in-middle)

        logger.debug(f"Position-aware ordering applied to {len(ordered)} chunks")
        return ordered

    # -------------------------------------------------------------------------
    # Challenge 18: Context window overflow management
    # -------------------------------------------------------------------------

    def _manage_context(
        self, results: List[SearchResult], query: str
    ) -> List[ContextItem]:
        """
        Ensure the total context fits within the model's context window.
        Applies compression if necessary.
        """
        max_tokens = self._ctx_cfg["max_context_tokens"]    # Hard cap from YAML
        strategy = self._ctx_cfg["compression_model"]       # "llm_lingua" | "extractive" | "refine"

        context_items: List[ContextItem] = []
        total_tokens = 0                            # Running token count

        for rank, result in enumerate(results, start=1):
            chunk_text = result.chunk.text

            # Estimate token count (rough approximation: 4 chars ≈ 1 token)
            estimated_tokens = len(chunk_text) // 4

            if total_tokens + estimated_tokens > max_tokens:
                # Budget exceeded — apply compression to remaining chunks
                if strategy == "llm_lingua":
                    chunk_text = self._compress_llm_lingua(chunk_text)
                elif strategy == "extractive":
                    chunk_text = self._compress_extractive(chunk_text, max_sentences=3)
                # Recalculate after compression
                estimated_tokens = len(chunk_text) // 4
                if total_tokens + estimated_tokens > max_tokens:
                    logger.warning(
                        f"Context budget exhausted at rank {rank} — stopping context assembly"
                    )
                    break                           # Stop adding chunks — budget is full

            result.chunk.text = chunk_text          # Apply compressed text in-place
            total_tokens += estimated_tokens
            context_items.append(ContextItem(chunk=result.chunk, score=result.score, rank=rank))

        return context_items

    def _compress_llm_lingua(self, text: str) -> str:
        """
        Compress a text chunk using an LLM to retain key information.
        Production implementation would call LLMLingua or a similar model.
        Here we use a simple LLM prompt as a stand-in.
        """
        ratio = self._ctx_cfg["llm_lingua_ratio"]   # Target compression ratio (e.g. 0.5)
        target_length = int(len(text) * ratio)       # Approximate target character count
        prompt = (
            f"Compress the following text to approximately {target_length} characters "
            f"while preserving all key facts and information. Return ONLY the compressed text.\n\n"
            f"{text}"
        )
        response = self._openai.chat.completions.create(
            model="gpt-3.5-turbo",                  # Use cheaper model for compression
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=target_length // 4 + 50,     # Rough token estimate from chars
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _compress_extractive(text: str, max_sentences: int = 3) -> str:
        """
        Simple extractive compression: keep only the first N sentences.
        Used as a cheap fallback when LLMLingua is unavailable.
        """
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)  # Split at sentence boundaries
        return " ".join(sentences[:max_sentences])     # Return first N sentences