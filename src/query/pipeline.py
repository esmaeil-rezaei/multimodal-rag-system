
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from src.config.settings import get_config
from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import Generator, GenerationResult
from src.indexing.embedder import QueryEmbedder
from src.indexing.vector_store import (
    DenseVectorStore,
    SparseIndex,
    HybridSearchEngine,
)
from src.operations.ops_middleware import (
    AccessControlMiddleware,
    PIIGuard,
    SemanticCache,
    TraceSpan,
)
from src.query.understanding import QueryUnderstanding
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger, set_correlation_id


logger = get_logger(__name__)


def _log_ragas_scores(report) -> None:
    """Print RAGAS evaluation scores clearly to the console."""
    logger.info(
        f"\n{'='*50}\n"
        f"  RAGAS EVALUATION RESULTS\n"
        f"  Faithfulness:      {report.generation.faithfulness:.3f}\n"
        f"  Answer Relevancy:  {report.generation.answer_relevancy:.3f}\n"
        f"  Context Precision: {report.retrieval.context_precision:.3f}\n"
        f"  Context Recall:    {report.retrieval.context_recall:.3f}\n"
        f"  Overall Score:     {report.overall_score:.3f}\n"
        f"{'='*50}"
    )


class QueryPipeline:
    """
    Full RAG query pipeline.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg

        self._query_understanding = QueryUnderstanding()
        dense_store               = DenseVectorStore()
        sparse_index              = SparseIndex()
        self._search_engine       = HybridSearchEngine(dense_store, sparse_index)
        self._dense_store         = dense_store
        self._query_embedder      = QueryEmbedder()
        self._retriever           = Retriever(self._search_engine, dense_store)
        self._generator           = Generator()
        self._cache               = SemanticCache()
        self._acl                 = AccessControlMiddleware()
        self._pii_guard           = PIIGuard()
        self._evaluator           = RAGEvaluator()



    def query(
        self,
        raw_query: str,
        auth_token: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        Runs the full RAG pipeline for a user query.

        Includes:
        - query understanding (+ optional HyDE generation)
        - hybrid retrieval (dense + sparse)
        - optional dual retrieval if HyDE is present
        - generation over retrieved context
        - caching, PII filtering, and evaluation

        Returns:
            GenerationResult with answer, citations, and metadata.
        """
        
        correlation_id = set_correlation_id()
        logger.info(
            "Query pipeline started",
            extra={"query": raw_query, "correlation_id": correlation_id},
        )

        # ACL
        with TraceSpan("acl_auth"):
            try:
                claims = self._acl.authenticate(auth_token or "")
            except Exception as exc:
                logger.warning(f"Auth failed: {exc}")
                claims = {"namespace": "default", "roles": ["public"]}
            namespace = self._acl.get_namespace(claims)


        with TraceSpan("query_understanding"):
            processed_query = self._query_understanding.process(
                query=raw_query,
                conversation_history=conversation_history,
            )


        with TraceSpan("query_embedding"):
            query_vector = self._query_embedder.embed_query(
                processed_query.final_query(),
                language=processed_query.language,
            )

            hyde_vector: Optional[np.ndarray] = None
            if processed_query.hypothetical_doc:
                hyde_vector = self._query_embedder.embed_query(
                    processed_query.hypothetical_doc,
                    language=processed_query.language,
                )
                logger.info(
                    "HyDE vector generated — dual retrieval will be used",
                    extra={"hyde_preview": processed_query.hypothetical_doc[:120]},
                )


        with TraceSpan("cache_lookup"):
            cached = self._cache.get(query_vector)
        if cached:
            logger.info("Returning cached response")
            return cached


        self._evaluator.update_reference_distribution(query_vector)


        with TraceSpan("retrieval", {"namespace": namespace}):
            if hyde_vector is not None:
                context_items = self._retriever.retrieve_dual(
                    pq=processed_query,
                    query_vector=query_vector,
                    hyde_vector=hyde_vector,
                    namespace=namespace,
                )
                logger.info(
                    f"Dual retrieval (query + HyDE): {len(context_items)} context items"
                )
            else:
                context_items = self._retriever.retrieve(
                    pq=processed_query,
                    query_vector=query_vector,
                    namespace=namespace,
                )


        with TraceSpan("generation"):
            result = self._generator.generate(
                query=processed_query.final_query(),
                context_items=context_items,
            )


        with TraceSpan("output_pii_scan"):
            result.answer = self._pii_guard.redact(result.answer, context="output")


        with TraceSpan("cache_store"):
            self._cache.put(query_vector, result)


        try:
            context_texts = [item.chunk.text for item in context_items]
            report = self._evaluator.evaluate_with_llm_judge(
                query=raw_query,
                answer=result.answer,
                context_texts=context_texts,
                generation_result=result,
            )
            _log_ragas_scores(report)
            logger.info(
                "Online evaluation",
                extra={
                    "overall_score": report.overall_score,
                    "ragas_scores":  report.ragas_scores,
                },
            )
        except Exception as exc:
            logger.warning(f"Online evaluation failed (non-fatal): {exc}")

        logger.info(
            "Query pipeline complete",
            extra={
                "faithfulness":        result.faithfulness_score,
                "citations":           len(result.citations),
                "prompt_tokens":       result.prompt_tokens,
                "completion_tokens":   result.completion_tokens,
                "hyde_used":           hyde_vector is not None,
            },
        )
        return result