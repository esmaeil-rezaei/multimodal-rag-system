

from __future__ import annotations

import json
from typing import Optional

from agents import RunContextWrapper, function_tool

from src.agents.context import RAGRunContext

from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import Generator
from src.indexing.embedder import QueryEmbedder
from src.indexing.vector_store import DenseVectorStore, HybridSearchEngine, SparseIndex
from src.operations.ops_middleware import (
    AccessControlMiddleware,
    PIIGuard,
    SemanticCache,
    TraceSpan,
)
from src.query.understanding import QueryUnderstanding, ProcessedQuery
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)


@function_tool
async def get_conversation_history(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:
    return json.dumps({
        "conversation_history": ctx.context.conversation_history or []
    })



@function_tool
async def prepare_query(
    ctx: RunContextWrapper[RAGRunContext],
    query: str,
) -> str:


    if not getattr(ctx.context, "correlation_id", None):
        ctx.context.correlation_id = set_correlation_id()
    logger.info(
        "understand_query tool started",
        extra={"query": query, "correlation_id": ctx.context.correlation_id},
    )

    qu = QueryUnderstanding()
    effective_query = query or ctx.context.raw_query

    with TraceSpan("understand_query"):
        try:
            pq = qu.process(
                query=effective_query,
                conversation_history=ctx.context.conversation_history or [],
            )
            ctx.context.processed_query = pq
            ctx.context.record("understand_query_tool", f"{len(pq.sub_questions)} sub-questions")


            sub_questions = [
                sq if isinstance(sq, str) else sq.text for sq in pq.sub_questions
            ]

            if pq.hypothetical_doc:
                logger.info(
                    "HyDE document generated — dual retrieval will be used",
                    extra={"hyde_preview": pq.hypothetical_doc[:120]},
                )

            return json.dumps({
                "standalone_query": pq.standalone_query,
                "sub_questions": pq.sub_questions,
                "search_queries": [pq.standalone_query] + sub_questions,
                "requires_hyde": pq.hypothetical_doc is not None,
                "metadata_filters": pq.metadata_filters,
            })

        except Exception as exc:
            logger.error("understand_query tool failed: %s", exc)
            return json.dumps({"error": str(exc), "standalone_query": query})



@function_tool
async def retrieve_context(
    ctx: RunContextWrapper[RAGRunContext],
    auth_token: Optional[str] = None,
) -> str:


    pq = ctx.context.processed_query
    if pq is None:
        return json.dumps({"error": "processed_query not set — run understand_query first"})


    with TraceSpan("acl_auth"):
        acl = AccessControlMiddleware()
        try:
            claims = acl.authenticate(auth_token or "")
        except Exception as exc:
            logger.warning("Auth failed (defaulting to public namespace): %s", exc)
            claims = {"namespace": "default", "roles": ["public"]}
        namespace = acl.get_namespace(claims)
        ctx.context.namespace = namespace


    with TraceSpan("query_embedding"):
        try:
            embedder      = QueryEmbedder()
            dense_store   = DenseVectorStore()
            sparse_index  = SparseIndex()
            search_engine = HybridSearchEngine(dense_store, sparse_index)
            retriever     = Retriever(search_engine, dense_store)

            query_vector = embedder.embed_query(
                pq.final_query(), language=pq.language
            )

            hyde_vector = None
            if pq.hypothetical_doc:
                hyde_vector = embedder.embed_query(
                    pq.hypothetical_doc, language=pq.language
                )
                logger.info(
                    "HyDE vector generated — dual retrieval will be used",
                    extra={"hyde_preview": pq.hypothetical_doc[:120]},
                )

        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return json.dumps({"error": str(exc), "chunks_retrieved": 0})


    with TraceSpan("cache_lookup"):
        cache  = SemanticCache()
        cached = cache.get(query_vector)

    if cached:
        logger.info("Cache hit — skipping retrieval and generation")
        ctx.context.generation_result = cached
        ctx.context.record("retrieve_context_tool", "cache_hit")
        return json.dumps({
            "chunks_retrieved": 0,
            "sources": [],
            "retrieval_method": "cache",
            "cache_hit": True,
        })


    try:
        evaluator = RAGEvaluator()
        evaluator.update_reference_distribution(query_vector)
        ctx.context._evaluator = evaluator      # reused in generate_answer
    except Exception as exc:
        logger.warning("Evaluator reference update failed (non-fatal): %s", exc)
        ctx.context._evaluator = None


    with TraceSpan("retrieval", {"namespace": namespace}):
        try:
            if hyde_vector is not None:
                context_items = retriever.retrieve_dual(
                    pq=pq,
                    query_vector=query_vector,
                    hyde_vector=hyde_vector,
                    namespace=namespace,
                )
                method = "hyde_dual"
                logger.info(
                    f"Dual retrieval (query + HyDE): {len(context_items)} context items"
                )
            else:
                context_items = retriever.retrieve(
                    pq=pq,
                    query_vector=query_vector,
                    namespace=namespace,
                )
                method = "dense"

            ctx.context.context_items = context_items
            ctx.context._query_vector = query_vector   
            ctx.context.record(
                "retrieve_context_tool", f"{len(context_items)} chunks via {method}"
            )

            sources = list({
                item.chunk.source_name
                for item in context_items
                if item.chunk.source_name
            })

            return json.dumps({
                "chunks_retrieved": len(context_items),
                "sources": sources,
                "retrieval_method": method,
                "cache_hit": False,
            })

        except Exception as exc:
            logger.error("retrieve_context tool failed: %s", exc)
            return json.dumps({"error": str(exc), "chunks_retrieved": 0})



@function_tool
async def generate_answer(
    ctx: RunContextWrapper[RAGRunContext],
) -> str:


    # if getattr(ctx.context, "generation_result", None) is not None:
    #     result = ctx.context.generation_result
    #     logger.info("generate_answer: returning cached generation result")
    #     return json.dumps({
    #         "answer": result.answer,
    #         "citations": [c.get("chunk_id") for c in result.citations],
    #         "faithfulness_score": result.faithfulness_score,
    #         "has_conflict": result.has_conflict,
    #     })

    pq            = ctx.context.processed_query
    context_items = ctx.context.context_items

    if not context_items:
        return json.dumps({
            "answer": "I could not find relevant information to answer your question.",
            "citations": [],
            "faithfulness_score": None,
            "has_conflict": False,
        })


    with TraceSpan("generation"):
        try:
            generator  = Generator()
            query_text = pq.original_query if pq else ctx.context.raw_query

            result = generator.generate(
                query=query_text,
                context_items=context_items,
            )
            ctx.context.generation_result = result
            ctx.context.record(
                "generate_answer_tool", f"faithfulness={result.faithfulness_score}"
            )

        except Exception as exc:
            logger.error("generate_answer tool failed during generation: %s", exc)
            return json.dumps({
                "answer": "An error occurred while generating the answer.",
                "citations": [],
                "faithfulness_score": None,
                "has_conflict": False,
            })


    with TraceSpan("output_pii_scan"):
        try:
            pii_guard     = PIIGuard()
            result.answer = pii_guard.redact(result.answer, context="output")
        except Exception as exc:
            logger.warning("Output PII scan failed (non-fatal): %s", exc)


    with TraceSpan("cache_store"):
        query_vector = getattr(ctx.context, "_query_vector", None)
        if query_vector is not None:
            try:
                cache = SemanticCache()
                cache.put(query_vector, result)
            except Exception as exc:
                logger.warning("Cache store failed (non-fatal): %s", exc)
        else:
            logger.warning("No _query_vector in context — skipping cache store")


    try:
        evaluator     = getattr(ctx.context, "_evaluator", None) or RAGEvaluator()
        context_texts = [item.chunk.text for item in context_items]
        raw_query     = pq.original_query if pq else ctx.context.raw_query

        report = evaluator.evaluate_online(
            query=raw_query,
            answer=result.answer,
            context_texts=context_texts,
        )
        logger.info(
            "Online evaluation complete",
            extra={
                "overall_score": report.overall_score,
                "scores": report.ragas_scores or report.custom_judge_scores,
            },
        )
    except Exception as exc:
        logger.warning("Online evaluation failed (non-fatal): %s", exc)


    logger.info(
        "generate_answer tool complete",
        extra={
            "faithfulness": result.faithfulness_score,
            "citations": len(result.citations),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "has_conflict": result.has_conflict,
            "hyde_used": getattr(ctx.context.processed_query, "hypothetical_doc", None)
            is not None,
        },
    )

    return json.dumps({
        "answer": result.answer,
        "citations": [c.get("chunk_id") for c in result.citations],
        "faithfulness_score": result.faithfulness_score,
        "has_conflict": result.has_conflict,
    })