# """
# DEACTIVATED BECAUSE OF USING AGENTIC AI
# """

# from __future__ import annotations

# import numpy as np

# from src.config.settings import get_config
# from src.evaluation.evaluator import RAGEvaluator
# from src.generation.generator import GenerationResult, Generator
# from src.indexing.embedder import QueryEmbedder
# from src.indexing.vector_store import (
#     DenseVectorStore,
#     HybridSearchEngine,
#     SparseIndex,
# )
# from src.operations.ops_middleware import (
#     AccessControlMiddleware,
#     PIIGuard,
#     SemanticCache,
#     TraceSpan,
# )
# from src.query.understanding import QueryUnderstanding
# from src.retrieval.retriever import Retriever
# from src.utils.logger import get_logger, set_correlation_id

# logger = get_logger(__name__)


# class QueryPipeline:
#     """
#     Full RAG query pipeline.
#     """

#     def __init__(self) -> None:
#         cfg = get_config()
#         self._cfg = cfg

#         self._query_understanding = QueryUnderstanding()
#         dense_store = DenseVectorStore()
#         sparse_index = SparseIndex()
#         self._search_engine = HybridSearchEngine(dense_store, sparse_index)
#         self._dense_store = dense_store
#         self._query_embedder = QueryEmbedder()
#         self._retriever = Retriever(self._search_engine, dense_store)
#         self._generator = Generator()
#         self._cache = SemanticCache()
#         self._acl = AccessControlMiddleware()
#         self._pii_guard = PIIGuard()
#         self._evaluator = RAGEvaluator()

#     def query(
#         self,
#         raw_query: str,
#         auth_token: str | None = None,
#         conversation_history: list[dict[str, str]] | None = None,
#     ) -> GenerationResult:
#         """
#         Runs the full RAG pipeline for a user query.

#         Includes:
#         - query understanding (+ optional HyDE generation)
#         - hybrid retrieval (dense + sparse)
#         - optional dual retrieval if HyDE is present
#         - generation over retrieved context
#         - caching, PII filtering, and evaluation

#         Returns:
#             GenerationResult with answer, citations, and metadata.
#         """

#         correlation_id = set_correlation_id()
#         logger.info(
#             "Query pipeline started",
#             extra={"query": raw_query, "correlation_id": correlation_id},
#         )

#         # ACL
#         with TraceSpan("acl_auth"):
#             try:
#                 claims = self._acl.authenticate(auth_token or "")
#             except Exception as exc:
#                 logger.warning(f"Auth failed: {exc}")
#                 claims = {"namespace": "default", "roles": ["public"]}
#             namespace = self._acl.get_namespace(claims)

#         with TraceSpan("query_understanding"):
#             processed_query = self._query_understanding.process(
#                 query=raw_query,
#                 conversation_history=conversation_history,
#             )

#         with TraceSpan("query_embedding"):
#             query_vector = self._query_embedder.embed_query(
#                 processed_query.final_query(),
#                 language=processed_query.language,
#             )

#             hyde_vector: np.ndarray | None = None
#             if processed_query.hypothetical_doc:
#                 hyde_vector = self._query_embedder.embed_query(
#                     processed_query.hypothetical_doc,
#                     language=processed_query.language,
#                 )
#                 logger.info(
#                     "HyDE vector generated — dual retrieval will be used",
#                     extra={"hyde_preview": processed_query.hypothetical_doc[:120]},
#                 )

#         with TraceSpan("cache_lookup"):
#             cached = self._cache.get(query_vector)
#         if cached:
#             logger.info("Returning cached response")
#             return cached

#         self._evaluator.update_reference_distribution(query_vector)

#         with TraceSpan("retrieval", {"namespace": namespace}):
#             if hyde_vector is not None:
#                 context_items = self._retriever.retrieve_dual(
#                     pq=processed_query,
#                     query_vector=query_vector,
#                     hyde_vector=hyde_vector,
#                     namespace=namespace,
#                 )
#                 logger.info(f"Dual retrieval (query + HyDE): {len(context_items)} context items")
#             else:
#                 context_items = self._retriever.retrieve(
#                     pq=processed_query,
#                     query_vector=query_vector,
#                     namespace=namespace,
#                 )

#         with TraceSpan("generation"):
#             result = self._generator.generate(
#                 query=processed_query.original_query,
#                 context_items=context_items,
#             )

#         with TraceSpan("output_pii_scan"):
#             result.answer = self._pii_guard.redact(result.answer, context="output")

#         with TraceSpan("cache_store"):
#             self._cache.put(query_vector, result)

#         try:
#             context_texts = [item.chunk.text for item in context_items]
#             report = self._evaluator.evaluate_online(
#                 query=raw_query,
#                 answer=result.answer,
#                 context_texts=context_texts,
#             )
#             logger.info(
#                 "Online evaluation complete",
#                 extra={
#                     "overall_score": report.overall_score,
#                     "scores": report.ragas_scores or report.custom_judge_scores,
#                 },
#             )
#         except Exception as exc:
#             logger.warning("Online evaluation failed (non-fatal): %s", exc)

#         logger.info(
#             "Query pipeline complete",
#             extra={
#                 "faithfulness": result.faithfulness_score,
#                 "citations": len(result.citations),
#                 "prompt_tokens": result.prompt_tokens,
#                 "completion_tokens": result.completion_tokens,
#                 "hyde_used": hyde_vector is not None,
#             },
#         )
#         return result


# -------------------------------------------

# import argparse

# def run_single_query(pipeline: QueryPipeline, question: str) -> None:
#     """Run a single query and print the result."""

#     print(f"\nQuery: {question}")
#     print("─" * 60)

#     result = pipeline.query(raw_query=question)

#     print(f"\nAnswer:\n{result.answer}")

#     if result.citations:
#         print(f"\nCitations ({len(result.citations)}):")
#         for cite in result.citations:
#             src = cite.get("source_name") or cite.get("source_file") or "unknown"
#             ts = (cite.get("ingestion_ts") or "")[:10]
#             print(f"  • [{cite['chunk_id'][:8]}…] {src} ({ts})")

#     if result.faithfulness_score is not None:
#         print(f"\nFaithfulness: {result.faithfulness_score:.2f}")

#     if result.has_conflict:
#         print(f"\nConflict: {result.conflict_resolution}")

#     print()


# def interactive_repl(pipeline: QueryPipeline) -> None:
#     """Interactive query mode (REPL)."""

#     history = []

#     print("\nRAG Interactive Mode (type 'exit' to quit)\n")

#     while True:
#         try:
#             question = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nGoodbye!")
#             break

#         if not question:
#             continue
#         if question.lower() in {"exit", "quit", "q"}:
#             print("Goodbye!")
#             break

#         result = pipeline.query(
#             raw_query=question,
#             conversation_history=history,
#         )

#         print(f"\nAssistant: {result.answer}\n")

#         history.append({"role": "user", "content": question})
#         history.append({"role": "assistant", "content": result.answer})


# def main() -> None:
#     parser = argparse.ArgumentParser(description="RAG query CLI")

#     parser.add_argument(
#         "question",
#         nargs="?",
#         help="Run a single query (omit for interactive mode)",
#     )
#     parser.add_argument(
#         "--interactive",
#         action="store_true",
#         help="Start interactive mode",
#     )

#     args = parser.parse_args()

#     pipeline = QueryPipeline()

#     if args.interactive or not args.question:
#         interactive_repl(pipeline)
#     else:
#         run_single_query(pipeline, args.question)


# if __name__ == "__main__":
#     main()
