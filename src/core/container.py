from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.agents.orchestrator import RAGOrchestrator
from src.config.settings import get_config
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.generation.generator import Generator
from src.indexing.embedder import EmbeddingRouter, QueryEmbedder
from src.indexing.vector_store import DenseVectorStore, HybridSearchEngine, SparseIndex
from src.operations.ops_middleware import AccessControlMiddleware, PIIGuard
from src.query.understanding import QueryUnderstanding
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

try:
    from src.graphrag.graph_retriever import GraphRetriever
    from src.graphrag.neo4j_store import Neo4jGraphStore

    _GRAPHRAG_AVAILABLE = True
except ImportError:
    Neo4jGraphStore = None
    GraphRetriever = None
    _GRAPHRAG_AVAILABLE = False

logger = get_logger(__name__)

_SLOW_THRESHOLD = 2.0  # seconds — steps above this log as WARNING


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed > _SLOW_THRESHOLD:
        logger.warning("SLOW init — %s took %.2fs (threshold %.1fs)", label, elapsed, _SLOW_THRESHOLD)
    else:
        logger.info("%s ready — %.2fs", label, elapsed)


@dataclass
class AppContainer:
    """Holds every application-level singleton."""

    embedder: object = field(default=None, repr=False)
    dense_store: object = field(default=None, repr=False)
    sparse_index: object = field(default=None, repr=False)
    search_engine: object = field(default=None, repr=False)
    retriever: object = field(default=None, repr=False)
    generator: object = field(default=None, repr=False)
    evaluator: object = field(default=None, repr=False)
    pii_guard: object = field(default=None, repr=False)
    acl: object = field(default=None, repr=False)
    query_understanding: object = field(default=None, repr=False)
    retrieval_evaluator: object = field(default=None, repr=False)
    graph_retriever: object | None = field(default=None, repr=False)
    orchestrator: object = field(default=None, repr=False)

    _started: bool = False

    def startup(self) -> None:
        """
        Initialise every singleton in dependency order.
        Must be called exactly once, inside the FastAPI lifespan.
        """
        if self._started:
            return

        t_total = time.perf_counter()
        logger.info("AppContainer.startup — initialising retrieval stack")

        with _timed("ACL + PIIGuard"):
            self.acl = AccessControlMiddleware()
            self.pii_guard = PIIGuard()

        with _timed("EmbeddingRouter + QueryEmbedder"):
            router = EmbeddingRouter()
            self.embedder = QueryEmbedder(router=router)

        with _timed("DenseVectorStore + SparseIndex + HybridSearchEngine"):
            self.dense_store = DenseVectorStore()
            self.sparse_index = SparseIndex()
            self.search_engine = HybridSearchEngine(self.dense_store, self.sparse_index)
            logger.info(
                "vector stores — dense=%s sparse_available=%s",
                type(self.dense_store).__name__,
                getattr(self.sparse_index, "_available", False),
            )

        with _timed("Retriever"):
            self.retriever = Retriever(self.search_engine, self.dense_store)

        with _timed("Generator + RAGEvaluator + RetrievalEvaluator"):
            self.generator = Generator()
            self.evaluator = RAGEvaluator()
            self.retrieval_evaluator = RetrievalEvaluator(
                embedder=self.embedder, retriever=self.retriever
            )

        with _timed("QueryUnderstanding"):
            self.query_understanding = QueryUnderstanding()

        with _timed("GraphRetriever"):
            self.graph_retriever = self._try_init_graph_retriever()

        with _timed("RAGOrchestrator"):
            self.orchestrator = RAGOrchestrator(container=self)

        self._started = True
        logger.info("AppContainer.startup complete — total %.2fs", time.perf_counter() - t_total)

    def shutdown(self) -> None:
        """Close connections gracefully."""
        if not self._started:
            return
        logger.info("AppContainer.shutdown — releasing resources")
        try:
            if self.dense_store and hasattr(self.dense_store, "_client"):
                self.dense_store._client.close()
        except Exception as exc:
            logger.warning("DenseVectorStore close error (non-fatal): %s", exc)
        self._started = False

    def _try_init_graph_retriever(self) -> object | None:
        cfg = get_config()
        gr_cfg = cfg.graphrag
        if not gr_cfg.get("enabled", False):
            logger.info("GraphRAG disabled — skipping GraphRetriever init")
            return None
        if not _GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG enabled in config but dependencies not installed.")
            return None
        try:
            with _timed("Neo4jGraphStore connect"):
                graph_store = Neo4jGraphStore()
            with _timed("GraphRetriever build"):
                gr = GraphRetriever(
                    graph_store=graph_store,
                    vector_retriever=self.retriever,
                    search_engine=self.search_engine,
                    dense_store=self.dense_store,
                )
            return gr
        except Exception as exc:
            logger.error("GraphRetriever init failed (non-fatal): %s", exc)
            return None


_container: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the application container. Raises if startup() was not called."""
    if _container is None or not _container._started:
        raise RuntimeError(
            "AppContainer not initialised. "
            "Call src.core.container.init_container() inside the FastAPI lifespan."
        )
    return _container


def init_container() -> AppContainer:
    """Create and start the container. Called once from app lifespan."""
    global _container
    _container = AppContainer()
    _container.startup()
    return _container
