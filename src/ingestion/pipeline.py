from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from src.config.settings import get_config
from src.indexing.embedder import EmbeddingRouter, QueryEmbedder
from src.indexing.vector_store import DenseVectorStore, SparseIndex
from src.ingestion.chunker import TextChunker, ChunkNode
from src.ingestion.consolidator import ChunkConsolidator
from src.ingestion.deduplicator import Deduplicator
from src.ingestion.parser import DocumentParser, ParsedChunk
from src.operations.ops_middleware import PIIGuard
from src.utils.logger import get_logger, set_correlation_id

try:
    from src.graphrag.extractor import GraphExtractor
    from src.graphrag.neo4j_store import Neo4jGraphStore
    from src.graphrag.community import CommunityDetector
    _GRAPHRAG_AVAILABLE = True
except ImportError as _graphrag_import_err:
    _GRAPHRAG_AVAILABLE = False
    GraphExtractor = None  # type: ignore
    Neo4jGraphStore = None  # type: ignore
    CommunityDetector = None  # type: ignore

logger = get_logger(__name__)

class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    scan → parse → stamp → PII redact → fingerprint
           → consolidate → dedup → chunk → embed → upsert
    """

    def __init__(self) -> None:
        self._cfg           = get_config()
        self._kb_cfg        = self._cfg.knowledge_base
        self._ingest_cfg    = self._cfg.ingestion
        self._versioning_cfg = self._ingest_cfg["versioning"]

        self._parser        = DocumentParser()
        self._pii_guard     = PIIGuard()
        self._consolidator  = ChunkConsolidator()
        self._deduplicator  = Deduplicator()
        self._chunker       = TextChunker()
        self._embedder      = EmbeddingRouter()
        self._dense_store   = DenseVectorStore()
        self._sparse_index  = SparseIndex()

        self._processed_hashes: Dict[str, str] = {}

        self._graph_enabled: bool = False
        self._graph_extractor = None
        self._graph_store = None

        gr_cfg = self._cfg.get("graphrag", {})
        if gr_cfg.get("enabled", False):
            if not _GRAPHRAG_AVAILABLE:
                logger.warning(
                    "graphrag.enabled=true in config but GraphRAG dependencies "
                    "are not installed.  Run: pip install neo4j python-louvain cdlib"
                )
            else:
                try:
                    self._graph_store     = Neo4jGraphStore()
                    self._graph_extractor = GraphExtractor()
                    self._graph_enabled   = True
                    logger.info("GraphRAG pipeline enabled (Neo4j + LLM extraction)")
                except Exception as exc:
                    logger.error(
                        "GraphRAG initialisation failed — continuing without graph layer: %s",
                        exc,
                    )

    def run(self, namespace: Optional[str] = None) -> Dict[str, int]:
        """
        Run a full ingestion pass over the knowledge base directory.

        When ``namespace`` is given the scan is restricted to
        ``{root_dir}/{namespace}/`` and extracted images are stored under
        ``artifacts/extracted_images/{namespace}/``, keeping each
        namespace's assets isolated.
        """

        set_correlation_id()
        root = Path(self._kb_cfg["root_dir"])

        if namespace:
            root = root / namespace

        if not root.exists():
            raise FileNotFoundError(f"Knowledge base root not found: {root}")

        # Redirect image output to a namespace-prefixed directory so that
        # artifacts from different namespaces never collide.  We temporarily
        # mutate the shared config dict (safe for single-process CLI runs) and
        # restore it in the finally block.
        parsing_cfg = self._ingest_cfg["parsing"]
        original_img_dir = parsing_cfg["image_output_dir"]
        if namespace:
            parsing_cfg["image_output_dir"] = f"{original_img_dir}/{namespace}"

        supported_ext: Set[str] = set(self._kb_cfg["supported_extensions"])
        files = [
            f for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_ext
        ]
        logger.info(f"Ingestion started: {len(files)} files discovered in {root}")

        stats = {"files_scanned": 0, "chunks_indexed": 0, "chunks_skipped": 0}

        try:
            for file_path in files:
                source_name = file_path.parent.name
                try:
                    indexed, skipped = self._ingest_file(file_path, source_name, namespace)
                    stats["files_scanned"] += 1
                    stats["chunks_indexed"] += indexed
                    stats["chunks_skipped"] += skipped
                except Exception as exc:
                    logger.error(f"Failed to ingest {file_path}: {exc}", exc_info=True)
        finally:
            # Restore original image dir so repeated calls in same process work correctly
            parsing_cfg["image_output_dir"] = original_img_dir

        logger.info("Ingestion complete", extra=stats)
        return stats

    def _ingest_file(
        self,
        file_path: Path,
        source_name: str,
        namespace: Optional[str],
    ) -> tuple[int, int]:
        """
        Process one file through the corrected pipeline stage order.

        Stage order (with rationale for each position):
          1. parse       — extract raw elements (per-element granularity)
          2. stamp       — attach ingestion_ts and doc_version before anything reads them
          3. PII redact  — mutate text BEFORE computing fingerprints
          4. fingerprint — compute chunk_id on the final clean text
          5. consolidate — group elements into sections
          6. dedup       — compare section-sized chunks, not sentence fragments
          7. chunk       — split sections; pass tables/images atomically
          8. embed+upsert
        """

        if self._versioning_cfg["delta_ingestion"]:
            file_hash = self._hash_file(file_path)
            if self._processed_hashes.get(str(file_path)) == file_hash:
                logger.debug(f"Delta skip (unchanged): {file_path.name}")
                return 0, 0
            self._processed_hashes[str(file_path)] = file_hash

        logger.info(f"Ingesting: {file_path.name} [{source_name}]")

        raw_chunks: List[ParsedChunk] = self._parser.parse_file(file_path, source_name)
        if not raw_chunks:
            logger.warning(f"No content extracted from {file_path.name}")
            return 0, 0

        ingestion_ts = datetime.now(timezone.utc).isoformat()
        doc_version  = self._extract_version(file_path)
        for chunk in raw_chunks:
            chunk.ingestion_ts = ingestion_ts
            chunk.doc_version  = doc_version

        for chunk in raw_chunks:
            original_text = chunk.text
            chunk.text = self._pii_guard.redact(chunk.text, context="ingestion")
            chunk.metadata["sensitivity"] = (
                "readacted" if chunk.text != original_text else "public"
            )

        for chunk in raw_chunks:
            chunk.chunk_id = chunk.compute_fingerprint()

        consolidated = self._consolidator.consolidate(raw_chunks)
        if not consolidated:
            logger.warning(f"Consolidation produced no chunks for {file_path.name}")
            return 0, 0

        unique_chunks = self._deduplicator.filter(consolidated)
        skipped = len(consolidated) - len(unique_chunks)

        chunk_nodes: List[ChunkNode] = self._chunker.chunk(unique_chunks)

        indexed = 0
        pairs = self._embedder.embed_nodes(chunk_nodes)

        for node, vector in pairs:
            chunk = node.chunk
            chunk.metadata["parent_id"]        = node.parent_id
            chunk.metadata["hierarchy_level"]  = node.level
            chunk.metadata["namespace"]        = namespace or "default"

            self._dense_store.upsert(chunk, vector, namespace=namespace)
            self._sparse_index.index_chunk(chunk)
            indexed += 1

        if self._graph_enabled and self._graph_extractor and self._graph_store:
            self._run_graph_extraction(
                [node.chunk for node, _ in pairs], file_path.name
            )

        logger.info(
            f"File ingested: {file_path.name} → "
            f"{indexed} chunks indexed, {skipped} skipped"
        )
        return indexed, skipped

    def _run_graph_extraction(
        self,
        chunks: "List[ParsedChunk]",
        file_name: str,
    ) -> None:
        """
        Run LLM-based entity/relationship extraction for all chunks of one file
        and persist the results to Neo4j.

        Uses asyncio.run() so it integrates cleanly with the synchronous
        ingestion pipeline without requiring the caller to be async.

        Extraction is batched (concurrency controlled inside GraphExtractor).
        After upsert, embeddings for new entity nodes are generated and stored.
        """
        try:
            text_chunks = [c for c in chunks if c.modality == "text" and c.text]
            if not text_chunks:
                return

            extraction_results = asyncio.run(
                self._graph_extractor.batch_extract(text_chunks)
            )
            entities_written = 0
            relationships_written = 0
            for result in extraction_results:
                self._graph_store.upsert_extraction(result)
                entities_written += len(result.entities)
                relationships_written += len(result.relationships)

            # Embed entity nodes that have no embedding yet.
            # We re-use the existing embedding infrastructure for consistency.
            self._embed_new_entities()

            logger.info(
                "GraphRAG extraction complete for %s: "
                "%d entities, %d relationships written to Neo4j",
                file_name,
                entities_written,
                relationships_written,
            )
        except Exception as exc:
            logger.error(
                "GraphRAG extraction failed for %s (non-fatal): %s",
                file_name, exc, exc_info=True,
            )

    def _embed_new_entities(self) -> None:
        """
        Fetch entity nodes that have no embedding and generate embeddings for them.

        Uses a Cypher query to find un-embedded entities, embeds their name +
        description in batches, and bulk-writes back to Neo4j.
        """
        try:
            with self._graph_store._session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NULL
                    RETURN e.node_id AS node_id, e.name AS name,
                           e.description AS description
                    LIMIT 500
                    """
                )
                rows = [dict(r) for r in result]

            if not rows:
                return

            texts = [
                f"{r['name']}: {r['description']}" if r.get("description") else r["name"]
                for r in rows
            ]
            emb = QueryEmbedder()
            pairs = []
            for row, text in zip(rows, texts):
                try:
                    vec = emb.embed_query(text)
                    pairs.append((row["node_id"], vec.tolist()))
                except Exception as exc:
                    logger.debug("Embedding failed for entity %s: %s", row["node_id"][:12], exc)

            if pairs:
                self._graph_store.batch_update_entity_embeddings(pairs)
                logger.debug("Embedded %d new entity nodes", len(pairs))

        except Exception as exc:
            logger.warning("Entity embedding step failed (non-fatal): %s", exc)

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """
        SHA-256 of file byte content.
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                sha.update(block)
        return sha.hexdigest()

    @staticmethod
    def _extract_version(file_path: Path) -> Optional[str]:
        """
        Extract a version string from the file name.
        """
        name = file_path.stem
        version_match = re.search(r"v(\d+(?:\.\d+)+)", name, re.IGNORECASE)
        if version_match:
            return version_match.group(0)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if date_match:
            return date_match.group(1)
        return None
