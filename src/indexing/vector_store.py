

# src/indexing/vector_store.py
# =============================================================================
# Stage 2 — Vector Index & Hybrid Search.
# Covers:
#   — Vector index scalability (HNSW, IVF-PQ, sharding)
#   — Sparse + dense hybrid indexing (BM25 + dense, RRF fusion)
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np    
import random                              

from qdrant_client import QdrantClient             
from qdrant_client.http import models as qdrant_models  
from elasticsearch import Elasticsearch            

from src.config.settings import get_config, get_secrets  
from src.ingestion.chunker import ChunkNode          
from src.ingestion.parser import ParsedChunk        
from src.utils.logger import get_logger             

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search result data class
# ---------------------------------------------------------------------------

class SearchResult:
    """Returned by every search method — wraps a chunk with its retrieval score."""

    def __init__(
        self,
        chunk: ParsedChunk,
        score: float,
        retrieval_method: str = "hybrid",           # "dense" | "sparse" | "hybrid"
        rank: int = 0,                              # Rank position in the result list
    ) -> None:
        self.chunk = chunk                          # The retrieved ParsedChunk
        self.score = score                          # Similarity or BM25 score
        self.retrieval_method = retrieval_method    # How this result was found
        self.rank = rank                            # Position in final ranked list


# ---------------------------------------------------------------------------
# Dense vector store (Qdrant)
# ---------------------------------------------------------------------------

class DenseVectorStore:
    """
    Wraps Qdrant for dense vector indexing and ANN search.
    Configures HNSW for high-recall or IVF-PQ for memory-efficient corpora.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._vs_cfg = cfg.vector_store            
        self._collection = self._vs_cfg["collection_name"]

        # Connect to Qdrant: use local embedded storage when URL is localhost,
        # remote cloud otherwise. Embedded mode needs no server or Docker.
        if "localhost" in sec.qdrant_url or "127.0.0.1" in sec.qdrant_url:
            local_path = str("qdrant_storage")
            self._client = QdrantClient(path=local_path)
            logger.info(f"Qdrant running in local embedded mode → {local_path}")
        else:
            self._client = QdrantClient(
                url=sec.qdrant_url,
                api_key=sec.qdrant_api_key,
                timeout=30,
            )
        self._ensure_collection()

    # -------------------------------------------------------------------------
    # Collection management
    # -------------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with HNSW parameters if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            logger.debug(f"Qdrant collection '{self._collection}' already exists — skipping creation")
            return

        hnsw_cfg = self._vs_cfg["hnsw"]           
        dim = get_config().embeddings["embedding_dimensions"]

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=dim,                         
                distance=qdrant_models.Distance.COSINE,  # Cosine similarity (matches L2-normalised vectors)
            ),
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=hnsw_cfg["m"],                         # HNSW M parameter — links per node (16 is standard)
                ef_construct=hnsw_cfg["ef_construct"],   # Build-time candidate list size (higher = better quality)
                full_scan_threshold=10_000,              # Below this count, use brute force (exact) instead of ANN
            ),
            # Payload index on metadata fields used for ACL filtering
            on_disk_payload=True,                        # Store payload on disk to reduce RAM usage
        )
        logger.info(f"Created Qdrant collection '{self._collection}' with HNSW(m={hnsw_cfg['m']})")

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def upsert(
        self, chunk: ParsedChunk, vector: np.ndarray, namespace: Optional[str] = None
    ) -> None:
        """
        Insert or update a single chunk-vector pair in Qdrant.

        Args:
            chunk:     The ParsedChunk to store as payload.
            vector:    Dense embedding vector (1D numpy array).
            namespace: Tenant namespace for ACL isolation.
        """
        payload = {
            "text": chunk.text,                     
            "source_file": chunk.source_file,
            "source_name": chunk.source_name,
            "modality": chunk.modality,
            "language": chunk.language,
            "doc_version": chunk.doc_version,
            "ingestion_ts": chunk.ingestion_ts,
            "namespace": namespace or "default",    
            "allowed_roles": chunk.metadata.get("allowed_roles", ["public"]),  # ACL list
            **chunk.metadata,                       # Spread all other metadata fields
        }

        payload = self._sanitize_payload(payload)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                qdrant_models.PointStruct(
                    id=self._chunk_id_to_int(chunk.chunk_id),  # Qdrant requires integer or UUID IDs
                    vector=vector.tolist(),        
                    payload=payload,             
                )
            ],
        )

    @staticmethod
    def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively strip any value that Qdrant cannot serialise to JSON.

        Qdrant payloads support only: str, int, float, bool, None, list, dict.
        Unstructured.io attaches objects like PixelSpace, CoordinatesMetadata,
        and numpy scalars to element metadata.  Any value whose type is not in
        the allowed set is converted to its string representation so the data
        is not silently lost but the upsert does not fail.
        """
        _ALLOWED = (str, int, float, bool, type(None))

        def _clean(v: Any) -> Any:
            if isinstance(v, _ALLOWED):
                return v
            if isinstance(v, dict):
                return {str(k): _clean(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(i) for i in v]
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            # Unknown type (PixelSpace, CoordinatesMetadata, etc.) — stringify it
            return str(v)

        return {str(k): _clean(v) for k, v in payload.items()}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        ANN search in Qdrant.

        Args:
            query_vector:    Dense query embedding.
            top_k:           Number of results to return.
            namespace:       Restrict results to this tenant namespace.
            metadata_filter: Additional metadata equality filters (e.g. date range).
            ef_search:       HNSW ef parameter for this query (overrides config default).
        """

        must_conditions = []

        if namespace and namespace != "default":

            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="namespace",
                    match=qdrant_models.MatchValue(value=namespace),
                )
            )

        query_filter = (
            qdrant_models.Filter(must=must_conditions) if must_conditions else None
        )

        search_params = qdrant_models.SearchParams(
            hnsw_ef=ef_search or self._vs_cfg["hnsw"]["ef_search"],  # Higher = better recall, slower
            exact=False,                            # Use ANN (not brute force)
        )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),            
            limit=top_k,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,                      # Return payload (text + metadata)
            with_vectors=False,                     # Don't return raw vectors (saves bandwidth)
        )

        search_results = []

        for hit in response.points:
            payload = hit.payload or {}
            chunk = ParsedChunk(
                text=payload.get("text", ""),
                source_file=payload.get("source_file"),
                source_name=payload.get("source_name"),
                modality=payload.get("modality", "text"),
                language=payload.get("language"),
                doc_version=payload.get("doc_version"),
                ingestion_ts=payload.get("ingestion_ts"),
                metadata=payload,                   # Store full payload as metadata
            )
            chunk.chunk_id = str(hit.id)
            search_results.append(
                SearchResult(chunk=chunk, score=hit.score, retrieval_method="dense")
            )

        return search_results

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _chunk_id_to_int(chunk_id: Optional[str]) -> int:
        """
        Convert a hex SHA-256 chunk_id to an integer for use as a Qdrant point ID.
        Takes the first 8 hex characters (32-bit) to keep IDs manageable.
        """
        if not chunk_id:
            return random.getrandbits(32)
        return int(chunk_id[:8], 16)                # First 8 hex chars → 32-bit integer


# ---------------------------------------------------------------------------
# Sparse BM25 index (Elasticsearch)
# ---------------------------------------------------------------------------

class SparseIndex:
    """
    Wraps Elasticsearch for BM25 keyword-based (sparse) search.
    Used in parallel with the dense index for hybrid retrieval.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._vs_cfg = cfg.vector_store
        self._index_name = self._vs_cfg["hybrid_search"]["sparse_index_name"]  # ES index name
        self._available = False

        # Connect to Elasticsearch
        es_kwargs: Dict[str, Any] = {"hosts": [sec.elasticsearch_url]}
        if sec.elasticsearch_api_key:
            es_kwargs["api_key"] = sec.elasticsearch_api_key 
        self._es = Elasticsearch(**es_kwargs)

        try:
            self._ensure_index()
            self._available = True
        except Exception as exc:
            logger.warning(
                f"Elasticsearch unavailable ({exc}). "
                "Sparse BM25 index disabled — hybrid search will use dense-only retrieval."
            )

    def _ensure_index(self) -> None:
        """Create the Elasticsearch index with optimised BM25 settings if absent."""
        if self._es.indices.exists(index=self._index_name):
            return                                 
        self._es.indices.create(
            index=self._index_name,
            body={
                "settings": {
                    "number_of_shards": 2,          # Parallelise across 2 shards for throughput
                    "number_of_replicas": 1,        # 1 replica for availability
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "k1": 1.2,              # BM25 term frequency saturation parameter
                            "b": 0.75,              # BM25 document length normalisation factor
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "similarity": "custom_bm25",  # Use tuned BM25 for this field
                        },
                        "chunk_id": {"type": "keyword"},        # Exact match for joins
                        "source_name": {"type": "keyword"},     # Facet filtering by source
                        "namespace": {"type": "keyword"},       # Tenant filtering
                        "ingestion_ts": {"type": "date"},       # Range filtering for freshness
                    }
                },
            },
        )
        logger.info(f"Created Elasticsearch index '{self._index_name}'")

    def index_chunk(self, chunk: ParsedChunk) -> None:
        """Index a single chunk into Elasticsearch for BM25 retrieval."""
        if not self._available:
            return
        doc = {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "source_name": chunk.source_name,
            "namespace": chunk.metadata.get("namespace", "default"),
            "ingestion_ts": chunk.ingestion_ts,
            **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
        }
        self._es.index(index=self._index_name, id=chunk.chunk_id, document=doc)

    def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        BM25 keyword search using Elasticsearch.
        Optionally filtered to a tenant namespace.
        """
        if not self._available:
            return []
        body: Dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query":     query,
                                    "operator":  "or",
                                    "fuzziness": "AUTO",   # AUTO: edit distance 1 for words ≤5 chars, 2 for longer
                                    "prefix_length": 1,    # First character must match — avoids noise
                                    "max_expansions": 50,  # Max fuzzy variants to consider per term
                                }
                            }
                        }  # BM25 fuzzy match — tolerates typos like "esmil" → "esmaeil"
                    ],
                    "filter": (
                        [{"term": {"namespace": namespace}}] if namespace else []
                    ),
                }
            },
        }
        response = self._es.search(index=self._index_name, body=body)
        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            chunk = ParsedChunk(
                text=src.get("text", ""),
                chunk_id=src.get("chunk_id"),
                source_file=src.get("source_file"),
                source_name=src.get("source_name"),
                ingestion_ts=src.get("ingestion_ts"),
                metadata=src,
            )
            results.append(
                SearchResult(chunk=chunk, score=hit["_score"], retrieval_method="sparse")
            )
        return results


# ---------------------------------------------------------------------------
# Hybrid search engine — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class HybridSearchEngine:
    """
    Runs dense and sparse search in parallel and merges results via RRF.
    """

    def __init__(
        self,
        dense_store: DenseVectorStore,
        sparse_index: SparseIndex,
    ) -> None:
        self._dense = dense_store                   # Qdrant dense index
        self._sparse = sparse_index                 # Elasticsearch BM25 index
        self._cfg = get_config().vector_store["hybrid_search"] 

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Execute both dense and sparse search, then merge via RRF.

        Args:
            query:          Raw query string (for BM25).
            query_vector:   Dense query embedding (for ANN).
            top_k:          Final number of results after fusion.
            namespace:      Tenant namespace for ACL filtering.
            metadata_filter: Additional metadata filters applied to dense search.
        """
        # -- Dense retrieval (ANN in Qdrant)
        dense_results = self._dense.search(
            query_vector=query_vector,
            top_k=top_k * 2,                        # Over-retrieve to compensate for fusion reordering
            namespace=namespace,
            metadata_filter=metadata_filter,
        )

        # -- Sparse retrieval (BM25 in Elasticsearch)
        sparse_results = self._sparse.search(
            query=query,
            top_k=top_k * 2,
            namespace=namespace,
        )

        # -- Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        return merged

    def _reciprocal_rank_fusion(
        self,
        dense: List[SearchResult],
        sparse: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF score = Σ  1 / (k + rank_i)
        where k is the RRF smoothing constant (typically 60) and
        rank_i is the 1-based position of a result in each list.
        """
        rrf_k = self._cfg["rrf_k"]                  # Smoothing constant (e.g. 60)
        scores: Dict[str, float] = {}               # chunk_id → cumulative RRF score
        chunk_map: Dict[str, SearchResult] = {}     # chunk_id → best SearchResult

        def _accumulate(results: List[SearchResult]) -> None:
            """Add RRF contribution from one result list."""
            for rank, result in enumerate(results, start=1):
                cid = result.chunk.chunk_id or result.chunk.text[:50]  # Use text prefix as fallback key
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)  # Accumulate RRF score
                if cid not in chunk_map:
                    chunk_map[cid] = result         # Keep the first occurrence for payload

        _accumulate(dense)                        
        _accumulate(sparse)                        

        # Sort by descending RRF score and return top_k results
        sorted_cids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
        merged = []
        for rank, cid in enumerate(sorted_cids, start=1):
            result = chunk_map[cid]
            result.score = scores[cid]              # Replace original score with RRF score
            result.retrieval_method = "hybrid"      # Mark as hybrid result
            result.rank = rank                      # Assign final rank
            merged.append(result)

        return merged