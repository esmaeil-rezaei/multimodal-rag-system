# src/indexing/embedder.py
# =============================================================================
# Stage 2 — Embedding Models.
# Covers:
#   - Domain-specific embedding model selection (legal, medical, …)
#   - Multi-lingual & cross-lingual retrieval
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np                               
from sentence_transformers import SentenceTransformer

from src.config.settings import get_config, get_secrets 
from src.ingestion.chunker import ChunkNode        
from src.ingestion.parser import ParsedChunk     
from src.utils.logger import get_logger      

logger = get_logger(__name__)


class EmbeddingRouter:
    """
    Routes each chunk to the most appropriate embedding model based on:
      1. The source sub-folder name (domain routing).
      2. The detected language of the chunk text (multilingual routing).

    Maintains a lazy-loaded model registry to avoid loading all models at startup.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._emb_cfg = self._cfg.embeddings  

        self._default_model_name: str = self._emb_cfg["default_model"]
        self._multilingual_model_name: str = self._emb_cfg["multilingual_model"]
        self._domain_model_map: Dict[str, str] = self._emb_cfg.get("domain_models", {})

        self._batch_size: int = self._emb_cfg["batch_size"]
        self._lang_detection: bool = self._emb_cfg["language_detection"]

        self._model_cache: Dict[str, SentenceTransformer] = {}
        self._load_model(self._default_model_name)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def embed_nodes(self, nodes: List[ChunkNode]) -> List[Tuple[ChunkNode, np.ndarray]]:
        """
        Embed a list of ChunkNodes, routing each to the appropriate model.

        Returns a list of (ChunkNode, embedding_vector) pairs.
        """
        # Group nodes by the model they should use for efficient batched encoding
        groups: Dict[str, List[Tuple[int, ChunkNode]]] = {}

        for idx, node in enumerate(nodes):
            model_name = self._select_model(node.chunk) 
            groups.setdefault(model_name, []).append((idx, node))

        result_pairs: List[Optional[Tuple[ChunkNode, np.ndarray]]] = [None] * len(nodes)

        for model_name, indexed_nodes in groups.items():
            model = self._load_model(model_name)        
            texts = [n.chunk.text for _, n in indexed_nodes] 

            # Encode in batches for memory efficiency
            vectors = self._batch_encode(model, texts)   # np.ndarray of shape (len, dim)

            for i, (original_idx, node) in enumerate(indexed_nodes):
                result_pairs[original_idx] = (node, vectors[i])

        # Filter out None entries (should not occur, but defensive)
        return [(node, vec) for pair in result_pairs if pair for node, vec in [pair]]

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------

    def _select_model(self, chunk: ParsedChunk) -> str:
        """
        Choose the embedding model for a single chunk.

        Priority order:
          1. Domain model (if chunk's source_name matches a domain_models key)
          2. Multilingual model (if language != English and cross-lingual enabled)
          3. Default model (fallback)
        """
        # Domain-specific model selection by source sub-folder name
        # e.g. "legal" found in "legal_corpus"
        source = (chunk.source_name or "").lower()
        for domain_key, domain_model in self._domain_model_map.items():
            if domain_key.lower() in source:         
                logger.debug(f"Domain model '{domain_model}' selected for source '{source}'")
                return domain_model

        # Multilingual routing — switch to multilingual encoder for non-English text
        if self._lang_detection and chunk.language and chunk.language != "en":
            logger.debug(
                f"Multilingual model selected for language '{chunk.language}'"
            )
            return self._multilingual_model_name

        return self._default_model_name        

    # -------------------------------------------------------------------------
    # Model loading — lazy, cached
    # -------------------------------------------------------------------------

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a SentenceTransformer model by name, caching it after first load.
        Models are downloaded from HuggingFace Hub on first use.
        """
        if model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(
                model_name,
                device="cuda" if self._gpu_available() else "cpu",  # Use GPU if available
            )
        return self._model_cache[model_name]

    # -------------------------------------------------------------------------
    # Batch encoding
    # -------------------------------------------------------------------------

    def _batch_encode(
        self, model: SentenceTransformer, texts: List[str]
    ) -> np.ndarray:
        """
        Encode a list of texts in mini-batches.
        Mini-batching prevents GPU OOM for large document sets.

        Returns np.ndarray of shape (len(texts), embedding_dim).
        """
        all_vectors: List[np.ndarray] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]  # Current mini-batch
            vectors = model.encode(
                batch,
                show_progress_bar=False,            # Suppress tqdm output in production
                normalize_embeddings=True,          # L2-normalise → cosine sim == dot product
                convert_to_numpy=True,              # Return numpy arrays, not torch tensors
            )
            all_vectors.append(vectors)

        return np.vstack(all_vectors)              

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _gpu_available() -> bool:
        """Check whether a CUDA-capable GPU is available for model inference."""
        try:
            import torch                            # Lazy import
            return torch.cuda.is_available()
        except ImportError:
            return False                            # No torch → no GPU support


class QueryEmbedder:
    """
    Lightweight wrapper that embeds a single query string.
    Shares the same EmbeddingRouter to reuse cached models.
    """

    def __init__(self, router: Optional[EmbeddingRouter] = None) -> None:
        self._router = router or EmbeddingRouter()
        self._cfg = get_config()
        self._default_model_name = self._cfg.embeddings["default_model"]

    def embed_query(self, query: str, language: Optional[str] = None) -> np.ndarray:
        """
        Embed a query string using the appropriate model.
        Returns a 1D numpy array of shape (embedding_dim,).
        """
        dummy_chunk = ParsedChunk(text=query, language=language)
        model_name = self._router._select_model(dummy_chunk)
        model = self._router._load_model(model_name)          # Load (or cache-hit) the model
        vector: np.ndarray = model.encode(
            query,
            normalize_embeddings=True,          
            convert_to_numpy=True,
        )
        return vector                               # 1D array of shape (dim,)
