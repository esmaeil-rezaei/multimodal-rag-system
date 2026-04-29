
# src/ingestion/deduplicator.py
# =============================================================================
# Duplicate & near-duplicate content detection.
# Two-phase approach:
#   Phase 1 — Exact dedup via SHA-256 hash lookup  (O(1) per chunk)
#   Phase 2 — Fuzzy dedup via MinHash + Jaccard similarity  (O(n) per chunk)
# =============================================================================

from __future__ import annotations

import hashlib                                  
from typing import Dict, List, Optional, Set

from datasketch import MinHash, MinHashLSH

from src.config.settings import get_config         
from src.ingestion.parser import ParsedChunk       
from src.utils.logger import get_logger            

logger = get_logger(__name__)


class Deduplicator:
    """
    Stateful deduplication engine.
    Maintains an in-memory seen-hashes set and an LSH index across the lifetime
    of a single ingestion run.  For production, these should be backed by Redis.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._dedup_cfg = cfg.ingestion["deduplication"] 

        self._num_perm: int = self._dedup_cfg["minhash_num_perm"]
        self._threshold: float = self._dedup_cfg["jaccard_threshold"]
        self._keep_strategy: str = self._dedup_cfg["keep_strategy"]

        self._exact_hashes: Set[str] = set()

        self._lsh = MinHashLSH(
            threshold=self._threshold,              # Jaccard threshold for LSH buckets
            num_perm=self._num_perm,                # Must match MinHash permutations
        )
        self._lsh_registry: Dict[str, tuple[ParsedChunk, MinHash]] = {}  # Maps LSH key → ParsedChunk for lookup

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def filter(self, chunks: List[ParsedChunk]) -> List[ParsedChunk]:
        """
        Accept a batch of ParsedChunks and return the deduplicated subset.

        Steps:
          1. Exact hash dedup (fastest, zero false negatives for identical text).
          2. MinHash fuzzy dedup (catches near-duplicates with minor edits).
          3. Apply keep_strategy to choose canonical representative when conflicts arise.
        """
        if not self._dedup_cfg["enabled"]:
            return chunks                         

        unique: List[ParsedChunk] = []              # Result accumulator
        duplicates_exact = 0                        # Counter for reporting
        duplicates_fuzzy = 0

        for chunk in chunks:
            # ---- Phase 1: Exact duplicate check --------------------------------
            norm_text = self._normalize(chunk.text)
            exact_hash = self._compute_exact_hash(norm_text)
            if exact_hash in self._exact_hashes:
                duplicates_exact += 1
                logger.debug(f"Exact duplicate skipped: {chunk.chunk_id}")
                continue                            

            # ---- Phase 2: Fuzzy / near-duplicate check -------------------------
            minhash = self._compute_minhash(norm_text)
            similar_keys: List[str] = self._lsh.query(minhash)

            if similar_keys:
                winner = self._resolve_conflict(chunk, similar_keys)
                if winner is not chunk:
                    # The existing candidate is preferred — skip incoming chunk
                    duplicates_fuzzy += 1
                    logger.debug(
                        f"Near-duplicate skipped (jaccard ≥ {self._threshold}): "
                        f"{chunk.chunk_id}"
                    )
                    continue

            # ---- Register as canonical -----------------------------------------
            self._exact_hashes.add(exact_hash)     # Mark exact hash as seen
            lsh_key = chunk.chunk_id or exact_hash  # Unique key for LSH registry
            try:
                self._lsh.insert(lsh_key, minhash)  # Add to MinHash LSH index
            except ValueError:
                pass                               # Key already exists — safe to ignore
            self._lsh_registry[lsh_key] = (chunk, minhash)    # Store reference for conflict resolution
            unique.append(chunk)                  

        logger.info(
            f"Deduplication complete: {len(unique)} unique, "
            f"{duplicates_exact} exact dups, {duplicates_fuzzy} fuzzy dups removed"
        )
        return unique

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())
    
    @staticmethod
    def _compute_exact_hash(text: str) -> str:
        """Return the SHA-256 hex digest of a text string (Phase 1 dedup key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _compute_minhash(self, text: str) -> MinHash:
        """
        Build a MinHash signature from the n-gram shingles of a text string.
        Character-level 3-grams are a robust choice for short text dedup.
        """
        mh = MinHash(num_perm=self._num_perm)     
        shingles = self._shingle(text, k=3)         # Extract character 3-grams as the feature set
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))      # Add each shingle to the MinHash sketch
        return mh
    
    @staticmethod
    def _shingle(text: str, k: int = 3) -> Set[str]:
        """
        Extract all character k-grams (shingles) from a text string.
        Using character-level rather than word-level shingles handles
        minor edits and formatting differences more robustly.
        """
        text = " ".join(text.lower().split())
        return {text[i : i + k] for i in range(len(text) - k + 1)}  # Sliding window of size k

    def _resolve_conflict(
        self, incoming: ParsedChunk, existing_keys: List[str]
    ) -> ParsedChunk:
        """
        Given an incoming chunk and the keys of similar existing chunks,
        return whichever ParsedChunk should be kept (the canonical version).
        """
        existing_chunks = [
            self._lsh_registry[k]
            for k in existing_keys
            if k in self._lsh_registry           # Defensive lookup
        ]
        if not existing_chunks:
            return incoming                     

        strategy = self._keep_strategy
        if strategy == "newest":
            # Keep the chunk with the latest ingestion timestamp
            all_candidates = existing_chunks + [incoming]
            return max(
                all_candidates,
                key=lambda c: c.ingestion_ts or "",  # Lexicographic ISO-8601 comparison
            )
        elif strategy == "oldest":
            # Keep the earliest ingested chunk
            all_candidates = existing_chunks + [incoming]
            return min(
                all_candidates,
                key=lambda c: c.ingestion_ts or "9999",
            )
        else:
            # "highest_quality": keep whichever has more text (proxy for information density)
            all_candidates = existing_chunks + [incoming]
            return max(all_candidates, key=lambda c: len(c.text))
