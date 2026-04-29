# Chunking Strategy
# Supports three modes:
#   fixed        — Token-window chunking (baseline, not recommended)
#   semantic     — Embedding-based topic-boundary chunking (production)
#   hierarchical+semantic — Three-level tree: document → section → paragraph nodes


import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import get_config, get_secrets
from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches an ATX heading at the start of a string or line
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Honorifics/abbreviations that end with a period but are NOT sentence endings.
# Used by _is_sentence_boundary() to guard against false splits.
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.$",
    re.IGNORECASE,
)

# Candidate split: a period/!/? followed by whitespace then an uppercase letter
# or digit/quote/bracket.  Fixed-width lookbehind only — Python re compatible.
_CANDIDATE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _is_sentence_boundary(left: str) -> bool:
    """
    Return True if the text to the left of a candidate split point is a real
    sentence ending (not an abbreviation or honorific).

    Called per candidate match so we can apply the variable-width abbreviation
    check without embedding it inside the regex lookbehind.
    """
    left = left.rstrip()
    if _ABBREV.search(left):
        return False
    return True


def _split_on_boundaries(text: str) -> List[str]:
    """
    Split prose text at true sentence boundaries.

    Strategy:
      1. Find all candidate split positions via _CANDIDATE_SPLIT.
      2. For each candidate, inspect the text to its left with _is_sentence_boundary().
      3. Only confirmed boundaries are used as split points.
    """
    sentences: List[str] = []
    prev = 0
    for match in _CANDIDATE_SPLIT.finditer(text):
        left_fragment = text[prev:match.start()]
        if _is_sentence_boundary(left_fragment):
            sentences.append(left_fragment.strip())
            prev = match.end()
    # Append the tail
    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)
    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkNode:
    """
    A node in the hierarchical chunk tree.
    Wraps a ParsedChunk with parent/child relationship metadata.
    """
    chunk: ParsedChunk
    level: str = "paragraph"       # document | section | paragraph
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Splits ConsolidatedChunk / ParsedChunk objects into index-ready ChunkNodes.

    Semantic mode uses embedding-based topic-boundary detection:
      1. Split text into sentences
      2. Embed each sentence via the configured embedding model
      3. Compute cosine similarity between adjacent sentence-window embeddings
      4. Detect split points at local similarity minima (topic shifts)
      5. Merge small resulting chunks up to chunk_size token ceiling
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._chunk_cfg = cfg.chunking
        self._tokeniser = tiktoken.get_encoding("cl100k_base")
        # Embedder callable is built once on first use; None until then.
        self._embedder = None

    # -------------------------------------------------------------------------
    # Embedding client (initialised once, reused across all chunks)
    # -------------------------------------------------------------------------

    def _get_embedder(self):
        """
        Return the embedding callable, constructing it on first call.

        Supports two backends controlled by chunking.embedding_backend:
          - "openai"                : OpenAI text-embedding-3-small (default)
          - "sentence_transformers" : local BAAI model (offline / cost-free)
        """
        if self._embedder is not None:
            return self._embedder

        backend = self._chunk_cfg.get("embedding_backend", "openai")

        if backend == "openai":
            client = OpenAI(api_key=get_secrets().open_ai_key)
            model  = self._chunk_cfg.get("embedding_model", "text-embedding-3-small")

            def embed(texts: List[str]) -> np.ndarray:
                response = client.embeddings.create(input=texts, model=model)
                return np.array([item.embedding for item in response.data], dtype=np.float32)

            self._embedder = embed

        elif backend == "sentence_transformers":
            model_name = self._chunk_cfg.get("embedding_model", "all-MiniLM-L6-v2")
            st_model   = SentenceTransformer(model_name)

            def embed(texts: List[str]) -> np.ndarray:
                return st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

            self._embedder = embed

        else:
            raise ValueError(f"Unknown embedding_backend: {backend!r}")

        return self._embedder



    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def chunk(self, document_chunks: List[ParsedChunk]) -> List[ChunkNode]:
        """
        Accept a list of Consolidated/ParsedChunks and return a flat list of
        ChunkNodes ready for embedding and indexing.
        """
        strategy  = self._chunk_cfg["strategy"]
        all_nodes: List[ChunkNode] = []

        for doc_chunk in document_chunks:

            if doc_chunk.modality in {"table", "image_caption"}:
                all_nodes.append(ChunkNode(chunk=doc_chunk, level="paragraph"))
                continue

            if strategy == "fixed":
                nodes = self._fixed_chunking(doc_chunk)
            elif strategy == "semantic":
                nodes = self._semantic_chunking(doc_chunk)
            elif strategy == "hierarchical":
                nodes = self._hierarchical_chunking(doc_chunk)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy!r}")

            all_nodes.extend(nodes)

        logger.info(
            f"Chunking produced {len(all_nodes)} nodes "
            f"from {len(document_chunks)} consolidated sections "
            f"[strategy={strategy}]"
        )
        return all_nodes

    # -------------------------------------------------------------------------
    # Fixed-size chunking (baseline — not recommended to be used in production)
    # -------------------------------------------------------------------------

    def _fixed_chunking(self, source: ParsedChunk) -> List[ChunkNode]:
        """
        Naive token-window chunking.  Breaks sentences arbitrarily.
        Included only as a baseline for ablation studies.
        """
        chunk_size = self._chunk_cfg["chunk_size"]
        overlap    = self._chunk_cfg["chunk_overlap"]
        tokens     = self._tokeniser.encode(source.text)
        nodes: List[ChunkNode] = []
        start = 0

        while start < len(tokens):
            end           = min(start + chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            text          = self._tokeniser.decode(window_tokens)

            if len(window_tokens) >= self._chunk_cfg["min_chunk_size"] or end == len(tokens):
                nodes.append(ChunkNode(
                    chunk=self._clone_chunk(source, text),
                    level="paragraph",
                ))
            start += chunk_size - overlap

        return nodes

    # -------------------------------------------------------------------------
    # Semantic chunking  ← production implementation
    # -------------------------------------------------------------------------

    def _semantic_chunking(
        self,
        source: ParsedChunk,
        override_chunk_size: Optional[int] = None,
    ) -> List[ChunkNode]:
        """
        Embedding-based semantic chunking.

        Algorithm
        ---------
        1. Split text into sentences (table blocks kept atomic).
        2. Build a rolling context window of `window_size` sentences and embed
           each window — this gives each sentence a context-aware representation
           rather than embedding it in isolation.
        3. Compute cosine similarity between every pair of adjacent windows.
        4. Detect split points at local similarity minima that fall below
           `breakpoint_percentile` of the similarity distribution.
        5. Collect sentences between split points into raw segments.
        6. Merge segments that are below `min_chunk_size` tokens into their
           neighbour, then split any segment that exceeds `chunk_size` tokens
           with a token-safe hard split (respects sentence boundaries).
        """
        chunk_size   = self._chunk_cfg["override_chunk_size"] or self._chunk_cfg["chunk_size"]
        min_size     = self._chunk_cfg["min_chunk_size"]
        percentile   = self._chunk_cfg.get("breakpoint_percentile", 0.25)
        window_size  = self._chunk_cfg.get("window_size", 3)

        sentences = self._split_sentences(source.text)

        # Edge case: very short text — emit as a single chunk
        if len(sentences) <= window_size:
            return [ChunkNode(
                chunk=self._clone_chunk(source, source.text.strip()),
                level="paragraph",
            )]

        # ------- Step 1: build rolling context windows and embed them -------

        windows = self._build_windows(sentences, window_size)
        embed   = self._get_embedder()
        embeddings = embed(windows)          # shape: (n_sentences, dim)

        # ------- Step 2: cosine similarity between adjacent windows -------

        similarities = self._adjacent_similarities(embeddings)
        
        # ------- Step 3: detect breakpoints — low similarity = topic shift -------

        breakpoint_indices = self._detect_breakpoints(similarities, percentile)

        # ------- Step 4: collect raw segments from breakpoint positions -------

        segments = self._sentences_to_segments(sentences, breakpoint_indices)

        # ------- Step 5: merge tiny segments and hard-split oversized ones -------

        segments = self._merge_small_segments(segments, min_size)
        segments = self._split_large_segments(segments, chunk_size, source)

        # ------- Step 6: build ChunkNodes -------

        nodes: List[ChunkNode] = []
        for seg_text in segments:
            tok_count = len(self._tokeniser.encode(seg_text))
            if tok_count < min_size and nodes:
                # Last-fragment guard: absorb into previous chunk
                prev_text = nodes[-1].chunk.text + " " + seg_text
                nodes[-1] = ChunkNode(
                    chunk=self._clone_chunk(source, prev_text.strip()),
                    level="paragraph",
                )
            else:
                nodes.append(ChunkNode(
                    chunk=self._clone_chunk(source, seg_text),
                    level="paragraph",
                ))

        # Always emit at least one node
        if not nodes:
            nodes.append(ChunkNode(
                chunk=self._clone_chunk(source, source.text.strip()),
                level="paragraph",
            ))

        return nodes


    # -------------------------------------------------------------------------
    # Hierarchical chunking
    # -------------------------------------------------------------------------

    def _hierarchical_chunking(self, source: ParsedChunk) -> List[ChunkNode]:
        """
        Build a three-level hierarchy:
          Level 0 (document)  — Full consolidated section as one node
          Level 1 (section)   — Heading-delimited sub-sections
          Level 2 (paragraph) — Semantic chunks within each section

        Paragraph-level splits use semantic chunking with a tighter
        override_chunk_size=400 appropriate for sub-section granularity.
        """
        doc_id = str(uuid.uuid4())
        document_node = ChunkNode(
            chunk=self._clone_chunk(source, source.text),
            level="document",
            parent_id=None,
            children_ids=[],
        )
        document_node.chunk.chunk_id = doc_id

        section_nodes:   List[ChunkNode] = []
        paragraph_nodes: List[ChunkNode] = []

        sections = self._split_sections(source.text)

        for section_text in sections:
            section_id = str(uuid.uuid4())

            heading = _extract_heading(section_text)
            depth   = _heading_depth(section_text)

            section_clone = self._clone_chunk(source, section_text)
            section_clone.metadata["section"]       = heading
            section_clone.metadata["section_depth"] = depth

            section_node = ChunkNode(
                chunk=section_clone,
                level="section",
                parent_id=doc_id,
                children_ids=[],
            )
            section_node.chunk.chunk_id = section_id
            document_node.children_ids.append(section_id)
            section_nodes.append(section_node)

            # Semantic chunking with a tighter ceiling for paragraph granularity
            para_source = self._clone_chunk(source, section_text)
            para_source.metadata["section"] = heading
            para_nodes = self._semantic_chunking(para_source, self._chunk_cfg["override_chunk_size"])

            for para_node in para_nodes:
                para_node.parent_id = section_id
                para_node.level     = "paragraph"
                para_node.chunk.metadata["section"]           = heading
                para_node.chunk.metadata["section_depth"]     = depth
                para_node.chunk.metadata["parent_section_id"] = section_id
                section_node.children_ids.append(para_node.chunk.chunk_id or "")
                paragraph_nodes.append(para_node)

        return [document_node] + section_nodes + paragraph_nodes

    # -------------------------------------------------------------------------
    # Semantic chunking helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_windows(sentences: List[str], window_size: int) -> List[str]:
        """
        Build a rolling context window for each sentence position.

        For sentence i, the window is the concatenation of sentences
        [i : i + window_size].  This gives each position a richer
        representation than embedding a single sentence in isolation.

        The last (window_size - 1) positions reuse the final window so
        the output length equals len(sentences).
        """
        windows = []
        n = len(sentences)
        for i in range(n):
            end    = min(i + window_size, n)
            window = " ".join(sentences[i:end])
            windows.append(window)
        return windows

    @staticmethod
    def _adjacent_similarities(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between every pair of adjacent embeddings.

        Returns an array of shape (n - 1,) where element i is
        cosine_similarity(embeddings[i], embeddings[i+1]).
        """
        n = len(embeddings)
        sims = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            a = embeddings[i].reshape(1, -1)
            b = embeddings[i + 1].reshape(1, -1)
            sims[i] = cosine_similarity(a, b)[0, 0]
        return sims

    @staticmethod
    def _detect_breakpoints(
        similarities: np.ndarray,
        percentile: float,
    ) -> List[int]:
        """
        Return indices where a chunk boundary should be inserted.

        A boundary is placed between sentence i and i+1 when
        similarities[i] falls below the `percentile`-th quantile of
        the similarity distribution — i.e. at the sharpest topic shifts.

        Index i in the return list means "split after sentence i".
        """
        if len(similarities) == 0:
            return []

        threshold = float(np.quantile(similarities, percentile))

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i)   # split after sentence i

        return breakpoints

    @staticmethod
    def _sentences_to_segments(
        sentences: List[str],
        breakpoint_indices: List[int],
    ) -> List[str]:
        """
        Collect sentences into text segments delimited by breakpoint positions.

        breakpoint_indices contains positions i meaning "split after sentence i",
        so segment boundaries are at i+1.
        """
        if not breakpoint_indices:
            return [" ".join(sentences)]

        segments: List[str] = []
        start = 0
        for bp in sorted(set(breakpoint_indices)):
            end = bp + 1           # exclusive end index
            segment = " ".join(sentences[start:end]).strip()
            if segment:
                segments.append(segment)
            start = end

        # Tail segment
        tail = " ".join(sentences[start:]).strip()
        if tail:
            segments.append(tail)

        return segments

    def _merge_small_segments(
        self,
        segments: List[str],
        min_size: int,
    ) -> List[str]:
        """
        Merge any segment that is below `min_size` tokens into its neighbour.

        Merge direction: prefer merging with the next segment; if there is no
        next segment, merge with the previous one.  This avoids emitting tiny
        orphan chunks from short transitional sentences.
        """
        if not segments:
            return segments

        merged: List[str] = []
        i = 0
        while i < len(segments):
            tok_count = len(self._tokeniser.encode(segments[i]))
            if tok_count < min_size:
                if i + 1 < len(segments):
                    # Absorb into the next segment (forward merge)
                    segments[i + 1] = segments[i] + " " + segments[i + 1]
                    i += 1
                    continue
                elif merged:
                    # Absorb into the previous segment (backward merge)
                    merged[-1] = merged[-1] + " " + segments[i]
                    i += 1
                    continue
            merged.append(segments[i])
            i += 1

        return merged

    def _split_large_segments(
        self,
        segments: List[str],
        chunk_size: int,
        source: ParsedChunk,
    ) -> List[str]:
        """
        Hard-split any segment that exceeds `chunk_size` tokens.

        Uses sentence boundaries rather than raw token slicing so output
        chunks always end at a sentence boundary.  Implemented as a greedy
        sentence-packing pass — the same approach as the old _semantic_chunking,
        but applied only as a safety net for oversized segments rather than as
        the primary splitting mechanism.
        """
        result: List[str] = []
        for segment in segments:
            tok_count = len(self._tokeniser.encode(segment))
            if tok_count <= chunk_size:
                result.append(segment)
                continue

            # Greedy sentence-pack within the oversized segment
            sentences     = self._split_sentences(segment)
            current_sents: List[str] = []
            current_count = 0

            for sent in sentences:
                sent_toks = len(self._tokeniser.encode(sent))
                if current_count + sent_toks > chunk_size and current_sents:
                    result.append(" ".join(current_sents).strip())
                    current_sents = []
                    current_count = 0
                current_sents.append(sent)
                current_count += sent_toks

            if current_sents:
                result.append(" ".join(current_sents).strip())

        return result

    # -------------------------------------------------------------------------
    # Shared text helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text at sentence boundaries.

        Markdown table blocks are extracted before sentence splitting and
        re-inserted as atomic tokens so they are never split mid-cell.

        Sentence splitting uses _split_on_boundaries(), which applies a
        two-pass approach (candidate regex + per-match abbreviation guard)
        to avoid the variable-width lookbehind that Python's re module
        does not support.
        """
        # Isolate table blocks (lines beginning with '|') from prose
        parts = re.split(r"(\n(?:\|[^\n]*\n)+)", "\n" + text)

        result: List[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if stripped.startswith("|"):
                # Entire table block → one atomic sentence
                result.append(stripped)
            else:
                result.extend(_split_on_boundaries(stripped))

        return result

    @staticmethod
    def _split_sections(text: str) -> List[str]:
        """
        Split Markdown at heading boundaries using a zero-width lookahead
        so the heading line stays at the start of the section it belongs to.
        """
        sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)

        if len(sections) > 1:
            return [s.strip() for s in sections if s.strip()]

        # No markdown headings — fall back to paragraph splitting
        sections = text.split("\n\n")
        return [s.strip() for s in sections if s.strip()]

    @staticmethod
    def _clone_chunk(source: ParsedChunk, new_text: str) -> ParsedChunk:
        """Shallow copy of a ParsedChunk with new text and a recomputed fingerprint."""
        clone = ParsedChunk(
            text=new_text,
            metadata=dict(source.metadata),
            source_file=source.source_file,
            source_name=source.source_name,
            modality=source.modality,
            language=source.language,
            doc_version=source.doc_version,
            ingestion_ts=source.ingestion_ts,
        )
        clone.chunk_id = clone.compute_fingerprint()
        return clone


# ---------------------------------------------------------------------------
# Module-level heading helpers
# ---------------------------------------------------------------------------

def _extract_heading(section_text: str) -> str:
    """Extract the heading text from the first heading line of a section string."""
    match = _HEADING_RE.search(section_text)
    return match.group(2).strip() if match else ""


def _heading_depth(section_text: str) -> int:
    """Return the depth (1–6) of the first heading line in a section string."""
    match = _HEADING_RE.search(section_text)
    return len(match.group(1)) if match else 0