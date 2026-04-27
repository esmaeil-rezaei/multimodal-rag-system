import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.parser import ParsedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class ConsolidatedChunk:
    """
    A section-level chunk produced by merging multiple ParsedChunks that share
    the same parent heading.  Carries both a merged metadata view and the full
    list of original per-element metadata for downstream traceability.
    """
    text: str
    source_file: str
    source_name: str
    modality: str = "text"
    language: str = ""
    doc_version: Optional[str] = None
    ingestion_ts: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    element_metadata: List[Dict[str, Any]] = field(default_factory=list)

    def compute_fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def chunk_id(self) -> Optional[str]:
        return self.metadata.get("chunk_id")

    @chunk_id.setter
    def chunk_id(self, value: str) -> None:
        self.metadata["chunk_id"] = value


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

class ChunkConsolidator:
    """
    Groups the small ParsedChunks emitted by the parser into section-sized
    ConsolidatedChunks, then passes them to the chunker.

    Text assembly rules per element category:
      Title         → markdown heading prefix (## heading text)
      Table         → wrapped in [TABLE]…[/TABLE] sentinel so the chunker
                      treats the whole table as an atomic unit
      Image         → wrapped in [IMAGE_DESCRIPTION]…[/IMAGE_DESCRIPTION]
      NarrativeText / ListItem / everything else → plain newline join
    """

    def consolidate(self, parsed_chunks: List[ParsedChunk]) -> List[ConsolidatedChunk]:
        """
        Group parsed_chunks by section, merge each group, return a list of
        ConsolidatedChunks in document order.
        """
        if not parsed_chunks:
            return []

        groups: Dict[str, List[ParsedChunk]] = {}
        group_order: List[str] = []             # Preserve document order

        for chunk in parsed_chunks:
            key = self._section_key(chunk)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(chunk)

        consolidated: List[ConsolidatedChunk] = []
        for key in group_order:
            merged = self._merge_group(groups[key])
            if merged.text.strip():             # Drop empty groups
                consolidated.append(merged)

        logger.info(
            f"Consolidation: {len(parsed_chunks)} elements → {len(consolidated)} sections"
        )
        return consolidated

    # -------------------------------------------------------------------------
    # Grouping key
    # -------------------------------------------------------------------------

    @staticmethod
    def _section_key(chunk: ParsedChunk) -> str:
        """
        The key used to group elements into the same consolidated chunk.
        Priority order:
          1. metadata["section"] — set by _parse_md's heading tracker or
             the element_to_chunk helper
          2. metadata["breadcrumb"] — if the parser attached a full path
          3. source_file — fallback so elements from different files never merge
        """
        section = (
            chunk.metadata.get("section")
            or chunk.metadata.get("breadcrumb")
            or ""
        )
        # Include source_file in the key so two files with identically named
        # sections don't accidentally merge into the same group
        return f"{chunk.source_file}::{section}"


    # -------------------------------------------------------------------------
    # Group merge
    # -------------------------------------------------------------------------

    def _merge_group(self, chunks: List[ParsedChunk]) -> ConsolidatedChunk:
        """Merge a list of same-section ParsedChunks into one ConsolidatedChunk."""
        text_parts: List[str] = []
        all_metadata: List[Dict[str, Any]] = [c.metadata for c in chunks]

        for chunk in chunks:
            category = chunk.metadata.get("category", "")
            part = self._format_element(chunk.text.strip(), category,
                                        chunk.metadata.get("section_depth", 2))
            if part:
                text_parts.append(part)

        merged_text = "\n\n".join(p for p in text_parts if p.strip())

        return ConsolidatedChunk(
            text=merged_text,
            source_file=chunks[0].source_file,
            source_name=chunks[0].source_name,
            modality=self._dominant_modality(chunks),
            language=self._dominant_language(chunks),
            doc_version=chunks[0].doc_version,
            ingestion_ts=chunks[0].ingestion_ts,
            metadata=self._merge_metadata(all_metadata),
            element_metadata=all_metadata,
        )

    # -------------------------------------------------------------------------
    # Text formatting per element category
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_element(text: str, category: str, depth: int) -> str:
        """
        Format a single element's text according to its category so downstream
        components (chunker, embedder) can handle it correctly.
        """
        if not text:
            return ""

        if category == "Title":
            # Re-emit as a markdown heading so the chunker's _split_sections
            # recognises it as a section boundary in hierarchical mode
            hashes = "#" * max(1, min(depth, 6))
            return f"{hashes} {text}"

        elif category == "Table":
            # Sentinel wrapping signals to the chunker: never split this block
            return f"[TABLE]\n{text}\n[/TABLE]"

        elif category == "Image":
            # Sentinel wrapping keeps the VLM caption atomic and labelled
            return f"[IMAGE_DESCRIPTION]\n{text}\n[/IMAGE_DESCRIPTION]"

        elif category in {"ListItem", "ListItem.Bulleted", "ListItem.Numbered"}:
            # Prefix with dash so merged list items stay readable as a list
            return f"- {text}" if not text.startswith("-") else text

        else:
            # NarrativeText, FigureCaption, Header, Footer, Address, etc.
            return text


    # -------------------------------------------------------------------------
    # Metadata merge strategies
    # -------------------------------------------------------------------------

    @staticmethod
    def _merge_metadata(all_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge per-element metadata dicts into one section-level dict.

        Rules:
          Scalar fields  → first non-null wins
          page_number    → page_start / page_end range (min / max)
          categories     → union list  (for retrieval filtering)
          element_ids    → ordered list (full provenance chain)
          convenience flags → has_table, has_image, has_list
        """
        if not all_meta:
            return {}

        merged: Dict[str, Any] = {}

        # Scalar: first non-null wins
        for key in ("section", "section_depth", "filename", "breadcrumb",
                    "front_matter", "suffix", "doc_version"):
            for m in all_meta:
                val = m.get(key)
                if val is not None and val != "":
                    merged[key] = val
                    break

        # Page range
        pages = [m["page_number"] for m in all_meta
                 if isinstance(m.get("page_number"), int)]
        if pages:
            merged["page_start"] = min(pages)
            merged["page_end"] = max(pages)

        # Union of all categories
        categories = list({m["category"] for m in all_meta if m.get("category")})
        merged["categories"] = categories
        merged["has_table"] = "Table" in categories
        merged["has_image"] = "Image" in categories
        merged["has_list"] = any(
            c.startswith("ListItem") for c in categories
        )

        # Full provenance: every element ID in order
        merged["element_ids"] = [
            m["element_id"] for m in all_meta if m.get("element_id")
        ]
        merged["element_count"] = len(all_meta)

        return merged

    # -------------------------------------------------------------------------
    # Dominant language / modality helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _dominant_language(chunks: List[ParsedChunk]) -> str:
        """Return the majority language across all non-None language values."""
        langs = [c.language for c in chunks if c.language]
        if not langs:
            return ""
        return Counter(langs).most_common(1)[0][0]

    @staticmethod
    def _dominant_modality(chunks: List[ParsedChunk]) -> str:
        """
        Determine the consolidated chunk's modality.
        If the group contains only tables → table.
        If only image captions → image_caption.
        Any prose present → text (prose wins because it drives the embedding).
        """
        modalities = {c.modality for c in chunks}
        if modalities == {"table"}:
            return "table"
        if modalities == {"image_caption"}:
            return "image_caption"
        return "text"