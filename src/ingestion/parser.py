# src/ingestion/parser.py
# Strategy summary:
#   1. PRIMARY SIGNAL  — unstructured.io classifies headings as Title elements.
#      category_depth (populated by hi_res strategy) encodes heading level.
#   2. FALLBACK SIGNAL — when category_depth is absent (scanned PDFs, fast
#      strategy), we infer depth from bounding-box height relative to the
#      tallest Title in the document (relative ranking, not absolute font size).
#   3. CROSS-FILE RELATEDNESS is a retrieval concern, not a parsing concern.
#      We emit a breadcrumb field so the retrieval layer can use semantic
#      similarity across sections from different files without coupling parsers.
# =============================================================================

from __future__ import annotations

import base64
import hashlib
import mimetypes
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import camelot
import openai
from langdetect import detect as detect_language
from unstructured.documents.elements import Element, Image, Table, Text, Title
from unstructured.partition.auto import partition
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedChunk:
    """
    One logical unit extracted from a source document.
    Canonical data structure passed between all pipeline stages.

    Metadata contract for the consolidator (all fields must be present
    regardless of source file type):
        category      — element type: Title, NarrativeText, Table, Image, …
        element_id    — unstructured element ID for provenance tracking
        section       — text of the nearest ancestor heading (leaf of stack)
        section_depth — numeric depth of that heading (1 = top-level)
        breadcrumb    — full path from root: "Introduction > Background"
        page_number   — PDF page (int) or None for non-PDF formats
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    source_file: Optional[str] = None
    source_name: Optional[str] = None
    modality: str = "text"                  # text | table | image_caption
    language: Optional[str] = None
    doc_version: Optional[str] = None
    ingestion_ts: Optional[str] = None

    def compute_fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Heading-tracker state machine
# ---------------------------------------------------------------------------
class _HeadingTracker:
    """
    Maintains a stack of (depth, heading_text) pairs and updates it as Title
    elements are encountered.

    This is the core mechanism that gives the consolidator section context for
    PDFs, exactly mirroring what ATX-heading parsing does for Markdown.

    Stack invariant: entries are in strictly increasing depth order.
    Pushing a heading at depth D pops everything with depth >= D first,
    so the stack always represents the current ancestor chain.

    Example trace:
        push(1, "Chapter 1")    → stack: [(1, "Chapter 1")]
        push(2, "Background")   → stack: [(1, "Chapter 1"), (2, "Background")]
        push(2, "Methods")      → stack: [(1, "Chapter 1"), (2, "Methods")]
        push(1, "Chapter 2")    → stack: [(1, "Chapter 2")]

    Breadcrumb for the above after the last push: "Chapter 2"
    Section (leaf) for the above after the last push: "Chapter 2"
    """

    def __init__(self) -> None:
        self._stack: List[Tuple[int, str]] = []

    def push(self, depth: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= depth:
            self._stack.pop()
        self._stack.append((depth, title))

    @property
    def section(self) -> str:
        """Text of the nearest heading (leaf of the stack)."""
        return self._stack[-1][1] if self._stack else ""

    @property
    def section_depth(self) -> int:
        """Numeric depth of the nearest heading."""
        return self._stack[-1][0] if self._stack else 0

    @property
    def breadcrumb(self) -> str:
        """Full ancestor path, e.g. 'Introduction > Background > Methods'."""
        return " > ".join(title for _, title in self._stack)


# ---------------------------------------------------------------------------
# Font-size depth inference (fallback when category_depth is absent)
# ---------------------------------------------------------------------------

def _infer_depth_from_font_size(
    title_elements: List[Element],
    max_levels: int = 3,
) -> Dict[int, int]:
    """
    Map element object ids → inferred heading depth when category_depth is
    not populated (e.g. fast strategy or scanned PDFs without OCR metadata).

    Algorithm:
        1. Extract the bounding-box height for every Title element.
        2. Sort unique heights descending (taller text = higher-level heading).
        3. Assign depth 1 to the tallest group, depth 2 to next, etc.
        4. Cap at max_levels to avoid over-segmentation on noisy PDFs.

    This is relative-within-document, not absolute, so it degrades gracefully
    even if font-size metadata is coarse or rounded.

    Returns: {id(element): depth_int}
    """
    heights: Dict[int, float] = {}
    for el in title_elements:
        meta = getattr(el, "metadata", None)
        coords = getattr(meta, "coordinates", None)
        if coords is None:
            continue
        # Unstructured coordinates: list of (x, y) corner tuples
        pts = getattr(coords, "points", None)
        if pts and len(pts) >= 2:
            ys = [p[1] for p in pts]
            heights[id(el)] = max(ys) - min(ys)

    if not heights:
        return {}

    # Cluster unique heights into at most max_levels buckets
    unique_sorted = sorted(set(heights.values()), reverse=True)
    # Simple equal-width bucketing
    n_levels = min(len(unique_sorted), max_levels)
    if n_levels == 0:
        return {}
    bucket_size = (unique_sorted[0] - unique_sorted[-1] + 1) / n_levels
    depth_map: Dict[int, int] = {}
    for eid, h in heights.items():
        bucket = int((unique_sorted[0] - h) / max(bucket_size, 1e-9))
        depth_map[eid] = min(bucket + 1, max_levels)  # 1-indexed depth
    return depth_map


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class DocumentParser:
    """
    Routes source files to the correct parsing strategy and returns a flat
    list of ParsedChunk objects — one per logical document element.

    Metadata contract: every chunk carries section/section_depth/breadcrumb
    regardless of source format.  The consolidator is format-agnostic.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._sec = get_secrets()
        self._ingest_cfg = self._cfg.ingestion
        self._openai = openai.OpenAI(api_key=self._sec.open_ai_key)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def parse_file(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        logger.info("Parsing file", extra={"file": str(file_path), "source": source_name})
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(file_path, source_name)
        elif suffix == ".md":
            return self._parse_md(file_path, source_name)
        elif suffix == ".txt":
            return self._parse_txt(file_path, source_name)
        elif suffix == ".docx":
            return self._parse_docx(file_path, source_name)
        elif suffix == ".html":
            return self._parse_html(file_path, source_name)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            return self._parse_standalone_image(file_path, source_name)
        else:
            logger.warning(f"Unsupported extension: {suffix} for {file_path}")
            return []

    # ---- PDF ----
    
    def _parse_pdf(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse PDF using unstructured.io + Camelot, emitting one chunk per
        element with full section context metadata.

        Section assignment strategy (two-pass):
          Pass 1 — collect all Title elements and build a font-size depth
                   inference map as a fallback for when category_depth is absent.
          Pass 2 — walk elements in document order, updating a _HeadingTracker
                   state machine on every Title.  Non-Title elements inherit the
                   current tracker state as their section/breadcrumb.

        This is strictly a formatting concern: we attach metadata.  The
        consolidator decides how to group based on that metadata.
        """
        parsing_cfg = self._ingest_cfg["parsing"]

        elements: List[Element] = partition_pdf(
            filename=str(file_path),
            strategy=parsing_cfg["pdf_strategy"],
            languages=parsing_cfg["ocr_languages"],
            extract_images_in_pdf=parsing_cfg["extract_images"],
            extract_image_block_output_dir=parsing_cfg["image_output_dir"],
        )

        # --- Pass 1: build fallback depth map from bounding-box heights ------
        title_elements = [el for el in elements if isinstance(el, Title)]
        fallback_depth_map = _infer_depth_from_font_size(title_elements)

        # --- Pass 2: walk elements in order with heading tracker -------------
        tracker = _HeadingTracker()
        chunks: List[ParsedChunk] = []

        for element in elements:
            if isinstance(element, Title):
                # Resolve depth: prefer unstructured's category_depth,
                # fall back to font-size inference, then default to 1.
                meta = getattr(element, "metadata", None)
                depth = (
                    getattr(meta, "category_depth", None)
                    or fallback_depth_map.get(id(element))
                    or 1
                )
                tracker.push(int(depth), element.text or "")

                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)

            elif isinstance(element, (Text,)):
                chunk = self._element_to_chunk(element, file_path, source_name, "text")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)

            elif isinstance(element, Table):
                chunk = self._element_to_chunk(element, file_path, source_name, "table")
                self._attach_section_context(chunk, tracker)
                chunks.append(chunk)

            elif isinstance(element, Image):
                if self._ingest_cfg["image_captioning"]["enabled"]:
                    caption = self._caption_image(element, file_path, source_name)
                    if caption:
                        self._attach_section_context(caption, tracker)
                        chunks.append(caption)

        if self._ingest_cfg["tables"]["extract_tables"]:
            camelot_chunks = self._extract_tables_camelot(file_path, source_name)
            for c in camelot_chunks:
                c.metadata.setdefault("section", tracker.section)
                c.metadata.setdefault("section_depth", tracker.section_depth)
                c.metadata.setdefault("breadcrumb", tracker.breadcrumb)
            chunks.extend(camelot_chunks)

        return chunks

    # ---- Markdown ----

    def _parse_md(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse Markdown using unstructured's partition_md with heading-tracker.
        Output shape matches _parse_pdf exactly.
        """
        text = _read_with_encoding_fallback(
            file_path, self._cfg.knowledge_base.get("encoding", "utf-8")
        )
        elements: List[Element] = partition_md(text=text)

        tracker = _HeadingTracker()
        chunks: List[ParsedChunk] = []

        for element in elements:
            if isinstance(element, Title):
                depth = getattr(
                    getattr(element, "metadata", None), "category_depth", 1
                ) or 1
                tracker.push(int(depth), element.text or "")

            chunk = self._element_to_chunk(element, file_path, source_name, "text")
            if isinstance(element, Table):
                chunk.modality = "table"
            self._attach_section_context(chunk, tracker)
            chunks.append(chunk)

        return chunks


    # ---- Plain text ----

    def _parse_txt(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse .txt as double-newline-separated paragraphs.
        No headings available — all chunks share section="" and depth=0.
        """
        text = _read_with_encoding_fallback(
            file_path, self._cfg.knowledge_base.get("encoding", "utf-8")
        )
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[ParsedChunk] = []
        for i, para in enumerate(paragraphs):
            chunk = ParsedChunk(
                text=para,
                modality="text",
                source_file=str(file_path),
                source_name=source_name,
                metadata={
                    "category": "NarrativeText",
                    "element_id": None,
                    "section": "",
                    "section_depth": 0,
                    "breadcrumb": "",
                    "page_number": None,
                    "paragraph_index": i,
                },
            )
            chunk.language = self._detect_language(para)
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks


    # ---- Table extraction ----

    def _extract_tables_camelot(
        self, file_path: Path, source_name: str
    ) -> List[ParsedChunk]:
        """
        Extract tables from a PDF using Camelot.
        Called once per file (not per element).

        Section metadata is intentionally left blank here and filled in by the
        caller (_parse_pdf) using the heading tracker's last known state, or
        by the consolidator via page-number proximity matching.
        """
        chunks: List[ParsedChunk] = []
        output_fmt = self._ingest_cfg["tables"]["output_format"]

        try:
            tables = camelot.read_pdf(str(file_path), pages="all", flavor="lattice")
        except Exception as exc:
            logger.warning(f"Camelot failed for {file_path}: {exc}")
            return chunks

        for i, table in enumerate(tables):
            df = table.df
            raw_header = df.iloc[0]
            col_metadata = [
                col.split("\n")[0] if "\n" in str(col) else str(col)
                for col in raw_header
            ]

            if output_fmt == "json":
                records = df.iloc[1:].to_dict(orient="records")
                text_repr = str(records)
            else:
                df_body = df.iloc[1:].copy()
                df_body.columns = col_metadata
                text_repr = df_body.to_markdown(index=False)

            chunk = ParsedChunk(
                text=text_repr,
                modality="table",
                source_file=str(file_path),
                source_name=source_name,
                metadata={
                    "category": "Table",
                    "element_id": None,
                    "table_index": i,
                    "page_number": table.page,
                    "columns": col_metadata,
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace,
                    "section": "",                     # section/breadcrumb filled by _parse_pdf after this returns
                    "section_depth": 0,
                    "breadcrumb": "",
                },
            )
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks


    # ---- Image captioning ----

    def _caption_image(
        self, element: Image, file_path: Path, source_name: str
    ) -> Optional[ParsedChunk]:
        """
        Caption an extracted image via GPT-4V.
        Falls back gracefully when image_path is missing.
        section/breadcrumb filled by caller after tracker update.
        """
        image_path: Optional[str] = getattr(
            getattr(element, "metadata", None), "image_path", None
        )
        if not image_path:
            logger.debug("Image element has no image_path — skipping captioning")
            return None

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except OSError as exc:
            logger.warning(f"Cannot read image file {image_path}: {exc}")
            return None

        if not image_bytes:
            return None

        mime_type, _ = mimetypes.guess_type(image_path)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type or 'image/png'};base64,{b64}"

        model = self._ingest_cfg["image_captioning"]["model"]
        max_tok = self._ingest_cfg["image_captioning"]["max_tokens"]

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url, "detail": "high"}},
                        {"type": "text",
                         "text": (
                             "Describe this figure or chart in detail, including all "
                             "axis labels, data series, trends, and key values visible."
                         )},
                    ],
                }],
                max_tokens=max_tok,
            )
        except Exception as exc:
            logger.error(f"VLM captioning failed: {exc}")
            return None

        caption: str = response.choices[0].message.content or ""

        chunk = ParsedChunk(
            text=caption,
            modality="image_caption",
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": "Image",
                "element_id": getattr(element, "id", None),
                "page_number": getattr(
                    getattr(element, "metadata", None), "page_number", None
                ),
                "section": "",       # section/breadcrumb filled by caller
                "section_depth": 0,
                "breadcrumb": "",
                "has_clip_embedding": False,
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk


    # ---- Standalone image files ----

    def _parse_standalone_image(
        self, file_path: Path, source_name: str
    ) -> List[ParsedChunk]:
        with open(file_path, "rb") as fh:
            image_bytes = fh.read()

        mime_type, _ = mimetypes.guess_type(str(file_path))
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type or 'image/png'};base64,{b64}"

        model = self._ingest_cfg["image_captioning"]["model"]
        max_tok = self._ingest_cfg["image_captioning"]["max_tokens"]

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url, "detail": "high"}},
                        {"type": "text", "text": "Describe this image in full detail."},
                    ],
                }],
                max_tokens=max_tok,
            )
        except Exception as exc:
            logger.error(f"Standalone image captioning failed for {file_path}: {exc}")
            return []

        caption = response.choices[0].message.content or ""
        chunk = ParsedChunk(
            text=caption,
            modality="image_caption",
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": "Image",
                "element_id": None,
                "original_filename": file_path.name,
                "section": "",
                "section_depth": 0,
                "breadcrumb": "",
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return [chunk]


    # ---- DOCX / HTML ----

    def _parse_docx(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        elements: List[Element] = partition(filename=str(file_path), strategy="fast")
        return self._walk_elements_with_tracker(elements, file_path, source_name)

    def _parse_html(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        elements: List[Element] = partition(filename=str(file_path), strategy="fast")
        return self._walk_elements_with_tracker(elements, file_path, source_name)

    def _walk_elements_with_tracker(
        self,
        elements: List[Element],
        file_path: Path,
        source_name: str,
    ) -> List[ParsedChunk]:
        """
        Generic heading-tracker walk for formats where we have unstructured
        elements but no dedicated parser (DOCX, HTML).
        Identical logic to _parse_pdf pass 2, but without font-size fallback
        (these formats don't expose coordinate metadata via the fast strategy).
        """
        tracker = _HeadingTracker()
        chunks: List[ParsedChunk] = []

        for element in elements:
            if not isinstance(element, (Text, Title, Table)):
                continue
            if isinstance(element, Title):
                meta = getattr(element, "metadata", None)
                depth = getattr(meta, "category_depth", 1) or 1
                tracker.push(int(depth), element.text or "")

            modality = "table" if isinstance(element, Table) else "text"
            chunk = self._element_to_chunk(element, file_path, source_name, modality)
            self._attach_section_context(chunk, tracker)
            chunks.append(chunk)

        return chunks


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _element_to_chunk(
        self,
        element: Element,
        file_path: Path,
        source_name: str,
        modality: str,
    ) -> ParsedChunk:
        text = element.text or ""
        meta = getattr(element, "metadata", None)
        category = type(element).__name__

        if isinstance(element, Table):
            modality = "table"

        chunk = ParsedChunk(
            text=text,
            modality=modality,
            source_file=str(file_path),
            source_name=source_name,
            metadata={
                "category": category,
                "element_id": getattr(element, "id", None),
                "page_number": getattr(meta, "page_number", None),
                "coordinates": getattr(meta, "coordinates", None),
                "section": "",  # section/breadcrumb filled by _attach_section_context
                "section_depth": 0,
                "breadcrumb": "",
            },
        )
        if modality == "text":
            chunk.language = self._detect_language(text)
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk

    @staticmethod
    def _attach_section_context(chunk: ParsedChunk, tracker: _HeadingTracker) -> None:
        """
        Stamp the current heading-tracker state onto a chunk's metadata.
        Called immediately after creating the chunk, before appending.
        """
        chunk.metadata["section"] = tracker.section
        chunk.metadata["section_depth"] = tracker.section_depth
        chunk.metadata["breadcrumb"] = tracker.breadcrumb

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        if len(text.strip()) < 20:
            return None
        try:
            return detect_language(text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level encoding helper
# ---------------------------------------------------------------------------

def _read_with_encoding_fallback(file_path: Path, preferred: str) -> str:
    """
    Read a text file trying encodings in order.
    utf-8-sig handles Windows BOM; cp1252 handles Western European Windows files.
    latin-1 never raises; final decode with errors="replace" is the safety net.
    """
    for encoding in [preferred, "utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_path.read_bytes().decode("utf-8", errors="replace")