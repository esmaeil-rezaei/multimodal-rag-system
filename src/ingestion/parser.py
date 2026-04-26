

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """
    text: str
    metadata: Dict[str, Any]       = field(default_factory=dict)
    chunk_id: Optional[str]        = None
    source_file: Optional[str]     = None
    source_name: Optional[str]     = None
    modality: str                  = "text"     # text | table | image_caption
    language: Optional[str]        = None
    doc_version: Optional[str]     = None
    ingestion_ts: Optional[str]    = None

    def compute_fingerprint(self) -> str:
        """
        SHA-256 of the text content.
        Always call this AFTER any text mutation (e.g. PII redaction) so the
        fingerprint matches what is actually stored and indexed.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
    

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DocumentParser:
    """
    Routes source files to the correct parsing strategy and returns a flat
    list of ParsedChunk objects — one per logical document element.

    All chunks at this stage are small (one element each).  The consolidator
    stage that follows is responsible for grouping them into section-sized
    units before they reach the chunker.
    """

    def __init__(self) -> None:
        self._cfg          = get_config()
        self._sec          = get_secrets()
        self._ingest_cfg   = self._cfg.ingestion
        self._kb_cfg       = self._cfg.knowledge_base
        self._openai        = openai.OpenAI(api_key=self._sec.open_ai_key)


    def parse_file(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse a single file into a list of ParsedChunk objects.
        Returns one chunk per logical element (title, paragraph, table, image).
        """
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
        

    # ----- PDF -----
    def _parse_pdf(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse PDF with unstructured.io for text/images and Camelot for tables.

        FIX: Camelot extraction is called ONCE per file, not once per element.
        Original had the camelot block inside the `for element in elements` loop,
        which ran a full Camelot pass for every element and duplicated every table.
        """
        chunks: List[ParsedChunk] = []
        parsing_cfg = self._ingest_cfg["parsing"]

        elements: List[Element] = partition_pdf(
            filename=str(file_path),
            strategy=parsing_cfg["pdf_strategy"],
            languages=parsing_cfg["ocr_languages"],
            extract_images_in_pdf=parsing_cfg["extract_images"],
            extract_image_block_output_dir=parsing_cfg["image_output_dir"],
        )

        for element in elements:
            if isinstance(element, (Text, Title)):
                chunks.append(self._element_to_chunk(element, file_path, source_name, "text"))

            elif isinstance(element, Table):
                # Unstructured-extracted table (plain text repr); Camelot provides richer version
                chunks.append(self._element_to_chunk(element, file_path, source_name, "table"))
            
            elif isinstance(element, Image):
                if self._ingest_cfg["image_captioning"]["enabled"]:
                    caption = self._caption_image(element, file_path, source_name)
                    if caption:
                        chunks.append(caption)

        if self._ingest_cfg["tables"]["extract_tables"]:
            chunks.extend(self._extract_tables_camelot(file_path, source_name))

        return chunks
    

    # ----- Markdown -----
    def _parse_md(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse Markdown using unstructured's dedicated partition_md.

        partition_md emits one Element per logical unit — Title, NarrativeText,
        Table, ListItem — exactly like partition_pdf does for PDFs.  This gives
        the consolidator consistent, section-grouped input for all file types.
        """
        text = _read_with_encoding_fallback(file_path, self._kb_cfg.get("encoding", "utf-8"))
        elements: List[Element] = partition_md(text=text)
        
        chunks: List[ParsedChunk] = []
        current_section: str = ""       # Track the most recently seen heading
        current_depth: int = 0
        
        for element in elements:
            # Update section tracker whenever we encounter a heading (Title)
            # Heading 1 → depth 1 Heading 2 → depth 2
            if isinstance(element, Title):
                current_section = element.text or ""
                current_depth = getattr(
                    getattr(element, "metadata", None), "category_depth", 1
                ) or 1

            chunk = self._element_to_chunk(element, file_path, source_name, "text")

            chunk.metadata["section"] = current_section
            chunk.metadata["section_depth"] = current_depth

            if isinstance(element, Table):
                chunk.modality = "table"

            chunks.append(chunk)

        return chunks

    # ----- Plain text -----
    def _parse_txt(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Parse plain .txt files.
        No heading structure available — return one chunk per double-newline
        paragraph so the consolidator receives paragraph-sized units.
        """
        text = _read_with_encoding_fallback(file_path, self._kb_cfg.get("encoding", "utf-8"))
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[ParsedChunk] = []
        for i, para in enumerate(paragraphs):
            chunk = ParsedChunk(
                text=para,
                modality="text",
                source_file=str(file_path),
                source_name=source_name,
                metadata={
                    "section": "",
                    "section_depth": 0,
                    "category": "NarrativeText",
                    "paragraph_index": i,
                },
            )
            chunk.language = self._detect_language(para)
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks



    # ----- Table extraction -----
    def _extract_tables_camelot(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Extract tables from a PDF using Camelot.
        Called once per file (not per element).
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
            # First row as column headers; handle multi-line merged headers
            raw_header = df.iloc[0]
            col_metadata = [
                col.split("\n")[0] if "\n" in str(col) else str(col)
                for col in raw_header
            ]

            if output_fmt == "json":
                records = df.iloc[1:].to_dict(orient="records")
                text_repr = str(records)
            elif output_fmt == "markdown":
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
                    "table_index": i,
                    "page": table.page,
                    "columns": col_metadata,
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace,
                    "section": "",      # Camelot has no section context; consolidator will fill
                },
            )
            chunk.chunk_id = chunk.compute_fingerprint()
            chunks.append(chunk)

        return chunks


    # ----- Image captioning  -----
    def _caption_image(
        self, element: Image, file_path: Path, source_name: str
    ) -> Optional[ParsedChunk]:
        """
        Caption an extracted image via GPT-4V.
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
                "section": "",
                "has_clip_embedding": False,
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk


    # ----- Standalone image files  -----
    def _parse_standalone_image(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        """
        Handle .png / .jpg files uploaded directly to the knowledge base.
        """

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
                "original_filename": file_path.name,
                "section": "",
            },
        )
        chunk.chunk_id = chunk.compute_fingerprint()
        return [chunk]


    # ----- DOCX / HTML  -----
    def _parse_docx(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        elements: List[Element] = partition(filename=str(file_path), strategy="fast")
        return [
            self._element_to_chunk(el, file_path, source_name, "text")
            for el in elements
            if isinstance(el, (Text, Title, Table))
        ]

    def _parse_html(self, file_path: Path, source_name: str) -> List[ParsedChunk]:
        elements: List[Element] = partition(filename=str(file_path), strategy="fast")
        return [
            self._element_to_chunk(el, file_path, source_name, "text")
            for el in elements
            if isinstance(el, (Text, Title, Table))
        ]



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
        """
        Convert a single unstructured Element into a ParsedChunk.

        It attaches category and element_id to metadata in addition to
        page_number and coordinates.  The consolidator needs category to know
        how to assemble the merged text (Title → heading prefix, Table → atomic,
        Image → sentinel wrap) and element_id for full provenance tracking.
        """
        text = element.text or ""
        meta = getattr(element, "metadata", None)
        category = type(element).__name__   # "Title", "NarrativeText", "Table", etc.

        # Override modality for table elements regardless of caller's default
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
                "section": "",          # Will be filled by _parse_md or consolidator
                "section_depth": 0,
            },
        )
 
        if modality == "text":
            chunk.language = self._detect_language(text)
        chunk.chunk_id = chunk.compute_fingerprint()
        return chunk

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        """Detect ISO 639-1 language code. Returns None on short or ambiguous text."""
        if len(text.strip()) < 20:
            return None
        try:
            return detect_language(text)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level encoding helper (used by _parse_md and _parse_txt)
# ---------------------------------------------------------------------------

def _read_with_encoding_fallback(file_path: Path, preferred: str) -> str:
    """
    Read a text file trying encodings in order of specificity.

      - utf-8-sig handles Windows Notepad BOM prefix silently
      - cp1252 handles Western European Windows files correctly
        (latin-1 accepts the same bytes but maps them differently for 0x80-0x9F)
      - latin-1 remains the last resort: it never raises but may mis-render
      - final decode(..., errors="replace") ensures we never crash on truly
        binary files; question marks are better than a pipeline halt
    """
    for encoding in [preferred, "utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # Absolute last resort — replace undecodable bytes with U+FFFD
    return file_path.read_bytes().decode("utf-8", errors="replace")