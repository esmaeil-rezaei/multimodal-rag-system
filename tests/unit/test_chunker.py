from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sentence_transformers", reason="full ML dependency stack not installed")
pytest.importorskip("sklearn", reason="full ML dependency stack not installed")

from src.ingestion.chunker import TextChunker, _extract_heading, _heading_depth  # noqa: E402
from src.ingestion.parser import ParsedChunk  # noqa: E402


@pytest.fixture()
def chunker() -> TextChunker:
    return TextChunker()




def test_split_sentences_basic():
    text = "This is sentence one. This is sentence two. Is this three?"
    sentences = TextChunker._split_sentences(text)
    assert sentences == [
        "This is sentence one.",
        "This is sentence two.",
        "Is this three?",
    ]


def test_split_sentences_handles_abbreviations():
    text = "Dr. Smith examined the patient. The results were normal."
    sentences = TextChunker._split_sentences(text)
    # "Dr." must NOT be treated as a sentence boundary
    assert sentences[0].startswith("Dr. Smith examined the patient.")
    assert sentences[-1] == "The results were normal."


def test_split_sentences_preserves_markdown_tables():
    text = "Intro paragraph.\n\n| Col A | Col B |\n|---|---|\n| 1 | 2 |\n\nOutro paragraph."
    sentences = TextChunker._split_sentences(text)
    table_parts = [s for s in sentences if s.startswith("|")]
    assert table_parts, "markdown table rows should be preserved as their own segment(s)"


def test_split_sentences_empty_text_returns_text():
    assert TextChunker._split_sentences("") == [""]




def test_split_sections_by_markdown_headings():
    text = "# Title\nIntro text.\n## Section A\nContent A.\n## Section B\nContent B."
    sections = TextChunker._split_sections(text)
    assert len(sections) == 3
    assert sections[0].startswith("# Title")
    assert sections[1].startswith("## Section A")
    assert sections[2].startswith("## Section B")


def test_split_sections_falls_back_to_blank_line_paragraphs():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    sections = TextChunker._split_sections(text)
    assert sections == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_extract_heading_and_depth():
    section = "## My Section Title\nSome content."
    assert _extract_heading(section) == "My Section Title"
    assert _heading_depth(section) == 2


def test_extract_heading_no_heading_returns_empty():
    section = "Just a plain paragraph with no heading."
    assert _extract_heading(section) == ""
    assert _heading_depth(section) == 0




def test_build_windows_window_size_2():
    sentences = ["A.", "B.", "C.", "D."]
    windows = TextChunker._build_windows(sentences, window_size=2)
    assert windows == ["A. B.", "B. C.", "C. D.", "D."]


def test_adjacent_similarities_identical_embeddings_are_one():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    sims = TextChunker._adjacent_similarities(embeddings)
    assert len(sims) == 2
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(1.0)


def test_adjacent_similarities_orthogonal_embeddings_are_zero():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sims = TextChunker._adjacent_similarities(embeddings)
    assert sims[0] == pytest.approx(0.0)


def test_detect_breakpoints_empty_similarities_returns_empty():
    assert TextChunker._detect_breakpoints(np.array([]), percentile=0.25) == []


def test_detect_breakpoints_finds_low_similarity_indices():
    similarities = np.array([0.9, 0.9, 0.1, 0.9])
    breakpoints = TextChunker._detect_breakpoints(similarities, percentile=0.5)
    assert 2 in breakpoints




def test_sentences_to_segments_no_breakpoints_returns_single_segment():
    sentences = ["A.", "B.", "C."]
    segments = TextChunker._sentences_to_segments(sentences, [])
    assert segments == ["A. B. C."]


def test_sentences_to_segments_splits_at_breakpoints():
    sentences = ["A.", "B.", "C.", "D."]
    # breakpoint after index 1 -> [A. B.] | [C. D.]
    segments = TextChunker._sentences_to_segments(sentences, [1])
    assert segments == ["A. B.", "C. D."]


def test_merge_small_segments_merges_undersized_into_next(chunker: TextChunker):
    # "Hi." is well under any reasonable min_size in tokens
    segments = ["Hi.", "This is a much longer following segment with many words in it."]
    merged = chunker._merge_small_segments(segments, min_size=5)
    assert len(merged) == 1
    assert merged[0].startswith("Hi.")


def test_merge_small_segments_keeps_segments_above_threshold(chunker: TextChunker):
    segments = [
        "This is a reasonably long first segment with several words.",
        "This is a reasonably long second segment with several words.",
    ]
    merged = chunker._merge_small_segments(segments, min_size=1)
    assert merged == segments


def test_split_large_segments_keeps_small_segment_intact(chunker: TextChunker):
    source = ParsedChunk(text="irrelevant", source_name="doc.md")
    segments = ["A short segment."]
    result = chunker._split_large_segments(segments, chunk_size=512, source=source)
    assert result == segments


def test_split_large_segments_splits_oversized_segment(chunker: TextChunker):
    source = ParsedChunk(text="irrelevant", source_name="doc.md")
    long_sentence = "This is a moderately long sentence used for testing. " * 50
    result = chunker._split_large_segments([long_sentence], chunk_size=20, source=source)
    assert len(result) > 1
