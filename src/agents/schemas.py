
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

class CitationItem(BaseModel):
    number: Optional[int] = None
    chunk_id: Optional[str] = None
    source_file: Optional[str] = None
    source_name: Optional[str] = None
    ingestion_ts: Optional[str] = None
    excerpt: Optional[str] = None

class GenerationOutput(BaseModel):

    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    faithfulness_score: Optional[float] = None
    has_conflict: bool = False

class DirectResponseOutput(BaseModel):
    answer: str
    intent: str
