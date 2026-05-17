
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field



class GenerationOutput(BaseModel):

    answer: str
    citations: List[str] = Field(default_factory=list)
    faithfulness_score: Optional[float] = None
    has_conflict: bool = False


class DirectResponseOutput(BaseModel):
    answer: str
    intent: str