
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.query.understanding import ProcessedQuery
from src.generation.generator import GenerationResult


@dataclass
class RAGRunContext:
    raw_query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    auth_token: Optional[str] = None
    correlation_id: str = ""
    namespace: str = "default"
    routing_decision = None
    processed_query: Optional[ProcessedQuery] = None
    context_items: List[Any] = field(default_factory=list)
    generation_result: Optional[GenerationResult] = None
    agent_trace: List[str] = field(default_factory=list)

    def record(self, agent_name: str, event: str) -> None:
        """Append a trace entry (agent_name: event) for observability."""
        self.agent_trace.append(f"[{agent_name}] {event}")