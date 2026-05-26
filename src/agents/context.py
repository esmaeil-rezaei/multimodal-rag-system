from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.query.understanding import ProcessedQuery
from src.generation.generator import GenerationResult

if TYPE_CHECKING:
    from src.core.container import AppContainer

@dataclass
class RAGRunContext:
    raw_query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    auth_token: Optional[str] = None
    correlation_id: str = ""
    namespace: str = "default"
    # Populated by prepare_query tool; may be a plain str before that runs
    processed_query: Optional[Any] = None
    context_items: List[Any] = field(default_factory=list)
    generation_result: Optional[GenerationResult] = None
    agent_trace: List[str] = field(default_factory=list)
    # Injected from orchestrator — tools read deps from here, no globals
    container: Optional["AppContainer"] = field(default=None, repr=False)
    # Internal: cached query vector for cache-store step
    _query_vector: Optional[Any] = field(default=None, repr=False)
    # Internal: evaluator reused between retrieve and generate steps
    _evaluator: Optional[Any] = field(default=None, repr=False)

    @property
    def query_routing_intent(self) -> str:
        if isinstance(self.processed_query, ProcessedQuery):
            return self.processed_query.query_routing_intent
        return "retrieval"

    def record(self, agent_name: str, event: str) -> None:
        self.agent_trace.append(f"[{agent_name}] {event}")
