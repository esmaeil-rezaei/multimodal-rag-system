"""Multi-turn conversational evaluation: condensation, follow-up retrieval, and session coherence."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.evaluation.retrieval_eval import LabeledQuery, RetrievalEvaluator
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.evaluation.evaluator import RAGEvaluator
    from src.query.understanding import QueryUnderstanding

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a golden multi-turn conversation."""

    query: str
    expected_answer_keywords: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expects_condensation: bool = False
    condensation_must_contain: list[str] = field(default_factory=list)
    condensation_must_not_contain: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldenConversation:
    """A golden multi-turn conversation: a fixed sequence of related turns."""

    conversation_id: str
    namespace: str = "default"
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    """Evaluation outcome for a single turn within a conversation."""

    turn_index: int
    query: str
    condensed_query: str
    condensation_expected: bool
    condensation_passed: bool
    retrieval_passed: bool
    keyword_coverage: float
    faithfulness: float
    answer_relevancy: float
    has_conflict: bool
    overall_passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class ConversationResult:
    """Aggregate result for one golden conversation."""

    conversation_id: str
    namespace: str
    num_turns: int
    turn_results: list[TurnResult]
    condensation_pass_rate: float
    retrieval_pass_rate: float
    mean_keyword_coverage: float
    session_coherence_score: float
    overall_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "namespace": self.namespace,
            "num_turns": self.num_turns,
            "condensation_pass_rate": round(self.condensation_pass_rate, 4),
            "retrieval_pass_rate": round(self.retrieval_pass_rate, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "session_coherence_score": round(self.session_coherence_score, 4),
            "overall_passed": self.overall_passed,
            "turns": [asdict(t) for t in self.turn_results],
        }


@dataclass
class MultiTurnEvalReport:
    """Aggregate report across the golden conversation suite."""

    run_id: str
    timestamp: str
    num_conversations: int
    num_turns: int
    pass_rate: float
    mean_condensation_pass_rate: float
    mean_retrieval_pass_rate: float
    mean_keyword_coverage: float
    mean_session_coherence: float
    per_conversation: list[ConversationResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  MULTI-TURN EVALUATION  [{self.run_id}]",
            f"  {self.timestamp}   conversations={self.num_conversations} turns={self.num_turns}",
            "=" * 60,
            f"  Pass Rate                    {self.pass_rate:.3f}",
            f"  Condensation Pass Rate       {self.mean_condensation_pass_rate:.3f}",
            f"  Follow-up Retrieval Pass     {self.mean_retrieval_pass_rate:.3f}",
            f"  Mean Keyword Coverage        {self.mean_keyword_coverage:.3f}",
            f"  Mean Session Coherence       {self.mean_session_coherence:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "multi_turn",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "num_conversations": self.num_conversations,
            "num_turns": self.num_turns,
            "pass_rate": round(self.pass_rate, 4),
            "mean_condensation_pass_rate": round(self.mean_condensation_pass_rate, 4),
            "mean_retrieval_pass_rate": round(self.mean_retrieval_pass_rate, 4),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 4),
            "mean_session_coherence": round(self.mean_session_coherence, 4),
            "per_conversation": [c.to_dict() for c in self.per_conversation],
        }


def load_golden_conversations(path: str) -> list[GoldenConversation]:
    """Load golden multi-turn conversations from a JSON file."""
    p = Path(path)
    if not p.exists():
        logger.error("Golden conversations file not found: %s", path)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    conversations: list[GoldenConversation] = []
    for item in raw:
        turns = [
            ConversationTurn(
                query=t["query"],
                expected_answer_keywords=t.get("expected_answer_keywords", []),
                relevant_chunk_ids=t.get("relevant_chunk_ids", []),
                expects_condensation=t.get("expects_condensation", False),
                condensation_must_contain=t.get("condensation_must_contain", []),
                condensation_must_not_contain=t.get("condensation_must_not_contain", []),
                metadata=t.get("metadata", {}),
            )
            for t in item.get("turns", [])
        ]
        conversations.append(
            GoldenConversation(
                conversation_id=item["conversation_id"],
                namespace=item.get("namespace", "default"),
                turns=turns,
                metadata=item.get("metadata", {}),
            )
        )
    logger.info(
        "Loaded %d golden conversations (%d turns total) from %s",
        len(conversations),
        sum(len(c.turns) for c in conversations),
        path,
    )
    return conversations


def _word_present(text: str, phrase: str) -> bool:
    """Whole-word/phrase, case-insensitive containment check."""
    pattern = r"\b" + re.escape(phrase.strip().lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def _check_condensation(
    condensed_query: str,
    must_contain: list[str],
    must_not_contain: list[str],
) -> bool:
    for phrase in must_contain:
        if not _word_present(condensed_query, phrase):
            return False
    for phrase in must_not_contain:
        if _word_present(condensed_query, phrase):
            return False
    return True


def _check_keywords(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


class MultiTurnEvaluator:

    def __init__(
        self,
        query_understanding: QueryUnderstanding,
        retrieval_evaluator: RetrievalEvaluator,
        rag_evaluator: RAGEvaluator | None = None,
        score_ledger_path: str | None = None,
        keyword_pass_threshold: float = 0.6,
        k: int = 10,
    ) -> None:
        self._qu = query_understanding
        self._ret_eval = retrieval_evaluator
        self._rag_eval = rag_evaluator
        self._ledger_path = Path(score_ledger_path) if score_ledger_path else None
        self._kw_threshold = keyword_pass_threshold
        self._k = k

    async def evaluate_turn(
        self,
        turn: ConversationTurn,
        history: list[dict[str, str]],
        namespace: str,
        orchestrator=None,
    ) -> TurnResult:
        """Evaluate a single turn given the conversation history accumulated so far."""
        try:
            condensed_query = self._qu.condense_with_history(turn.query, history)
        except Exception as exc:
            logger.error(
                "condense_with_history failed for '%s': %s", turn.query[:60], exc, exc_info=True
            )
            condensed_query = turn.query

        condensation_passed = True
        if turn.expects_condensation:
            condensation_passed = _check_condensation(
                condensed_query, turn.condensation_must_contain, turn.condensation_must_not_contain
            )

        retrieval_passed = True
        if turn.relevant_chunk_ids:
            labeled = LabeledQuery(
                query=condensed_query,
                relevant_chunk_ids=turn.relevant_chunk_ids,
                namespace=namespace,
            )
            ret_result = self._ret_eval.evaluate_query(labeled, self._k)
            retrieval_passed = ret_result.hit

        kw_cov = 1.0
        faithfulness = 0.0
        answer_relevancy = 0.0
        has_conflict = False
        details: dict[str, Any] = {"condensed_query": condensed_query}

        if orchestrator is not None:
            try:
                gen_result = await orchestrator.run(
                    raw_query=turn.query,
                    conversation_history=history,
                    namespace=namespace,
                )
                answer = gen_result.answer
                has_conflict = bool(getattr(gen_result, "has_conflict", False))
                kw_cov = _check_keywords(answer, turn.expected_answer_keywords)
                details["answer_preview"] = answer[:200]

                if self._rag_eval is not None and turn.relevant_chunk_ids:
                    context_texts = [
                        c.get("text", "") for c in (gen_result.citations or []) if c.get("text")
                    ]
                    if context_texts:
                        eval_report = self._rag_eval.evaluate_with_ragas(
                            query=turn.query,
                            answer=answer,
                            context_texts=context_texts,
                        )
                        faithfulness = eval_report.generation.faithfulness
                        answer_relevancy = eval_report.generation.answer_relevancy

                history.append({"role": "user", "content": turn.query})
                history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                logger.error(
                    "Orchestrator call failed for '%s': %s", turn.query[:60], exc, exc_info=True
                )
        else:
            history.append({"role": "user", "content": turn.query})
            history.append(
                {"role": "assistant", "content": " ".join(turn.expected_answer_keywords)}
            )

        overall_passed = condensation_passed and retrieval_passed and kw_cov >= self._kw_threshold

        return TurnResult(
            turn_index=len(history) // 2 - 1,
            query=turn.query,
            condensed_query=condensed_query,
            condensation_expected=turn.expects_condensation,
            condensation_passed=condensation_passed,
            retrieval_passed=retrieval_passed,
            keyword_coverage=kw_cov,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            has_conflict=has_conflict,
            overall_passed=overall_passed,
            details=details,
        )

    async def evaluate_conversation(
        self,
        conversation: GoldenConversation,
        orchestrator=None,
    ) -> ConversationResult:
        history: list[dict[str, str]] = []
        turn_results: list[TurnResult] = []

        for turn in conversation.turns:
            result = await self.evaluate_turn(
                turn, history, conversation.namespace, orchestrator=orchestrator
            )
            turn_results.append(result)

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        condensation_checked = [t for t in turn_results if t.condensation_expected]
        condensation_pass_rate = (
            _mean([float(t.condensation_passed) for t in condensation_checked])
            if condensation_checked
            else 1.0
        )
        retrieval_pass_rate = _mean([float(t.retrieval_passed) for t in turn_results])
        mean_kw_coverage = _mean([t.keyword_coverage for t in turn_results])

        conflict_rate = _mean([float(t.has_conflict) for t in turn_results])
        session_coherence_score = 1.0 - conflict_rate

        overall_passed = all(t.overall_passed for t in turn_results)

        return ConversationResult(
            conversation_id=conversation.conversation_id,
            namespace=conversation.namespace,
            num_turns=len(turn_results),
            turn_results=turn_results,
            condensation_pass_rate=condensation_pass_rate,
            retrieval_pass_rate=retrieval_pass_rate,
            mean_keyword_coverage=mean_kw_coverage,
            session_coherence_score=session_coherence_score,
            overall_passed=overall_passed,
        )

    async def run_suite(
        self,
        conversations: list[GoldenConversation],
        orchestrator=None,
    ) -> MultiTurnEvalReport:
        """Run the full multi-turn golden suite and return an aggregated report."""
        if not conversations:
            raise ValueError("conversations must not be empty.")

        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        timestamp = datetime.now(timezone.utc).isoformat()

        per_conversation: list[ConversationResult] = []
        for conv in conversations:
            result = await self.evaluate_conversation(conv, orchestrator=orchestrator)
            per_conversation.append(result)
            status = "PASS" if result.overall_passed else "FAIL"
            logger.debug(
                "[%s] %s  cond=%.2f retr=%.2f kw=%.2f coherence=%.2f",
                conv.conversation_id,
                status,
                result.condensation_pass_rate,
                result.retrieval_pass_rate,
                result.mean_keyword_coverage,
                result.session_coherence_score,
            )

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        report = MultiTurnEvalReport(
            run_id=run_id,
            timestamp=timestamp,
            num_conversations=len(per_conversation),
            num_turns=sum(c.num_turns for c in per_conversation),
            pass_rate=_mean([float(c.overall_passed) for c in per_conversation]),
            mean_condensation_pass_rate=_mean([c.condensation_pass_rate for c in per_conversation]),
            mean_retrieval_pass_rate=_mean([c.retrieval_pass_rate for c in per_conversation]),
            mean_keyword_coverage=_mean([c.mean_keyword_coverage for c in per_conversation]),
            mean_session_coherence=_mean([c.session_coherence_score for c in per_conversation]),
            per_conversation=per_conversation,
        )

        logger.info(report.summary())
        self._append_to_ledger(report.to_dict())
        return report

    def _append_to_ledger(self, record: dict[str, Any]) -> None:
        if not self._ledger_path:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Multi-turn report appended to ledger: %s", self._ledger_path)
