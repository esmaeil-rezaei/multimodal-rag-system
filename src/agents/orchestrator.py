from __future__ import annotations

import asyncio
import json
import random
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    set_default_openai_key,
)
from src.agents.agents import OrchestratorAgent
from src.agents.context import RAGRunContext
from src.agents.schemas import DirectResponseOutput, GenerationOutput
from src.query.understanding import ProcessedQuery
from src.config.settings import get_config, get_secrets
from src.evaluation.cost_latency_eval import CostLatencyEvaluator, LatencyTimer
from src.evaluation.fairness_eval import FairnessEvaluator
from src.generation.generator import GenerationResult
from src.utils.logger import get_logger, set_correlation_id

if TYPE_CHECKING:
    from src.core.container import AppContainer

logger = get_logger(__name__)
_cfg = get_config()
_sec = get_secrets()

_PREF_DIR = Path(_cfg.log.get("preferences_dir", "logs/user_preferences"))

_DETECT_EXTRACT_PROMPT = """\
You are a behavioral instruction detector for an AI research assistant.

Classify the user message and return EXACTLY ONE of these formats:

1. Pure behavioral instruction — applies going forward only, no immediate action:
   PREF: <concise imperative sentence>
   Use when the user sets a general preference for future responses.
   Examples: "be concise" → PREF: Be concise.

2. Behavioral instruction + re-apply to the PREVIOUS answer immediately:
   PREF: <instruction> | REDO
   Use when the instruction is about HOW the last answer was presented and the user
   clearly wants it redone (e.g. they say "now", "for the answer", "show them",
   "return them", "make it shorter", "translate it").
   Examples: "show citations now" → PREF: Show citations. | REDO
             "return citations for the answer" → PREF: Show citations. | REDO
             "make it shorter" → PREF: Be concise. | REDO
             
3. Behavioral instruction COMBINED with a new question to answer:
   PREF: <instruction> | QUERY: <standalone question>
   Use when the message contains BOTH a preference AND a distinct new question.
   Examples: "answer in bullet points. what is the methodology" →
             PREF: Answer in bullet points. | QUERY: What is the methodology?
             "cite the context. what are the findings" →
             PREF: Include citations in answers. | QUERY: What are the findings?

4. Reset / clear all preferences:
   CLEAR
   Use for: "forget everything I said", "reset your instructions", "start fresh", \
"clear your memory", "ignore all previous preferences".

5. Regular question, greeting, or follow-up — NO behavioral instruction:
   NULL
   Examples: "ok", "thanks", "got it", "what is MCI?" → NULL

Rules:
- A behavioral instruction tells the assistant HOW to behave.
- A regular question asks for information.
- "ok", "thanks", "got it", "great" alone → NULL
- Return ONLY the classified output — no explanation, no prose."""


def _pref_path(namespace: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", namespace)
    return _PREF_DIR / f"{safe}.json"


def _load_preferences(namespace: str) -> list[str]:
    path = _pref_path(namespace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not load preferences for namespace '%s': %s", namespace, exc)
        return []


def _save_preferences(namespace: str, preferences: list[str]) -> None:
    path = _pref_path(namespace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preferences, indent=2, ensure_ascii=False))
        logger.info("Preferences saved → %s  contents=%s", path, preferences)
    except Exception as exc:
        logger.warning("Could not save preferences for namespace '%s': %s", namespace, exc)


class _PrefDetection:
    __slots__ = ("preference", "follow_on_query", "redo", "clear_all")

    def __init__(
        self,
        preference: str | None = None,
        follow_on_query: str | None = None,
        redo: bool = False,
        clear_all: bool = False,
    ) -> None:
        self.preference = preference
        self.follow_on_query = follow_on_query
        self.redo = redo
        self.clear_all = clear_all

    @property
    def has_action(self) -> bool:
        return bool(self.preference or self.clear_all)


# ---------------------------------------------------------------------------
# Fast local preference classifier — zero LLM calls for the common cases.
# Saves ~400 ms per query whenever the result is unambiguous.
# ---------------------------------------------------------------------------
_FAST_CLEAR_RE = re.compile(
    r"\b("
    r"forget (?:everything|all|all preferences)|"
    r"reset (?:your |all )?(?:instructions?|preferences?|settings?)|"
    r"start fresh|"
    r"clear (?:your |all )?(?:memory|preferences?|instructions?)|"
    r"ignore all (?:previous |prior )?preferences?"
    r")\b",
    re.IGNORECASE,
)
_FAST_NULL_RE = re.compile(
    r"^(?:ok|okay|thanks?|thank you|cool|great|nice|sure|interesting|"
    r"got it|i see|makes sense|sounds good|no problem|"
    r"hello|hi|hey|good|alright|perfect|wonderful|awesome)\s*[.!?]?$",
    re.IGNORECASE,
)
_FAST_QUESTION_RE = re.compile(
    r"^(?:what|who|why|how|when|where|which|is |are |does |do |can |could |would |"
    r"should |explain |describe |tell me |show |find |list |give |help |"
    r"summarize |summarise |compare )",
    re.IGNORECASE,
)


def _fast_detect_preference(query: str) -> _PrefDetection | None:
    """
    Zero-LLM-call classifier.  Returns a _PrefDetection for unambiguous cases
    (clear CLEAR signals, obvious small-talk, obvious questions) and ``None``
    when the result is ambiguous enough to warrant an LLM call.
    """
    q = query.strip()
    if _FAST_CLEAR_RE.search(q):
        return _PrefDetection(clear_all=True)
    if _FAST_NULL_RE.match(q):
        return _PrefDetection()  # NULL — no preference action
    # Long query that starts with a question word → almost certainly retrieval
    if len(q.split()) >= 5 and _FAST_QUESTION_RE.match(q):
        return _PrefDetection()  # NULL
    return None  # ambiguous — fall through to LLM


def _pref_ack(extracted_pref: str, updated_prefs: list[str]) -> str:
    """Human-readable acknowledgment for a stored / cleared preference."""
    if extracted_pref == "__CLEARED__":
        return "Done — all your formatting preferences have been cleared."
    n = len(updated_prefs)
    plural = "preferences" if n != 1 else "preference"
    return (
        f"Got it! I've saved that preference "
        f"({n} {plural} active). It will apply to all future answers."
    )


def _detect_preference(query: str, openai_client) -> _PrefDetection:
    # Fast path — skip LLM for unambiguous cases
    _fast = _fast_detect_preference(query)
    if _fast is not None:
        logger.debug("Preference fast-path returned %s", _fast.__class__.__name__)
        return _fast

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _DETECT_EXTRACT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Preference detection failed (%s); skipping.", exc)
        return _PrefDetection()

    if not raw or raw.upper() == "NULL":
        return _PrefDetection()

    if raw.upper() == "CLEAR":
        return _PrefDetection(clear_all=True)

    if raw.upper().startswith("PREF:"):
        body = raw[5:].strip()
        if "|" in body:
            parts = body.split("|", 1)
            pref = parts[0].strip()
            suffix = parts[1].strip()
            if suffix.upper() == "REDO":
                return _PrefDetection(preference=pref, redo=True)
            if suffix.upper().startswith("QUERY:"):
                suffix = suffix[6:].strip()
            return _PrefDetection(preference=pref, follow_on_query=suffix or None)
        return _PrefDetection(preference=body)

    return _PrefDetection(preference=raw)


_RECONCILE_PROMPT = """\
You manage a persistent list of behavioral instructions for an AI assistant.

Existing instructions (0-indexed):
{existing}

New instruction: "{new}"

Choose ONE action:
- REPLACE N  — the new instruction directly contradicts or supersedes instruction N \
(e.g. "Show citations" supersedes "Do not show citations"). N is the 0-based index.
- ADD         — the new instruction is genuinely new and does not conflict with any existing one.
- SKIP        — the new instruction is already covered by an existing one (duplicate or subset).

Return ONLY one of these exact strings: REPLACE <N>, ADD, or SKIP. Nothing else."""


def _reconcile_preferences(new_pref: str, existing: list[str], openai_client) -> list[str]:
    if not existing:
        return [new_pref]

    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(existing))
    prompt = _RECONCILE_PROMPT.format(existing=numbered, new=new_pref)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        decision = (response.choices[0].message.content or "").strip().upper()

        if decision == "SKIP":
            return existing
        if decision.startswith("REPLACE"):
            parts = decision.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(existing):
                    updated = list(existing)
                    updated[idx] = new_pref
                    logger.info(
                        "Preference reconciliation: replaced index %d ('%s') with '%s'",
                        idx,
                        existing[idx],
                        new_pref,
                    )
                    return updated
    except Exception as exc:
        logger.warning("Preference reconciliation failed (%s); appending.", exc)

    return existing + [new_pref]


async def _timed_status_ticker(queue: asyncio.Queue) -> None:
    """
    Background task: push pre-programmed retrieval/reasoning status messages
    into *queue* at realistic intervals.  Cancelled when the agent finishes.
    Timings are tuned to the observed pipeline stages (query expansion ~3 s,
    embedding ~3 s, retrieval ~0.5 s, inter-tool LLM routing ~5 s per hop).
    """
    steps = [
        (3.5,  "Expanding and decomposing sub-questions..."),
        (7.5,  "Building semantic search vectors..."),
        (11.5, "Searching and re-ranking documents..."),
        (16.0, "Reasoning over retrieved context..."),
        (22.0, "Synthesising evidence from sources..."),
    ]
    for delay, msg in steps:
        await asyncio.sleep(delay)
        await queue.put(msg)


class RAGOrchestrator:

    def __init__(self, container: AppContainer) -> None:
        self._container = container
        self._pii_guard = container.pii_guard
        self._acl = container.acl
        self._qu = container.query_understanding
        self._max_turns = _cfg.query.get("agents", {}).get("max_turns", 15)

        set_default_openai_key(_sec.openai_api_key)

        self._agent = OrchestratorAgent
        self._cl_evaluator = CostLatencyEvaluator(
            score_ledger_path=_cfg.log.get("cost_latency_ledger", "logs/cost_latency.jsonl")
        )
        self._fairness_evaluator = FairnessEvaluator(
            retrieval_evaluator=container.retrieval_evaluator,
            score_ledger_path=_cfg.log.get("fairness_ledger", "logs/fairness.jsonl"),
        )
        self._fairness_sample_rate: float = _cfg.evaluation.get("fairness", {}).get(
            "live_sample_rate", 0.05
        )

    async def run(
        self,
        raw_query: str,
        auth_token: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        namespace: str | None = None,
    ) -> GenerationResult:

        correlation_id = set_correlation_id()
        timer = LatencyTimer()
        logger.info(
            "RAGOrchestrator.run started",
            extra={"query": raw_query[:120], "correlation_id": correlation_id},
        )

        if namespace is None:
            namespace = "default"
            try:
                claims = self._acl.authenticate(auth_token or "")
                namespace = self._acl.get_namespace(claims)
            except Exception as exc:
                logger.warning("ACL auth failed (%s); using default namespace.", exc)

        user_preferences = _load_preferences(namespace)
        openai_client = self._container.generator._openai

        extracted_pref: str | None = None
        effective_query = raw_query

        # Run routing classification, history compression, and preference detection
        # all concurrently.  get_routing_intent is now LLM-based (gpt-4o-mini,
        # max_tokens=5, ~300 ms) but overlaps with the other two, adding zero
        # wall-clock latency.
        base_history, detection, routing_intent = await asyncio.gather(
            asyncio.to_thread(self._qu.compress_history, list(conversation_history or [])),
            asyncio.to_thread(_detect_preference, raw_query, openai_client),
            asyncio.to_thread(self._qu.get_routing_intent, raw_query, list(conversation_history or [])),
        )

        if detection.clear_all:
            if user_preferences:
                user_preferences = []
                _save_preferences(namespace, user_preferences)
                logger.info("User preferences cleared for namespace '%s'.", namespace)
            extracted_pref = "__CLEARED__"
            routing_intent = "conversational"

        elif detection.preference:
            extracted_pref = detection.preference
            updated = _reconcile_preferences(extracted_pref, user_preferences, openai_client)
            if updated != user_preferences:
                user_preferences = updated
                _save_preferences(namespace, user_preferences)

            if detection.follow_on_query:
                effective_query = detection.follow_on_query
                routing_intent = "retrieval"
                logger.info(
                    "Mixed message — routing question '%s' to retrieval.", effective_query[:80]
                )
            elif detection.redo:
                last_question = next(
                    (
                        m["content"]
                        for m in reversed(conversation_history or [])
                        if m.get("role") == "user"
                    ),
                    None,
                )
                if last_question:
                    effective_query = last_question
                    routing_intent = "retrieval"
                    logger.info("REDO — re-running previous query '%s'.", effective_query[:80])
                else:
                    routing_intent = "conversational"
            else:
                routing_intent = "conversational"

        history_with_pref = list(base_history)
        if extracted_pref == "__CLEARED__":
            history_with_pref = history_with_pref + [
                {
                    "role": "system",
                    "content": "[USER PREFERENCE RESET] All previous preferences have been cleared.",
                }
            ]
        elif extracted_pref:
            history_with_pref = history_with_pref + [
                {"role": "system", "content": f"[USER PREFERENCE STORED] {extracted_pref}"}
            ]

        with timer.stage("condense"):
            if history_with_pref:
                condensed_query = self._qu.condense_with_history(effective_query, history_with_pref)
            else:
                condensed_query = effective_query

        if condensed_query != raw_query and history_with_pref and routing_intent == "retrieval":
            routing_intent = "followup"

        ctx = RAGRunContext(
            raw_query=raw_query,
            conversation_history=history_with_pref,
            processed_query=condensed_query,  # type: ignore[arg-type]
            query_routing_intent=routing_intent,
            auth_token=auth_token,
            correlation_id=correlation_id,
            namespace=namespace,
            container=self._container,
            user_preferences=user_preferences,
        )

        run_result = None
        if routing_intent == "conversational" and extracted_pref is not None:
            # A preference was just stored/cleared and no follow-on question exists.
            # Acknowledge directly — no agent or retrieval needed.
            logger.info(
                "direct_conversational: preference acknowledged (no retrieval)",
                extra={"correlation_id": correlation_id},
            )
            return GenerationResult(
                answer=_pref_ack(extracted_pref, user_preferences),
                model_used=_cfg.generation.get("fast_model", "gpt-4o-mini"),
            )
        elif routing_intent == "conversational":
            # Direct conversational reply — no retrieval needed.
            logger.info(
                "direct_conversational path (non-stream): LLM social reply",
                extra={"correlation_id": correlation_id},
            )
            _conv_msgs: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful knowledge assistant. "
                        "Respond naturally and concisely to the user's conversational "
                        "message. Keep your reply to 1–3 sentences."
                    ),
                },
                *[
                    m for m in (history_with_pref or [])
                    if m.get("role") in ("user", "assistant")
                ][-6:],
                {"role": "user", "content": raw_query},
            ]
            try:
                _conv_resp = await asyncio.to_thread(
                    self._container.generator._openai.chat.completions.create,
                    model=_cfg.generation.get("fast_model", "gpt-4o-mini"),
                    messages=_conv_msgs,
                    temperature=0.7,
                    max_tokens=150,
                )
                _conv_answer = (
                    _conv_resp.choices[0].message.content or ""
                ).strip()
            except Exception as exc:
                logger.warning("Conversational reply failed (%s); using fallback.", exc)
                _conv_answer = "I'm here to help. Could you clarify what you're looking for?"
            try:
                _conv_answer = self._pii_guard.redact(_conv_answer, context="output")
            except Exception:
                pass
            return GenerationResult(
                answer=_conv_answer,
                model_used=_cfg.generation.get("fast_model", "gpt-4o-mini"),
            )

        elif routing_intent == "retrieval":
            # Direct pipeline: skip Agents SDK routing, call tools in fixed order.
            # Saves ~10–15 s of inter-tool LLM routing overhead.
            logger.info(
                "direct_pipeline path selected",
                extra={"correlation_id": correlation_id},
            )
            try:
                with timer.stage("agent"):
                    await self._run_direct_pipeline(ctx)
            except Exception as exc:
                logger.error(
                    "Direct pipeline error: %s",
                    exc,
                    extra={"correlation_id": correlation_id},
                    exc_info=True,
                )
                return GenerationResult(
                    answer="An internal error occurred. Please try again.",
                    model_used="error",
                )
        else:
            # Safety fallback — all three intents should be covered above.
            try:
                with timer.stage("agent"):
                    run_result = await Runner.run(
                        self._agent,
                        input=raw_query,
                        context=ctx,
                        max_turns=self._max_turns,
                    )
            except InputGuardrailTripwireTriggered as exc:
                logger.warning(
                    "Input guardrail tripped: %s", exc, extra={"correlation_id": correlation_id}
                )
                return GenerationResult(
                    answer=(
                        "Your query could not be processed. "
                        "Please check the input and try again."
                    ),
                    model_used="",
                )
            except OutputGuardrailTripwireTriggered as exc:
                logger.warning(
                    "Output guardrail tripped: %s", exc, extra={"correlation_id": correlation_id}
                )
                return GenerationResult(
                    answer=(
                        "I was unable to generate a sufficiently reliable answer "
                        "from the available sources. Please try rephrasing your question."
                    ),
                    model_used="",
                )
            except Exception as exc:
                logger.error(
                    "Agent pipeline error: %s",
                    exc,
                    extra={"correlation_id": correlation_id},
                    exc_info=True,
                )
                return GenerationResult(
                    answer="An internal error occurred. Please try again.",
                    model_used="error",
                )

        result = self._extract_result(run_result, ctx)

        # PII scan only for the agent path (direct pipeline already applied it internally)
        if run_result is not None:
            try:
                result.answer = self._pii_guard.redact(result.answer, context="output")
            except Exception as exc:
                logger.warning("Output PII scan failed (non-fatal): %s", exc)

        try:
            self._cl_evaluator.record_request(
                query=raw_query,
                namespace=namespace,
                timer=timer,
                generation_result=result,
            )
        except Exception as exc:
            logger.warning("Cost/latency recording failed (non-fatal): %s", exc)

        if random.random() < self._fairness_sample_rate:
            try:
                self._fairness_evaluator.check_live_async(raw_query, namespace)
            except Exception as exc:
                logger.warning("Fairness check failed to start (non-fatal): %s", exc)

        logger.info(
            "RAGOrchestrator.run complete",
            extra={
                "correlation_id": correlation_id,
                "pipeline": "direct" if routing_intent == "retrieval" else "agent",
                "agent_trace": ctx.agent_trace,
                "answer_length": len(result.answer),
                "latency_ms": round(timer.total_ms, 1),
            },
        )
        return result

    async def run_stream(
        self,
        raw_query: str,
        auth_token: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        namespace: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of run().  Yields SSE-formatted strings:

          data: {"event": "status",  "message": "..."}\\n\\n
          data: {"event": "token",   "content": "..."}\\n\\n   (repeated per token)
          data: {"event": "done",    "citations": [...], ...}\\n\\n
          data: {"event": "error",   "message": "..."}\\n\\n   (on failure)

        The caller wraps this in FastAPI's StreamingResponse with
        media_type="text/event-stream".
        """

        def _sse(event: str, data: dict) -> str:
            return f"data: {json.dumps({'event': event, **data})}\n\n"

        correlation_id = set_correlation_id()
        logger.info(
            "RAGOrchestrator.run_stream started",
            extra={"query": raw_query[:120], "correlation_id": correlation_id},
        )

        # ---------- namespace / ACL ----------
        if namespace is None:
            namespace = "default"
            try:
                claims = self._acl.authenticate(auth_token or "")
                namespace = self._acl.get_namespace(claims)
            except Exception as exc:
                logger.warning("ACL auth failed (%s); using default namespace.", exc)

        yield _sse("status", {"message": "Analyzing query..."})

        # ---------- preferences / routing ----------
        user_preferences = _load_preferences(namespace)
        openai_client = self._container.generator._openai

        extracted_pref: str | None = None
        effective_query = raw_query

        # Run routing classification, history compression, and preference detection
        # all concurrently.  get_routing_intent is now LLM-based (gpt-4o-mini,
        # max_tokens=5, ~300 ms) but overlaps with the other two operations so it
        # adds zero wall-clock latency.
        base_history, detection, routing_intent = await asyncio.gather(
            asyncio.to_thread(self._qu.compress_history, list(conversation_history or [])),
            asyncio.to_thread(_detect_preference, raw_query, openai_client),
            asyncio.to_thread(self._qu.get_routing_intent, raw_query, list(conversation_history or [])),
        )

        if detection.clear_all:
            if user_preferences:
                user_preferences = []
                _save_preferences(namespace, user_preferences)
            extracted_pref = "__CLEARED__"
            routing_intent = "conversational"

        elif detection.preference:
            extracted_pref = detection.preference
            updated = _reconcile_preferences(extracted_pref, user_preferences, openai_client)
            if updated != user_preferences:
                user_preferences = updated
                _save_preferences(namespace, user_preferences)

            if detection.follow_on_query:
                effective_query = detection.follow_on_query
                routing_intent = "retrieval"
            elif detection.redo:
                last_question = next(
                    (
                        m["content"]
                        for m in reversed(conversation_history or [])
                        if m.get("role") == "user"
                    ),
                    None,
                )
                if last_question:
                    effective_query = last_question
                    routing_intent = "retrieval"
                else:
                    routing_intent = "conversational"
            else:
                routing_intent = "conversational"

        history_with_pref = list(base_history)
        if extracted_pref == "__CLEARED__":
            history_with_pref = history_with_pref + [
                {
                    "role": "system",
                    "content": "[USER PREFERENCE RESET] All previous preferences have been cleared.",
                }
            ]
        elif extracted_pref:
            history_with_pref = history_with_pref + [
                {"role": "system", "content": f"[USER PREFERENCE STORED] {extracted_pref}"}
            ]

        if history_with_pref:
            condensed_query = self._qu.condense_with_history(effective_query, history_with_pref)
        else:
            condensed_query = effective_query

        if condensed_query != raw_query and history_with_pref and routing_intent == "retrieval":
            routing_intent = "followup"

        # ---------- pipeline run (retrieval deferred → stream; or agent) ----------
        yield _sse("status", {"message": "Searching the knowledge base..."})

        ctx = RAGRunContext(
            raw_query=raw_query,
            conversation_history=history_with_pref,
            processed_query=condensed_query,
            query_routing_intent=routing_intent,
            auth_token=auth_token,
            correlation_id=correlation_id,
            namespace=namespace,
            container=self._container,
            user_preferences=user_preferences,
            stream_mode=True,
        )

        if routing_intent == "conversational" and extracted_pref is not None:
            # Preference stored/cleared with no follow-on question — acknowledge
            # directly, skipping the agent entirely (prevents 10-min agent loops).
            logger.info(
                "direct_conversational: preference acknowledged (stream, no retrieval)",
                extra={"correlation_id": correlation_id},
            )
            ack = _pref_ack(extracted_pref, user_preferences)
            yield _sse("token", {"content": ack})
            yield _sse(
                "done",
                {
                    "answer": ack,
                    "citations": [],
                    "faithfulness_score": None,
                    "has_conflict": False,
                    "model_used": _cfg.generation.get("fast_model", "gpt-4o-mini"),
                    "agent_name": "Assistant",
                },
            )
            return
        elif routing_intent == "retrieval":
            # Direct pipeline: no Agents SDK overhead — saves ~10–15 s.
            logger.info(
                "direct_pipeline path selected (stream)",
                extra={"correlation_id": correlation_id},
            )
            # Emit timed status labels while the synchronous pipeline runs.
            status_q: asyncio.Queue[str] = asyncio.Queue()
            ticker = asyncio.create_task(_timed_status_ticker(status_q))
            try:
                pipeline_task = asyncio.create_task(self._run_direct_pipeline(ctx))
                while not pipeline_task.done():
                    await asyncio.sleep(0.2)
                    while not status_q.empty():
                        yield _sse("status", {"message": status_q.get_nowait()})
                pipeline_task.result()  # re-raise any exception
            except Exception as exc:
                logger.error("Direct pipeline error (stream): %s", exc, exc_info=True)
                yield _sse("error", {"message": "An internal error occurred. Please try again."})
                return
            finally:
                ticker.cancel()
        elif routing_intent == "followup" and history_with_pref:
            # ---------- Direct followup: stream from history, bypass agent ----------
            # This avoids the ~10-15s Agents SDK overhead and provides true token
            # streaming.  The model buffers the first 16 chars before yielding so
            # it can detect the NEEDS_RETRIEVAL signal before any tokens reach the
            # client.  If history is insufficient the code falls through to the
            # direct retrieval pipeline and then to Cases 1/2/3 below.
            logger.info(
                "direct_followup path selected (stream)",
                extra={"correlation_id": correlation_id},
            )
            _fup_gen_cfg = _cfg.generation
            _fup_model: str = _fup_gen_cfg.get("fast_model", "gpt-4o-mini")
            _fup_history = [
                m for m in history_with_pref if m.get("role") in ("user", "assistant")
            ][-10:]
            _fup_history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in _fup_history
            )
            _fup_question = condensed_query or raw_query
            _fup_sys = (
                "You are a precise question-answering assistant. "
                "Answer ONLY using the conversation history provided. "
                "If the history does NOT contain enough information to answer, "
                "respond with exactly: NEEDS_RETRIEVAL"
            )

            _fup_buf: list[str] = []
            _fup_streaming_started = False
            _fup_answered = False

            try:
                _fup_stream = (
                    await self._container.generator._async_openai.chat.completions.create(
                        model=_fup_model,
                        messages=[
                            {"role": "system", "content": _fup_sys},
                            {
                                "role": "user",
                                "content": (
                                    f"History:\n{_fup_history_text}\n\nQuestion: {_fup_question}"
                                ),
                            },
                        ],
                        temperature=0.0,
                        max_tokens=_fup_gen_cfg.get("max_tokens", 1024),
                        stream=True,
                    )
                )
                yield _sse("status", {"message": "Generating answer..."})
                async for _fup_chunk in _fup_stream:
                    _fup_delta = (
                        (_fup_chunk.choices[0].delta.content or "")
                        if _fup_chunk.choices
                        else ""
                    )
                    if not _fup_delta:
                        continue
                    _fup_buf.append(_fup_delta)
                    _fup_acc = "".join(_fup_buf)
                    if not _fup_streaming_started:
                        if len(_fup_acc) < 16:
                            continue  # buffer until we can rule out NEEDS_RETRIEVAL
                        if _fup_acc.strip().upper().startswith("NEEDS_RETRIEVAL"):
                            break  # fall through to retrieval pipeline below
                        _fup_streaming_started = True
                        _fup_answered = True
                        yield _sse("token", {"content": _fup_acc})  # flush buffer
                    else:
                        yield _sse("token", {"content": _fup_delta})

                # Handle short responses (total < 16 chars) that never triggered the
                # streaming-started branch above.
                if not _fup_streaming_started and _fup_buf:
                    _fup_acc = "".join(_fup_buf)
                    if not _fup_acc.strip().upper().startswith("NEEDS_RETRIEVAL"):
                        _fup_answered = True
                        yield _sse("token", {"content": _fup_acc})
            except Exception as _fup_exc:
                logger.warning(
                    "Direct followup stream failed (%s); falling back to retrieval.",
                    _fup_exc,
                )
                _fup_answered = False

            if _fup_answered:
                _fup_answer = "".join(_fup_buf).strip()
                try:
                    _fup_answer = self._pii_guard.redact(_fup_answer, context="output")
                except Exception:
                    pass
                yield _sse(
                    "done",
                    {
                        "answer": _fup_answer,
                        "citations": [],
                        "faithfulness_score": None,
                        "has_conflict": False,
                        "model_used": _fup_model,
                        "agent_name": "Follow-up Agent",
                    },
                )
                return

            # NEEDS_RETRIEVAL — run the retrieval pipeline and fall through to
            # Cases 1/2/3 below for generation.
            logger.info(
                "followup NEEDS_RETRIEVAL — switching to retrieval pipeline",
                extra={"correlation_id": correlation_id},
            )
            ctx.query_routing_intent = "retrieval"
            yield _sse("status", {"message": "Searching the knowledge base..."})
            status_q: asyncio.Queue[str] = asyncio.Queue()
            ticker = asyncio.create_task(_timed_status_ticker(status_q))
            try:
                pipeline_task = asyncio.create_task(self._run_direct_pipeline(ctx))
                while not pipeline_task.done():
                    await asyncio.sleep(0.2)
                    while not status_q.empty():
                        yield _sse("status", {"message": status_q.get_nowait()})
                pipeline_task.result()
            except Exception as exc:
                logger.error(
                    "Retrieval pipeline error (followup fallback): %s", exc, exc_info=True
                )
                yield _sse("error", {"message": "An internal error occurred. Please try again."})
                return
            finally:
                ticker.cancel()

        elif routing_intent == "conversational":
            # ---------- Direct conversational reply — no retrieval ----------
            # Pure social/greeting messages: generate a short, context-aware
            # reply without touching the retrieval stack.
            logger.info(
                "direct_conversational path (stream)",
                extra={"correlation_id": correlation_id},
            )
            _conv_msgs: list[dict] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful knowledge assistant. "
                        "Respond naturally and concisely to the user's conversational "
                        "message. Keep your reply to 1–3 sentences."
                    ),
                },
                *[
                    m for m in (history_with_pref or [])
                    if m.get("role") in ("user", "assistant")
                ][-6:],
                {"role": "user", "content": raw_query},
            ]
            _conv_stream = (
                await self._container.generator._async_openai.chat.completions.create(
                    model=_cfg.generation.get("fast_model", "gpt-4o-mini"),
                    messages=_conv_msgs,
                    temperature=0.7,
                    max_tokens=150,
                    stream=True,
                )
            )
            yield _sse("status", {"message": "Generating response..."})
            _conv_buf: list[str] = []
            async for _chunk in _conv_stream:
                _delta = (
                    (_chunk.choices[0].delta.content or "") if _chunk.choices else ""
                )
                if _delta:
                    _conv_buf.append(_delta)
                    yield _sse("token", {"content": _delta})
            _conv_answer = "".join(_conv_buf).strip()
            try:
                _conv_answer = self._pii_guard.redact(_conv_answer, context="output")
            except Exception:
                pass
            yield _sse(
                "done",
                {
                    "answer": _conv_answer,
                    "citations": [],
                    "faithfulness_score": None,
                    "has_conflict": False,
                    "model_used": _cfg.generation.get("fast_model", "gpt-4o-mini"),
                    "agent_name": "Assistant",
                },
            )
            return

        else:
            # Safety fallback — should rarely be reached since all three intents
            # (conversational, followup, retrieval) are now handled above.
            status_q = asyncio.Queue()
            ticker = asyncio.create_task(_timed_status_ticker(status_q))
            try:
                run_task = asyncio.create_task(
                    Runner.run(
                        self._agent,
                        input=raw_query,
                        context=ctx,
                        max_turns=self._max_turns,
                    )
                )
                while not run_task.done():
                    await asyncio.sleep(0.2)
                    while not status_q.empty():
                        yield _sse("status", {"message": status_q.get_nowait()})
                run_task.result()  # re-raise exceptions

            except InputGuardrailTripwireTriggered:
                yield _sse(
                    "token",
                    {"content": "Your query could not be processed. Please check the input and try again."},
                )
                yield _sse("done", {"answer": "", "citations": [], "faithfulness_score": None, "has_conflict": False, "model_used": "", "agent_name": ""})
                return
            except OutputGuardrailTripwireTriggered:
                yield _sse(
                    "token",
                    {"content": "I was unable to generate a sufficiently reliable answer from the available sources."},
                )
                yield _sse("done", {"answer": "", "citations": [], "faithfulness_score": None, "has_conflict": False, "model_used": "", "agent_name": ""})
                return
            except Exception as exc:
                logger.error("Agent pipeline error (stream): %s", exc, exc_info=True)
                yield _sse("error", {"message": "An internal error occurred. Please try again."})
                return
            finally:
                ticker.cancel()

        # ---------- Case 1: cache hit or history answer ----------
        if ctx.generation_result is not None:
            result = ctx.generation_result
            try:
                result.answer = self._pii_guard.redact(result.answer, context="output")
            except Exception:
                pass
            yield _sse("status", {"message": "Generating answer..."})
            # Stream word-by-word for consistent UX even on cache hits
            for word in result.answer.split(" "):
                yield _sse("token", {"content": word + " "})
            yield _sse(
                "done",
                {
                    "answer": result.answer,
                    "citations": result.citations or [],
                    "faithfulness_score": result.faithfulness_score,
                    "has_conflict": result.has_conflict or False,
                    "model_used": result.model_used or "cache",
                    "agent_name": "RAG Agent",
                },
            )
            return

        # ---------- Case 2: no context (retrieval failed) ----------
        if not ctx.context_items:
            yield _sse(
                "token",
                {"content": "I could not find relevant information to answer your question."},
            )
            yield _sse("done", {"citations": [], "faithfulness_score": None, "has_conflict": False, "model_used": "", "agent_name": "RAG Agent"})
            return

        # ---------- Case 3: stream generation token by token ----------

        # Start conflict detection concurrently while we send the status event.
        # _detect_conflicts() is a blocking gpt-3.5-turbo call (~1 s); overlapping
        # it with the SSE transmission recovers that second for free.
        _conflict_task = asyncio.create_task(
            asyncio.to_thread(
                self._container.generator._detect_conflicts, ctx.context_items
            )
        )

        yield _sse("status", {"message": "Generating answer..."})

        pq = ctx.processed_query
        query_text = pq.original_query if isinstance(pq, ProcessedQuery) else raw_query
        extra_instructions = ctx.user_preferences or None

        # Use the cheaper fast model when context is sparse (≤ 2 chunks)
        use_fast_model = len(ctx.context_items) <= 2

        # Await conflict result — should be ready or very nearly so by now
        _conflict_info = await _conflict_task

        result: GenerationResult | None = None
        async for item in self._container.generator.generate_stream(
            query=query_text,
            context_items=ctx.context_items,
            extra_instructions=extra_instructions,
            _conflict_info=_conflict_info,
            use_fast_model=use_fast_model,
        ):
            if isinstance(item, str):
                yield _sse("token", {"content": item})
            elif isinstance(item, GenerationResult):
                result = item

        if result is None:
            yield _sse("error", {"message": "Generation failed unexpectedly."})
            return

        # PII redact
        try:
            result.answer = self._pii_guard.redact(result.answer, context="output")
        except Exception as exc:
            logger.warning("Output PII scan failed (stream, non-fatal): %s", exc)

        # Cache store
        if ctx._query_vector is not None:
            try:
                from src.operations.ops_middleware import SemanticCache
                cache = SemanticCache()
                cache.put(ctx._query_vector, result, namespace=ctx.namespace or None)
            except Exception as exc:
                logger.warning("Cache store failed (stream, non-fatal): %s", exc)

        # Fairness sampling
        if random.random() < self._fairness_sample_rate:
            try:
                self._fairness_evaluator.check_live_async(raw_query, namespace)
            except Exception as exc:
                logger.warning("Fairness check failed to start (stream, non-fatal): %s", exc)

        logger.info(
            "RAGOrchestrator.run_stream complete",
            extra={
                "correlation_id": correlation_id,
                "citations": len(result.citations or []),
                "has_conflict": result.has_conflict,
            },
        )

        yield _sse(
            "done",
            {
                "answer": result.answer,   # clean version with [1]/[2] citation numbers
                "citations": result.citations or [],
                "faithfulness_score": result.faithfulness_score,
                "has_conflict": result.has_conflict or False,
                "model_used": result.model_used or "unknown",
                "agent_name": "RAG Agent",
            },
        )

    async def _run_direct_pipeline(self, ctx: RAGRunContext) -> None:
        """
        Fixed-order retrieval pipeline that bypasses Agents SDK routing.

        Calls query-understanding → embedding fan-out → cache lookup → retrieval
        → (optionally) generation in a direct sequence, eliminating ~10–15 s of
        inter-tool LLM routing overhead vs the agent path.

        When ``ctx.stream_mode`` is True the generation step is skipped; the
        caller (``run_stream``) handles streaming generation after this returns.

        Note: input/output guardrails are not applied in this path.
        """
        from src.operations.ops_middleware import SemanticCache, TraceSpan

        if not ctx.correlation_id:
            ctx.correlation_id = set_correlation_id()

        qu         = self._container.query_understanding
        embedder   = self._container.embedder
        retriever  = self._container.retriever
        generator  = self._container.generator
        pii_guard  = self._container.pii_guard
        evaluator  = self._container.evaluator
        namespace  = ctx.namespace or "default"

        effective_query = (
            ctx.processed_query
            if isinstance(ctx.processed_query, str) and ctx.processed_query
            else ctx.raw_query
        )

        # ── Step 1: Query understanding (HyDE + decompose run in parallel inside qu.process) ──
        logger.info(
            "direct_pipeline: understand_query",
            extra={"query": effective_query[:80], "correlation_id": ctx.correlation_id},
        )
        with TraceSpan("understand_query"):
            pq = await asyncio.to_thread(
                qu.process, effective_query, ctx.raw_query, ctx.conversation_history
            )
        ctx.processed_query = pq
        ctx.record("direct_pipeline", f"understand: {len(pq.sub_questions)} sub-q")

        # ── Step 2: Batched embedding ────────────────────────────────────────────────────────
        # embed_batch() encodes all query-time texts in one model forward pass,
        # replacing the prior ThreadPoolExecutor fan-out of N separate encode() calls.
        with TraceSpan("query_embedding"):
            _sq_texts: list[str] = [
                sq if isinstance(sq, str) else str(sq) for sq in pq.sub_questions
            ]
            _has_hyde = bool(pq.hypothetical_doc)
            _all_texts: list[str] = [pq.final_query()]
            if _has_hyde:
                _all_texts.append(pq.hypothetical_doc)  # type: ignore[arg-type]
            _sq_start = len(_all_texts)
            _all_texts.extend(_sq_texts)

            _vectors = await asyncio.to_thread(
                embedder.embed_batch, _all_texts, pq.language
            )
            query_vector: np.ndarray = _vectors[0]
            hyde_vector: np.ndarray | None = _vectors[1] if _has_hyde else None
            sub_vectors: list[tuple[str, object]] = [
                (_sq, _vectors[_sq_start + _i]) for _i, _sq in enumerate(_sq_texts)
            ]

        if sub_vectors:
            logger.info("direct_pipeline: %d sub-question vectors", len(sub_vectors))
        if hyde_vector is not None:
            logger.info(
                "direct_pipeline: HyDE vector generated",
                extra={"hyde_preview": pq.hypothetical_doc[:120]},
            )

        # ── Step 3: Semantic cache lookup ────────────────────────────────────────────────────
        with TraceSpan("cache_lookup"):
            cache = SemanticCache()
            cached = cache.get(
                query_vector,
                query_routing_intent=ctx.query_routing_intent,
                namespace=namespace,
            )

        if cached is not None:
            logger.info("direct_pipeline: cache hit")
            ctx.generation_result = cached
            ctx._query_vector = query_vector
            ctx.record("direct_pipeline", "cache_hit")
            return

        try:
            evaluator.update_reference_distribution(query_vector)
            ctx._evaluator = evaluator
        except Exception as exc:
            logger.warning("Evaluator ref update failed (non-fatal): %s", exc)
            ctx._evaluator = None

        # ── Step 4: Retrieval ────────────────────────────────────────────────────────────────
        with TraceSpan("retrieval", {"namespace": namespace}):
            if sub_vectors:
                context_items = retriever.retrieve_multi_query(
                    pq=pq,
                    query_vector=query_vector,
                    sub_vectors=sub_vectors,
                    hyde_vector=hyde_vector,
                    namespace=namespace,
                )
                method = "multi_query"
            elif hyde_vector is not None:
                context_items = retriever.retrieve_dual(
                    pq=pq,
                    query_vector=query_vector,
                    hyde_vector=hyde_vector,
                    namespace=namespace,
                )
                method = "hyde_dual"
            else:
                context_items = retriever.retrieve(
                    pq=pq, query_vector=query_vector, namespace=namespace
                )
                method = "dense"

        ctx.context_items = context_items
        ctx._query_vector = query_vector
        ctx.record("direct_pipeline", f"retrieved: {len(context_items)} chunks via {method}")
        logger.info("direct_pipeline: %d chunks via %s", len(context_items), method)

        # ── Step 5: Generation (skipped in stream_mode — caller streams instead) ───────────
        if ctx.stream_mode:
            return

        if not context_items:
            ctx.generation_result = GenerationResult(
                answer="I could not find relevant information to answer your question.",
                model_used="",
            )
            return

        user_preferences = ctx.user_preferences or None
        query_text = pq.original_query
        use_fast_model = len(context_items) <= 2

        with TraceSpan("generation"):
            result = generator.generate(
                query=query_text,
                context_items=context_items,
                extra_instructions=user_preferences,
                use_fast_model=use_fast_model,
            )
        ctx.generation_result = result
        ctx.record("direct_pipeline", f"generated: faithfulness={result.faithfulness_score}")

        with TraceSpan("output_pii_scan"):
            try:
                result.answer = pii_guard.redact(result.answer, context="output")
            except Exception as exc:
                logger.warning("Output PII scan failed (non-fatal): %s", exc)

        with TraceSpan("cache_store"):
            try:
                cache.put(query_vector, result, namespace=ctx.namespace or None)
            except Exception as exc:
                logger.warning("Cache store failed (non-fatal): %s", exc)

        try:
            ev = ctx._evaluator or evaluator
            from src.agents.tools import _run_online_eval_bg
            context_texts = [item.chunk.text for item in context_items]
            threading.Thread(
                target=_run_online_eval_bg,
                args=(ev, query_text, result.answer, context_texts),
                daemon=True,
                name="ragas-online-eval-direct",
            ).start()
        except Exception as exc:
            logger.warning("Online eval failed to start (non-fatal): %s", exc)

    def _extract_result(self, run_result, ctx: RAGRunContext) -> GenerationResult:
        last_agent_name = (
            run_result.last_agent.name
            if run_result is not None and getattr(run_result, "last_agent", None)
            else "direct_pipeline"
        )

        if ctx.generation_result is not None:
            # For the direct pipeline (run_result is None), preserve the model
            # name already stamped by the generator (e.g. "gpt-4o-mini").
            # For the agent path, use the last agent's name.
            if run_result is not None or not ctx.generation_result.model_used:
                ctx.generation_result.model_used = last_agent_name
            return ctx.generation_result

        raw_output = getattr(run_result, "final_output", None)

        if isinstance(raw_output, GenerationOutput):
            return GenerationResult(
                answer=raw_output.answer,
                citations=[c.model_dump() for c in (raw_output.citations or [])],
                faithfulness_score=raw_output.faithfulness_score,
                has_conflict=raw_output.has_conflict,
                model_used=last_agent_name,
            )

        if isinstance(raw_output, DirectResponseOutput):
            return GenerationResult(answer=raw_output.answer, model_used=last_agent_name)

        if isinstance(raw_output, str) and raw_output.strip():
            return GenerationResult(answer=raw_output.strip(), model_used=last_agent_name)

        for cls in (GenerationOutput, DirectResponseOutput):
            try:
                output = run_result.final_output_as(cls, raise_if_incorrect_type=False)
                if output:
                    if isinstance(output, GenerationOutput):
                        return GenerationResult(
                            answer=output.answer,
                            citations=[c.model_dump() for c in (output.citations or [])],
                            faithfulness_score=output.faithfulness_score,
                            has_conflict=output.has_conflict,
                            model_used=last_agent_name,
                        )
                    return GenerationResult(answer=output.answer, model_used=last_agent_name)
            except Exception:
                pass

        logger.warning(
            "Unrecognised final_output type from agent '%s': %s — value: %s",
            last_agent_name,
            type(raw_output),
            repr(raw_output)[:200],
        )
        return GenerationResult(
            answer="I'm sorry, I wasn't able to process that. Please try again.",
            model_used=last_agent_name,
        )
