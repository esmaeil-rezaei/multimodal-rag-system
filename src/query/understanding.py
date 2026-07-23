from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
import spacy
from langdetect import detect as _langdetect

from src.config.settings import get_config, get_secrets
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedQuery:
    """
    The output of the query understanding stage.
    Carries the rewritten query, sub-questions, and metadata filters.
    """

    original_query: str
    expanded_query: str = ""
    standalone_query: str = ""
    sub_questions: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    hypothetical_doc: str | None = None
    language: str | None = None
    query_routing_intent: str = "retrieval"

    def final_query(self) -> str:
        """Return the best query string to use for retrieval."""
        return self.expanded_query or self.standalone_query or self.original_query


class QueryUnderstanding:
    """
    Applies the full query understanding pipeline to a raw user query.
    Each step is enabled/disabled via config.yaml.
    """

    def __init__(self) -> None:
        cfg = get_config()
        sec = get_secrets()
        self._q_cfg = cfg.query
        self._openai = openai.OpenAI(
            api_key=sec.openai_api_key,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            max_retries=0,
        )

        self._nlp: spacy.Language | None = None
        if self._q_cfg["entity_recognition"]["enabled"]:
            try:
                self._nlp = spacy.load(self._q_cfg["entity_recognition"]["model"])
            except OSError:
                logger.warning(
                    "spaCy model not found. Run: python -m spacy download en_core_web_trf"
                )

    def process(
        self,
        query: str,
        raw_query: str,
        conversation_history: list[dict[str, str]],
    ) -> ProcessedQuery:
        """Run the full query understanding pipeline on a raw query."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        pq = ProcessedQuery(
            original_query=raw_query,
            standalone_query=query,
        )

        pq.query_routing_intent = self._classify_routing_intent(
            raw_query, ctx_history=conversation_history or []
        )

        working_query = pq.standalone_query

        # Fan-out independent LLM calls (HyDE expansion, sub-question decomposition,
        # NER entity filters) in parallel — saves ~1–1.5 s vs sequential.
        expand_enabled = self._q_cfg["expansion"]["enabled"]

        # HyDE generates a hypothetical document to improve retrieval for longer,
        # nuanced queries.  For very short queries (< 5 words) the signal is too
        # sparse for expansion to help, so skip it to save ~1–1.5 s.
        if expand_enabled and len(working_query.split()) < 5:
            expand_enabled = False
            logger.debug(
                "HyDE skipped: query too short (%d words)", len(working_query.split())
            )

        decomp_enabled = self._q_cfg["decomposition"]["enabled"]
        ner_enabled    = self._q_cfg["entity_recognition"]["enabled"]
        n_workers = sum([expand_enabled, decomp_enabled, ner_enabled])

        if n_workers > 0:
            with ThreadPoolExecutor(max_workers=n_workers) as _pool:
                _futs: dict[str, Any] = {}
                if expand_enabled:
                    _futs["expand"]   = _pool.submit(self._expand_query,   working_query)
                if decomp_enabled:
                    _futs["decompose"] = _pool.submit(self._decompose_query, working_query)
                if ner_enabled:
                    _futs["ner"]      = _pool.submit(self._extract_filters, working_query)

                if "expand" in _futs:
                    pq.expanded_query, pq.hypothetical_doc = _futs["expand"].result()
                else:
                    pq.expanded_query = working_query

                if "decompose" in _futs:
                    pq.sub_questions = _futs["decompose"].result()

                if "ner" in _futs:
                    pq.metadata_filters = _futs["ner"].result()
        else:
            pq.expanded_query = working_query

        try:
            pq.language = _langdetect(query)
        except Exception:
            pq.language = "en"

        logger.info(
            "Query processed",
            extra={
                "original": query,
                "standalone": pq.standalone_query,
                "sub_questions": len(pq.sub_questions),
                "filters": pq.metadata_filters,
            },
        )
        return pq

    def get_routing_intent(self, query: str, history: list[dict[str, str]]) -> str:
        """
        Public wrapper around the rule-based routing classifier.
        Returns 'conversational', 'followup', or 'retrieval'.
        """
        return self._classify_routing_intent(query, ctx_history=history)

    def compress_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Summarise overflow turns into a [CONVERSATION SUMMARY] system message.
        Controlled by query.conversation.compress_history in config.yaml."""
        conv_cfg = self._q_cfg.get("conversation", {})
        if not conv_cfg.get("compress_history", False):
            return history

        threshold: int = conv_cfg.get("compress_threshold", 14)
        window: int = conv_cfg.get("history_window", 6)
        model: str = conv_cfg.get("compress_model", "gpt-4o-mini")
        max_tok: int = conv_cfg.get("compress_max_tokens", 400)

        if len(history) <= threshold:
            return history

        split = max(0, len(history) - window)
        old_turns = history[:split]
        recent_turns = history[split:]

        if (
            old_turns
            and old_turns[0].get("role") == "system"
            and old_turns[0].get("content", "").startswith("[CONVERSATION SUMMARY]")
        ):
            existing_summary = old_turns[0]["content"]
            turns_to_add = old_turns[1:]
        else:
            existing_summary = None
            turns_to_add = old_turns

        turns_text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns_to_add)

        if existing_summary:
            user_content = (
                f"Existing summary:\n{existing_summary}\n\n"
                f"New turns to incorporate:\n{turns_text}\n\n"
                "Update the summary to include the new turns. "
                "Preserve all key facts, answers, and user preferences."
            )
        else:
            user_content = (
                f"Conversation turns:\n{turns_text}\n\n"
                "Write a concise factual summary. "
                "Preserve key questions, answers, and user preferences stated. "
                "Do not answer questions — only summarise what was said."
            )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation compressor. "
                            "Summarise the provided turns into a dense, factual paragraph."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=max_tok,
            )
            summary_text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("History compression failed (%s); keeping full history.", exc)
            return history

        compressed = [
            {"role": "system", "content": f"[CONVERSATION SUMMARY]\n{summary_text}"}
        ] + list(recent_turns)
        logger.info(
            "History compressed: %d turns → summary + %d recent turns",
            len(history),
            len(recent_turns),
        )
        return compressed

    def condense_with_history(self, query: str, history: list[dict[str, str]]) -> str:
        """
        Turns a follow-up question into a standalone query by using chat history.
        """

        window = self._q_cfg["conversation"]["history_window"]
        model = self._q_cfg["conversation"]["condense_model"]

        trimmed = history[-window:]

        history_str = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in trimmed
        )
        MAX_HISTORY_CHARS = 3_000
        if len(history_str) > MAX_HISTORY_CHARS:
            history_str = "..." + history_str[-MAX_HISTORY_CHARS:]

        system_prompt = (
            "You are a query rewriting engine for a RAG system.\n"
            "Your task is to convert a follow-up question into a standalone question.\n\n"
            "STRICT RULES:\n"
            "1. Use ONLY information explicitly present in the conversation history.\n"
            "2. DO NOT guess, infer, or add missing details.\n"
            "3. If the question is already standalone, return it unchanged but correcting typos if there is any.\n"
            "4. Preserve technical terms exactly as written.\n"
            "5. Do NOT answer the question.\n"
            "6. Output ONLY the rewritten question.\n"
            "7. If the user message is casual conversation, chit-chat, greeting, acknowledgement, "
            "reaction, or small talk (examples: 'ok', 'great', 'thanks', 'interesting', "
            "'hello', 'cool', 'nice'), return the EXACT original user query unchanged.\n"
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Conversation history:\n{history_str}\n\n"
                            f"Follow-up question: {query}\n\n"
                            "Rewritten standalone question:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip() or query
        except Exception as exc:
            logger.warning("Condense-with-history failed (%s); using raw query.", exc)
            return query

    def _expand_query(self, query: str) -> tuple[str, str | None]:
        """Expand the query using HyDE or query2doc; returns (expanded_query, hypothetical_doc)."""

        method = self._q_cfg["expansion"]["method"]  # "hyde" | "query2doc" | "prf"

        if method == "hyde":
            model = self._q_cfg["expansion"]["hyde_model"]
            max_tok = self._q_cfg["expansion"]["hyde_max_tokens"]
            try:
                response = self._openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Before generating, correct any spelling mistakes in the question and "
                                "consider name variations and alternative spellings.\n"
                                "Then write a plausible 2-3 sentence passage as if extracted from a source document "
                                "that directly answers the question in its corrected and variant forms.\n"
                                "Do not refuse. Return ONLY the passage."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                    max_tokens=max_tok,
                )
                hypothetical_doc = response.choices[0].message.content.strip()

                return f"{query} {hypothetical_doc}", hypothetical_doc
            except Exception as exc:
                logger.warning("HyDE expansion failed (%s); using raw query.", exc)
                return query, None

        elif method == "query2doc":
            try:
                response = self._openai.chat.completions.create(
                    model=self._q_cfg["expansion"]["hyde_model"],
                    messages=[
                        {
                            "role": "user",
                            "content": f"Expand this search query into a detailed question: {query}",
                        }
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                expanded = response.choices[0].message.content.strip()
                return f"{query} {expanded}", None
            except Exception as exc:
                logger.warning("query2doc expansion failed (%s); using raw query.", exc)
                return query, None

        else:
            return query, None

    def _classify_routing_intent(
        self,
        query: str,
        ctx_history: list[dict[str, str]],
    ) -> str:
        """
        LLM-based routing classifier.

        Uses the model configured under ``query.intent_routing.model``
        (default: gpt-4o-mini) with ``max_tokens=5`` — extremely cheap and fast.
        This method is designed to be called via ``asyncio.to_thread`` from the
        orchestrator so it runs concurrently with history compression and preference
        detection, adding zero wall-clock latency.

        Returns one of: ``'conversational'``, ``'followup'``, ``'retrieval'``.
        """
        q = query.strip()
        if not q:
            return "conversational"

        routing_cfg = self._q_cfg.get("intent_routing", {})
        model: str = routing_cfg.get("model", "gpt-4o-mini")

        # Provide the last few turns so the model can correctly classify
        # follow-up messages that rely on prior context.
        history_excerpt = "\n".join(
            f"{m['role'].upper()}: {m['content'][:150]}"
            for m in (ctx_history or [])[-4:]
            if m.get("role") in ("user", "assistant")
        )

        system = (
            "Classify the user's message into exactly one routing intent.\n\n"
            "• conversational — a social or casual message that requires no knowledge-base "
            "lookup: greetings, small talk, one-word reactions, acknowledgments, or questions "
            "about the assistant itself (e.g. 'hi', 'thanks', 'interesting', 'ok', "
            "'good morning', 'how are you', 'who are you').\n"
            "• followup — the message explicitly references something already discussed in "
            "the conversation and asks to elaborate, clarify, or summarise it "
            "(e.g. 'tell me more', 'expand on that', 'what did you mean by X', 'elaborate').\n"
            "• retrieval — a factual question or domain topic that requires searching a "
            "knowledge base (e.g. 'Alzheimer disease', 'what is HER2?', 'CT scan protocols', "
            "'why does amyloid accumulate?').\n\n"
            "Reply with exactly one word: conversational, followup, or retrieval."
        )

        user_msg = f"Message: {q}"
        if history_excerpt:
            user_msg = f"Recent conversation:\n{history_excerpt}\n\nMessage: {q}"

        try:
            resp = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            raw = (resp.choices[0].message.content or "retrieval").strip().lower()
            for label in ("conversational", "followup", "retrieval"):
                if label in raw:
                    logger.debug("Routing intent for %r → %s", q[:60], label)
                    return label
            logger.warning("Unexpected routing label %r for query %r; defaulting to retrieval.", raw, q[:60])
            return "retrieval"
        except Exception as exc:
            logger.warning("Routing LLM call failed (%s); defaulting to retrieval.", exc)
            return "retrieval"

    def _decompose_query(self, query: str) -> list[str]:
        """
        Decompose a complex multi-intent query into atomic sub-questions.
        """
        max_sub = self._q_cfg["decomposition"]["max_sub_questions"]
        model = self._q_cfg["decomposition"].get(
            "model", self._q_cfg["conversation"]["condense_model"]
        )

        system_prompt = (
            f"You are a question decomposition assistant. "
            f"If the question is already simple and self-contained, return it as a single sub-question. "
            f"If it is complex or multi-hop, break it into at most {max_sub} simpler, "
            f"self-contained sub-questions. Return ONLY a numbered list, one per line."
        )

        try:
            response = self._openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Query decomposition failed (%s); skipping.", exc)
            return []

        sub_questions = []
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\d+[.)]\s*", "", line.strip())
            if cleaned:
                sub_questions.append(cleaned)

        return sub_questions[:max_sub]

    def _extract_filters(self, query: str) -> dict[str, Any]:
        """
        Run NER on the query to extract named entities and temporal references.
        These become metadata filters applied during vector store search.
        """
        filters: dict[str, Any] = {}

        if self._nlp:
            doc = self._nlp(query)
            for ent in doc.ents:
                filter_key = f"entity_{ent.label_}"
                filters.setdefault(filter_key, []).append(ent.text)
                logger.debug("NER entity found: %s = '%s'", ent.label_, ent.text)

        if self._q_cfg["entity_recognition"]["temporal_grounding"]:
            date_filter = self._extract_date_filter(query)
            if date_filter:
                filters["date_range"] = date_filter

        return filters

    @staticmethod
    def _extract_date_filter(query: str) -> dict[str, str] | None:
        """
        Extract a date range filter from common temporal expressions in the query.
        Returns a dict with "gte" and "lte" ISO-8601 date strings, or None.
        """

        quarter_match = re.search(r"Q([1-4])\s+(\d{4})", query, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2))
            quarter_starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
            quarter_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            return {
                "gte": f"{year}-{quarter_starts[quarter]}",
                "lte": f"{year}-{quarter_ends[quarter]}",
            }

        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            year = year_match.group(1)
            return {"gte": f"{year}-01-01", "lte": f"{year}-12-31"}

        return None
