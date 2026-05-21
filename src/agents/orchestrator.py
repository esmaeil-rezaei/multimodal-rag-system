

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import openai
from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    RunConfig,
    set_default_openai_key,
)

from src.agents.agents import OrchestratorAgent
from src.agents.context import RAGRunContext
from src.agents.schemas import DirectResponseOutput, GenerationOutput
from src.config.settings import get_config, get_secrets
from src.generation.generator import GenerationResult
from src.operations.ops_middleware import AccessControlMiddleware, PIIGuard
from src.query.understanding import QueryUnderstanding
from src.utils.logger import get_logger, set_correlation_id

logger = get_logger(__name__)
_cfg = get_config()
_sec = get_secrets()


set_default_openai_key(_sec.openai_api_key)


class RAGOrchestrator:

    def __init__(self) -> None:
        self._acl = AccessControlMiddleware()
        self._pii_guard = PIIGuard()
        self._query_understanding = QueryUnderstanding()
        self._max_turns = _cfg.query.get("agents", {}).get("max_turns", 15)


    async def run(
        self,
        raw_query: str,
        auth_token: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:

        correlation_id = set_correlation_id()

        logger.info(
            "RAGOrchestrator.run started",
            extra={"query": raw_query[:120], "correlation_id": correlation_id},
        )


        namespace = "default"
        try:
            claims = self._acl.authenticate(auth_token or "")
            namespace = self._acl.get_namespace(claims)
        except Exception as exc:
            logger.warning("ACL auth failed (%s); using default namespace.", exc)



        condensed_query = self._query_understanding.condense_with_history(
            raw_query,
            conversation_history or []
            )

        ctx = RAGRunContext(
            raw_query=raw_query,
            conversation_history=conversation_history or [],
            processed_query = condensed_query,
            auth_token=auth_token,
            correlation_id=correlation_id,
            namespace=namespace,
        )

        
        try:
            run_result = await Runner.run(
                OrchestratorAgent,
                input=raw_query,
                context=ctx,
                max_turns=self._max_turns,
            )

        except InputGuardrailTripwireTriggered as exc:
            logger.warning(
                "Input guardrail tripped: %s",
                exc,
                extra={"correlation_id": correlation_id},
            )
            return GenerationResult(
                answer=(
                    "Your query could not be processed. "
                    "Please check the input and try again."
                ),
                model_used="guardrail",
            )

        except OutputGuardrailTripwireTriggered as exc:
            logger.warning(
                "Output guardrail tripped: %s",
                exc,
                extra={"correlation_id": correlation_id},
            )
            return GenerationResult(
                answer=(
                    "I was unable to generate a sufficiently reliable answer "
                    "from the available sources. Please try rephrasing your question."
                ),
                model_used="guardrail",
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

        
        try:
            result.answer = self._pii_guard.redact(result.answer, context="output")
        except Exception as exc:
            logger.warning("Output PII scan failed (non-fatal): %s", exc)

        logger.info(
            "RAGOrchestrator.run complete",
            extra={
                "correlation_id": correlation_id,
                "last_agent": run_result.last_agent.name if run_result.last_agent else "unknown",
                "agent_trace": ctx.agent_trace,
                "answer_length": len(result.answer),
            },
        )
        return result



    def _extract_result(self, run_result, ctx: RAGRunContext) -> GenerationResult:

        if ctx.generation_result is not None:
            return ctx.generation_result

        last_agent_name = (
            run_result.last_agent.name if run_result.last_agent else "unknown"
        )
        raw_output = getattr(run_result, "final_output", None)


        if isinstance(raw_output, GenerationOutput):
            return GenerationResult(
                answer=raw_output.answer,
                citations=[{"chunk_id": c} for c in (raw_output.citations or [])],
                faithfulness_score=raw_output.faithfulness_score,
                has_conflict=raw_output.has_conflict,
                model_used=last_agent_name,
            )

        if isinstance(raw_output, DirectResponseOutput):
            return GenerationResult(
                answer=raw_output.answer,
                model_used=last_agent_name,
            )


        if isinstance(raw_output, str) and raw_output.strip():
            return GenerationResult(
                answer=raw_output.strip(),
                model_used=last_agent_name,
            )


        try:
            output = run_result.final_output_as(
                GenerationOutput, raise_if_incorrect_type=False
            )
            if output:
                return GenerationResult(
                    answer=output.answer,
                    citations=[{"chunk_id": c} for c in (output.citations or [])],
                    faithfulness_score=output.faithfulness_score,
                    has_conflict=output.has_conflict,
                    model_used=last_agent_name,
                )
        except Exception:
            pass

        try:
            output = run_result.final_output_as(
                DirectResponseOutput, raise_if_incorrect_type=False
            )
            if output:
                return GenerationResult(
                    answer=output.answer,
                    model_used=last_agent_name,
                )
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