from __future__ import annotations

from agents import Agent, ModelSettings, handoff

from src.agents.context import RAGRunContext
from src.agents.guardrails import (
    output_faithfulness_guardrail,
    output_length_guardrail,
    pii_input_guardrail,
)
from src.agents.schemas import DirectResponseOutput, GenerationOutput
from src.agents.tools import (
    get_conversation_history,
    generate_answer,
    generate_from_history,
    retrieve_context,
    prepare_query,
)
from src.agents.graph_tools import graph_query
from src.config.settings import get_config

_cfg = get_config()
_orchestrator_model = _cfg.query.get("agents", {}).get("orchestrator_model", "gpt-4o")
_worker_model       = _cfg.query.get("agents", {}).get("worker_model",       "gpt-4o")

ConversationalAgent: Agent[RAGRunContext] = Agent(
    name="ConversationalAgent",
    handoff_description=(
        "Handles greetings, casual conversation, and general non-retrieval interactions."
    ),
    instructions=(
        "You are a friendly and concise conversational assistant.\n\n"
        "The user's message does not require document retrieval or external knowledge lookup.\n"
        "Respond naturally, conversationally, and briefly.\n\n"
        "Rules:\n"
        "- Keep responses short and warm.\n"
        "- Do not fabricate factual information.\n"
        "- If the user appears to ask a knowledge-intensive question, "
        "encourage them to provide more details instead of answering from prior knowledge.\n"
        "- Never claim to have retrieved information."
    ),
    model=_orchestrator_model,
    model_settings=ModelSettings(temperature=0.7),
    output_type=DirectResponseOutput,
    input_guardrails=[pii_input_guardrail],
)

RetrievalAgent: Agent[RAGRunContext] = Agent(
    name="RetrievalAgent",
    handoff_description=(
        "Handles queries requiring retrieval from external knowledge sources."
    ),
    instructions=(
        "You are a retrieval-augmented generation (RAG) assistant.\n\n"
        "Your task is to answer the user's question using ONLY retrieved context.\n\n"

        "Execution flow (mandatory):\n\n"
        "STEP 1 — Call prepare_query.\n"
        "STEP 2 — Call retrieve_context.\n"
        "STEP 3 — Optionally call graph_query.\n"
        "STEP 4 — Call generate_answer.\n\n"

        "Rules:\n"
        "- NEVER use prior knowledge.\n"
        "- NEVER fabricate information.\n"
        "- NEVER say 'I am retrieving', 'let me search', or narrate your steps.\n"
        "- NEVER produce any output before generate_answer completes.\n"
        "- Preserve all [CITE:...] markers exactly.\n"
        "- If retrieval is insufficient, return the generated fallback response exactly.\n"
        "- If conflict information exists, include it verbatim.\n"
        "- Do not add extra commentary, disclaimers, or preambles."
    ),
    tools=[prepare_query, retrieve_context, graph_query, generate_answer],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    output_type=GenerationOutput,
    input_guardrails=[pii_input_guardrail],
    output_guardrails=[
        output_faithfulness_guardrail,
        output_length_guardrail,
    ],
)

FollowUpAgent: Agent[RAGRunContext] = Agent(
    name="FollowUpAgent",
    instructions=(
        "You are a follow-up resolution agent. Execute these steps in order:\n\n"

        "STEP 1: Call get_conversation_history.\n\n"

        "STEP 2: Call generate_from_history.\n\n"

        "STEP 3: Read the tool result:\n"
        "  - If needs_retrieval is false → you are done. Do NOT output any text.\n"
        "  - If needs_retrieval is true → hand off to RetrievalAgent.\n\n"

        "ABSOLUTE RULES:\n"
        "  - Never produce any text yourself.\n"
        "  - Never skip steps.\n"
        "  - Never ask the user for clarification.\n"
        "  - Hand off silently."
    ),
    tools=[get_conversation_history, generate_from_history],
    handoffs=[handoff(RetrievalAgent)],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    # NOTE: output_type intentionally omitted — an agent with handoffs cannot
    # also enforce a structured output_type (the framework would block handoffs
    # waiting for a typed output that never arrives from the receiving agent).
    # generate_from_history stores its result on ctx.generation_result directly.
)

OrchestratorAgent: Agent[RAGRunContext] = Agent(
    name="OrchestratorAgent",
    instructions=(
        "You are the routing controller for a multi-agent RAG system.\n\n"
        "Your ONLY job is to hand off to the correct agent. Never answer directly.\n\n"

        "STEP 1 — Call get_conversation_history.\n\n"

        "STEP 2 — Hand off using these rules (evaluate in this order):\n\n"

        "FollowUpAgent — EVALUATE FIRST\n"
        "Route here when conversation history exists AND the message is connected "
        "to a prior turn: it references something already discussed, asks for "
        "elaboration or clarification on a previous answer, uses pronouns or "
        "shorthand that only make sense given prior context, requests a summary "
        "of the conversation, or expresses disagreement with a prior answer.\n"
        "When history exists and the intent is ambiguous, prefer FollowUpAgent "
        "over RetrievalAgent.\n\n"

        "ConversationalAgent\n"
        "Route here only for pure small talk or social pleasantries with no "
        "information need and no connection to prior content.\n\n"

        "RetrievalAgent\n"
        "Route here for direct knowledge questions that stand alone, with no "
        "reference to prior turns and no ambiguity.\n\n"

        "Always hand off immediately. Never answer directly."
    ),
    tools=[get_conversation_history],
    handoffs=[
        handoff(FollowUpAgent),
        handoff(ConversationalAgent),
        handoff(RetrievalAgent),
    ],
    model=_orchestrator_model,
    model_settings=ModelSettings(temperature=0.0),
    input_guardrails=[pii_input_guardrail],
)
