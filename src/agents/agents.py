"""

Agents:

1. ConversationalAgent
   Handles casual interactions and general conversation
   (e.g., greetings, small talk, simple AI chat).

2. RetrievalAgent
   Handles queries that require retrieving external knowledge
   from documents, vector stores, databases, or APIs.

3. FollowUpAgent
   Handles follow-up questions and response refinements using
   previously retrieved context without performing new retrieval.

"""

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
    retrieve_context,
    prepare_query,
)
from src.config.settings import get_config

_cfg = get_config()
_orchestrator_model = _cfg.query.get("agents", {}).get("orchestrator_model", "gpt-4o")
_worker_model       = _cfg.query.get("agents", {}).get("worker_model", "gpt-4o")



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
        "- Rewrite the query into a fully self-contained form.\n"
        "- Generate sub-questions if necessary.\n"
        "- Prepare metadata filters.\n\n"
        "STEP 2 — Call retrieve_context.\n"
        "- Retrieve relevant passages for the main query and sub-questions.\n"
        "- Never answer before retrieval.\n\n"
        "STEP 3 — Call generate_answer.\n"
        "- Generate a grounded answer strictly from retrieved context.\n\n"
        "Rules:\n"
        "- NEVER use prior knowledge.\n"
        "- NEVER fabricate information.\n"
        "- Preserve all [CITE:...] markers exactly.\n"
        "- If retrieval is insufficient, return the generated fallback response exactly.\n"
        "- If conflict information exists, include it verbatim.\n"
        "- Do not add extra commentary, disclaimers, or preambles."
    ),
    tools=[prepare_query, retrieve_context, generate_answer],
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
    handoff_description=(
        "Handles follow-up questions and response refinements using previously "
        "retrieved context before deciding whether new retrieval is necessary."
    ),
    instructions=(
        "You are a contextual refinement assistant.\n\n"
        "Your PRIMARY goal is to answer from conversation history WITHOUT retrieval.\n\n"
        "STEP 1 — Check conversation history first (MANDATORY).\n"
        "- Read the <conversation_history> carefully.\n"
        "- If the answer is already present in the history, answer directly.\n"
        "- Short follow-ups and clarification requests almost ALWAYS refer to the "
        "previous answer — use it.\n"
        "- Only hand off to RetrievalAgent if the topic has clearly shifted to something "
        "NOT mentioned anywhere in the conversation history, or the history does not "
        "contain enough information to answer the user's current question.\n\n"
        "STEP 2 — Decide: can history answer this?\n"
        "- YES → call generate_answer directly using ctx.context.context_items "
        "if available, or answer inline.\n"
        "- NO (genuinely new topic or insufficient history) → hand off to RetrievalAgent.\n\n"
        "Rules:\n"
        "- NEVER fabricate information.\n"
        "- NEVER use unsupported prior knowledge.\n"
        "- Preserve all [CITE:...] markers exactly.\n"
        "- If retrieval is insufficient, return the fallback response exactly.\n"
        "- Keep answers concise and grounded in context."
    ),
    handoffs=[
        handoff(RetrievalAgent),
    ],
    tools=[get_conversation_history],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    output_type=GenerationOutput,
    output_guardrails=[
        output_faithfulness_guardrail,
        output_length_guardrail,
    ],
)



OrchestratorAgent: Agent[RAGRunContext] = Agent(
    name="OrchestratorAgent",
    instructions=(
        "You are the routing controller for a multi-agent RAG system.\n\n"
        "Your ONLY responsibility is to select the correct agent.\n"
        "You must NEVER answer the user's question directly.\n\n"
        "Routing Rules:\n\n"
        "1. ConversationalAgent\n"
        "- Greetings (hi, hello, hey)\n"
        "- Pure small talk with no information need (how are you, I see, great)\n"
        "- Reactions and acknowledgements (ok, thanks, interesting)\n\n"
        "2. RetrievalAgent\n"
        "- ANY question about a person, place, thing, or concept\n"
        "- 'Do you know X?' — this is a knowledge query, NOT small talk\n"
        "- 'Tell me about X', 'Who is X', 'What is X'\n"
        "- 'Can you find information about X'\n"
        "- Any named entity (person name, company, location) in the query\n"
        "- Fact-based or knowledge-intensive requests\n\n"
        "3. FollowUpAgent\n"
        "- Follow-up questions referencing a previous answer\n"
        "- Requests to simplify, expand, or clarify a previous answer\n"
        "- Short fragments that only make sense in context of prior turns\n\n"
        "CRITICAL RULES:\n"
        "- 'Do you know [name]?' is ALWAYS a RetrievalAgent query.\n"
        "- Any message containing a proper noun (name, place, company) "
        "that isn't pure small talk goes to RetrievalAgent or FollowUpAgent.\n"
        "- When in doubt between ConversationalAgent and RetrievalAgent, "
        "always prefer RetrievalAgent.\n"
        "- Always hand off immediately. Never answer directly."
    ),
    handoffs=[
        handoff(ConversationalAgent),
        handoff(RetrievalAgent),
        handoff(FollowUpAgent),
    ],
    model=_orchestrator_model,
    model_settings=ModelSettings(temperature=0.0),
    input_guardrails=[pii_input_guardrail],
)