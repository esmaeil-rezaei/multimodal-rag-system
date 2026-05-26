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
        "You are a CONTEXT RESOLUTION agent in a RAG system.\n\n"

        "STEP 1: Call get_conversation_history.\n\n"

        "STEP 2: Decide:\n\n"

        "CASE A — CONTEXT ANSWERABLE:\n"
        "- If the question clearly refers to previous conversation AND you can answer from history\n"
        "- → call generate_answer\n\n"

        "CASE B — ASKS FOR NEW KNOWLEDGE:\n"
        "- If the question requires external information (math, science, facts, definitions, examples)\n"
        "- → hand off to RetrievalAgent\n\n"

        "CASE C — AMBIGUOUS OR PRONOUN-REFERENCING:\n"
        "- If question is ambiguous OR refers to prior topic with unclear reference\n"
        "- → hand off to RetrievalAgent\n\n"

        "CRITICAL RULES:\n"
        "- NEVER produce explanatory text.\n"
        "- NEVER say 'I don't know'.\n"
        "- NEVER narrate reasoning.\n"
        "- Either generate_answer OR handoff — nothing else.\n"
        "- Never say I've handed off your question to another agent. Just hand it off silently.\n"
    ),
    tools=[get_conversation_history, generate_answer],
    handoffs=[handoff(RetrievalAgent)],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    # NOTE: output_type and output_guardrails are intentionally omitted here.
    # An agent with handoffs cannot also enforce a structured output_type,
    # because when it hands off, the framework expects the *receiving* agent
    # to produce the output. Adding output_type here causes the handoff to
    # silently fail — the model tries to hand off but the framework blocks it
    # waiting for a GenerationOutput that never comes.
)

OrchestratorAgent: Agent[RAGRunContext] = Agent(
    name="OrchestratorAgent",
    instructions=(
        "You are the routing controller for a multi-agent RAG system.\n\n"
        "Your ONLY responsibility is to select the correct agent.\n"
        "You must NEVER answer the user's question directly.\n\n"

        "STEP 1 — Always call get_conversation_history first.\n"
        "You need the history to make a correct routing decision.\n\n"

        "STEP 2 — Apply routing rules in this exact order:\n\n"

        "3. FollowUpAgent — CHECK THIS FIRST before Retrieval\n"
        "Route here if ANY of the following are true:\n"
        "  a) Explicit callbacks:\n"
        "     - Contains: 'you just', 'you said', 'you mentioned', 'as you said',\n"
        "       'but you', 'above', 'earlier', 'previous', 'last time'\n"
        "  b) Referential pronouns (when history exists):\n"
        "     - Starts with or contains: 'it', 'its', 'they', 'their', 'this', 'that',\n"
        "       'these', 'those', 'the same', 'the method', 'the approach',\n"
        "       'the algorithm', 'the model', 'the result', 'the comparison'\n"
        "  c) Expansion requests on a prior topic:\n"
        "     - 'tell me more', 'elaborate', 'expand', 'clarify', 'explain more',\n"
        "       'more detail', 'give an example', 'can you summarize', 'summarize this',\n"
        "       'summarize the chat', 'list the takeaways', 'itemize'\n"
        "  d) Implicit topic continuations (NO retrieval keywords, history exists):\n"
        "     - Short query (under 6 words) with no named entity AND history exists\n"
        "     - Vague nouns that only make sense given prior context:\n"
        "       'superiority', 'the difference', 'the advantage', 'the features',\n"
        "       'the findings', 'the comparison', 'the results', 'why?', 'how so?',\n"
        "       'really?', 'are you sure?', 'prove it'\n"
        "  e) Correction or disagreement with the previous answer:\n"
        "     - 'that is wrong', 'not quite', 'i disagree', 'actually', 'but wait',\n"
        "       'that's not what i asked'\n\n"

        "1. ConversationalAgent\n"
        "Route here ONLY if ALL of the following are true:\n"
        "  - No information need whatsoever\n"
        "  - No named entity present\n"
        "  - History is empty OR message is clearly a greeting/reaction\n"
        "  - Examples: 'hi', 'hello', 'thanks', 'ok', 'cool', 'got it',\n"
        "    'how are you', 'what is your name'\n\n"

        "2. RetrievalAgent\n"
        "Route here if:\n"
        "  - Query is a direct knowledge question with no reference to prior turns\n"
        "  - No follow-up signals from section 3 are present\n"
        "  - Examples: 'what is HYBRID algorithm?', 'who is Alan Turing?',\n"
        "    'explain transformer architecture'\n\n"

        "CRITICAL RULES:\n"
        "- Evaluate FollowUpAgent BEFORE RetrievalAgent. Most ambiguous cases\n"
        "  belong to FollowUp, not Retrieval.\n"
        "- 'Do you know [name]?' → RetrievalAgent (knowledge query, not small talk).\n"
        "- Any request to summarize, recap, or list takeaways from THIS conversation\n"
        "  → FollowUpAgent (it has get_conversation_history).\n"
        "- A vague noun or short phrase that only makes sense given prior context\n"
        "  → FollowUpAgent, even if it looks like a retrieval query.\n"
        "- When in doubt between Retrieval and FollowUp and history exists → FollowUp.\n"
        "- When in doubt between Conversational and Retrieval → Retrieval.\n"
        "- Always hand off immediately. Never answer directly."
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
