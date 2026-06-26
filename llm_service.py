# llm_service.py
# Handles all LangChain + Gemini LLM interaction logic

import os
import json
import re
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from models import ChatRequest, ChatResponse, SuggestedService

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines the assistant's persona, scope, and output format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable and friendly IT support agent for a company that provides the following services:

  • Web Development     – Custom websites, web apps, e-commerce, CMS solutions
  • ERP Solutions       – Enterprise Resource Planning implementation and integration
  • AI Integration      – Embedding AI/ML capabilities into business workflows
  • Cyber Security      – Threat assessment, penetration testing, security audits
  • IoT Solutions       – Smart device connectivity and industrial IoT platforms
  • CRM Solutions       – Customer Relationship Management setup and customisation

Your responsibilities:
1. RESOLVE issues — diagnose the user's problem and give clear, actionable steps.
2. EDUCATE users — explain technical concepts in plain, jargon-free language when they ask.
3. SUGGEST services — if the conversation reveals a need, naturally recommend the relevant service(s) from the list above. Only suggest services that are genuinely relevant; do not force-fit.

Tone: professional yet conversational — like a senior support engineer who also has good people skills. Be concise but thorough.

--- OUTPUT FORMAT ---
Always respond with a valid JSON object and nothing else. No markdown fences, no preamble.

{{
  "reply": "<your main response to the user>",
  "suggested_services": [
    {{
      "name": "<service name from the list above>",
      "reason": "<one-sentence reason why this service fits the user's situation>"
    }}
  ]
}}

If no services are relevant, return an empty array for suggested_services.
Do NOT hallucinate services outside the list above."""


def _build_llm() -> ChatGoogleGenerativeAI:
    """Initialise the Gemini LLM via LangChain."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in environment variables.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",   # fast, cost-effective flash model
        google_api_key=api_key,
        temperature=0.4,            # balanced: consistent but not robotic
        max_tokens=1024,
    )


def _build_prompt_template() -> ChatPromptTemplate:
    """Build the LangChain prompt template with history support."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),   # injected conversation history
        ("human", "{query}"),
    ])


def _convert_history(history: list) -> list:
    """Convert Pydantic ConversationMessage objects to LangChain message objects."""
    lc_messages = []
    for msg in history:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
    return lc_messages


def _parse_llm_output(raw: str) -> dict:
    """
    Safely parse the LLM's JSON output.
    Falls back gracefully if the model returns slightly malformed JSON.
    """
    # Strip accidental markdown code fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: extract what we can
        logger.warning("LLM returned non-JSON output; using raw text as reply.")
        return {
            "reply": cleaned,
            "suggested_services": []
        }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def get_chat_response(request: ChatRequest) -> ChatResponse:
    """
    Core function: sends the user query + history to Gemini via LangChain
    and returns a structured ChatResponse.

    Args:
        request: Validated ChatRequest from the FastAPI endpoint.

    Returns:
        ChatResponse with reply text and relevant service suggestions.
    """
    try:
        llm = _build_llm()
        prompt = _build_prompt_template()
        chain = prompt | llm | StrOutputParser()   # LangChain pipe operator

        lc_history = _convert_history(request.history or [])

        raw_output: str = chain.invoke({
            "query": request.query,
            "history": lc_history,
        })

        parsed = _parse_llm_output(raw_output)

        # Build suggested services safely
        suggested = [
            SuggestedService(
                name=svc.get("name", ""),
                reason=svc.get("reason", "")
            )
            for svc in parsed.get("suggested_services", [])
            if svc.get("name")  # skip malformed entries
        ]

        return ChatResponse(
            success=True,
            session_id=request.session_id,
            reply=parsed.get("reply", "I'm sorry, I couldn't generate a response. Please try again."),
            suggested_services=suggested,
        )

    except EnvironmentError as env_err:
        logger.error("Configuration error: %s", env_err)
        raise   # re-raise so FastAPI returns 500 with a clear message

    except Exception as exc:
        logger.exception("Unexpected error during LLM call: %s", exc)
        raise
