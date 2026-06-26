# main.py
# FastAPI application — entry point for the IT support chatbot backend

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import ChatRequest, ChatResponse, ErrorResponse
from llm_service import get_chat_response

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup and once on shutdown."""
    logger.info("IT Support Chatbot backend is starting up...")
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("⚠️  GEMINI_API_KEY is not set. Requests will fail until it is configured.")
    yield
    logger.info("IT Support Chatbot backend is shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IT Support Chatbot API",
    description=(
        "Backend for an AI-powered IT support chatbot. "
        "Accepts user queries and returns structured JSON responses with "
        "support answers and relevant service suggestions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the frontend origin(s) to call this API
# Update `allow_origins` with your actual frontend URL in production.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # replace with e.g. ["https://yourwebsite.com"] in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — ensures all errors return valid JSON
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc)
    error_body = ErrorResponse(error=str(exc))
    return JSONResponse(
        status_code=500,
        content=error_body.model_dump(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Quick health-check endpoint."""
    return {"status": "ok", "message": "IT Support Chatbot API is running."}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health-check for load balancers / monitoring."""
    return {
        "status": "healthy",
        "service": "IT Support Chatbot",
        "version": "1.0.0",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a message to the IT support chatbot",
    responses={
        200: {"description": "Successful response from the chatbot"},
        422: {"description": "Validation error — check your request body"},
        500: {"description": "Internal server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    **Primary endpoint.**

    Accepts a JSON body with the user's query (and optionally conversation
    history) and returns the chatbot's structured reply.

    **Request body:**
    ```json
    {
      "query": "How do I secure my company's network?",
      "session_id": "user-123",
      "history": [
        { "role": "user", "content": "Hello" },
        { "role": "assistant", "content": "Hi! How can I help you today?" }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "success": true,
      "session_id": "user-123",
      "reply": "To secure your network, start with ...",
      "suggested_services": [
        {
          "name": "Cyber Security",
          "reason": "A security audit would identify vulnerabilities in your current setup."
        }
      ],
      "error": null
    }
    ```
    """
    logger.info(
        "Chat request | session=%s | query_len=%d",
        request.session_id or "anonymous",
        len(request.query),
    )

    response = await get_chat_response(request)

    logger.info(
        "Chat response | session=%s | services_suggested=%d",
        request.session_id or "anonymous",
        len(response.suggested_services),
    )

    return response


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,   # set to False in production
    )
