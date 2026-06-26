# models.py
# Pydantic schemas for request validation and response structuring

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    """Represents a single turn in the conversation history."""
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """
    Incoming POST request body from the frontend.

    Fields:
        query       : The user's current message/question.
        session_id  : Optional identifier to track conversation context.
        history     : Optional prior conversation turns for multi-turn support.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's message or question",
        examples=["What is ERP and how can it help my business?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for tracking conversations"
    )
    history: Optional[list[ConversationMessage]] = Field(
        default=[],
        description="Prior conversation turns for multi-turn context"
    )


class SuggestedService(BaseModel):
    """A service recommended by the assistant based on the user's query."""
    name: str = Field(description="Name of the IT service")
    reason: str = Field(description="Why this service is relevant to the user's query")


class ChatResponse(BaseModel):
    """
    JSON response returned to the frontend.

    Fields:
        success           : Whether the request was processed successfully.
        session_id        : Echoed back for frontend session tracking.
        reply             : The assistant's main response text.
        suggested_services: List of relevant services surfaced from the query.
        error             : Error message, present only when success is False.
    """
    success: bool = Field(description="Whether the request succeeded")
    session_id: Optional[str] = Field(default=None)
    reply: str = Field(description="Assistant's response to the user")
    suggested_services: list[SuggestedService] = Field(
        default=[],
        description="IT services relevant to the user's query"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if success is False"
    )


class ErrorResponse(BaseModel):
    """Generic error response wrapper."""
    success: bool = False
    error: str
    reply: str = "Something went wrong. Please try again."
    suggested_services: list = []
