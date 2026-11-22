from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    session_id: str = Field(..., min_length=1, description="Conversation session identifier")
    message: str = Field(..., min_length=1, description="User's current message")


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    context_used: List[str] = Field(default_factory=list)
    memories_saved: List[str] = Field(default_factory=list)


class MemoryItem(BaseModel):
    memory_key: str
    category: str
    content: str
    importance: float
    updated_at: str


class UserMemoriesResponse(BaseModel):
    user_id: str
    memories: List[MemoryItem]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    db_path: str
