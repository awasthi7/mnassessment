from typing import Dict, List

from app.config import Settings
from app.database import Database
from app.llm import LLMProvider
from app.memory import extract_memories, retrieve_relevant_memories
from app.schemas import ChatRequest, ChatResponse


class ChatService:
    """Orchestrates memory extraction, relevance retrieval, session history, and LLM generation."""

    def __init__(self, db: Database, llm: LLMProvider, settings: Settings) -> None:
        self.db = db
        self.llm = llm
        self.settings = settings

    def handle_chat(self, request: ChatRequest) -> ChatResponse:
        # 1. Extract and persist durable long-term memories (ignoring ephemeral chatter/questions)
        extracted = extract_memories(request.message)
        memories_saved: List[str] = []
        for mem in extracted:
            self.db.upsert_memory(
                user_id=request.user_id,
                memory_key=mem.memory_key,
                category=mem.category,
                content=mem.content,
                keywords=mem.keywords,
                importance=mem.importance,
            )
            memories_saved.append(mem.memory_key)

        # 2. Retrieve only relevant long-term memories for the current message
        all_memories = self.db.get_user_memories(request.user_id)
        relevant_memories, context_used = retrieve_relevant_memories(
            query=request.message,
            memories=all_memories,
            top_k=self.settings.top_k_memories,
            min_score=self.settings.min_relevance_score,
        )

        # 3. Fetch bounded recent conversation history for this session
        recent_history = self.db.get_recent_messages(
            user_id=request.user_id,
            session_id=request.session_id,
            limit=self.settings.max_history_messages,
        )

        # 4. Build system prompt + bounded message list and generate response
        system_prompt = self._build_system_prompt(relevant_memories)
        llm_messages: List[Dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in recent_history
        ]
        llm_messages.append({"role": "user", "content": request.message})

        assistant_reply = self.llm.generate(system_prompt=system_prompt, messages=llm_messages)

        # 5. Persist both turns to session history after generation
        self.db.add_message(request.user_id, request.session_id, "user", request.message)
        self.db.add_message(request.user_id, request.session_id, "assistant", assistant_reply)

        return ChatResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            response=assistant_reply,
            context_used=context_used,
            memories_saved=memories_saved,
        )

    @staticmethod
    def _build_system_prompt(relevant_memories: List[Dict[str, object]]) -> str:
        lines = [
            "You are MyNaksh AI, a warm, thoughtful personalized astrology and life guidance assistant.",
            "Use the relevant user memories below and recent conversation history to personalize your answer.",
            "If a language_preference is provided, respond in that preferred language.",
            "Do not invent facts about the user that are not present in the provided memories or conversation.",
            "",
            "Relevant User Memories:",
        ]
        if relevant_memories:
            for mem in relevant_memories:
                lines.append(f"- [{mem['memory_key']}]: {mem['content']}")
        else:
            lines.append("- None")
        return "\n".join(lines)
