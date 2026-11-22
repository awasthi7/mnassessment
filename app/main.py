from fastapi import FastAPI

from app.config import Settings, get_settings
from app.database import Database
from app.llm import LLMProvider, build_llm_provider
from app.schemas import ChatRequest, ChatResponse, HealthResponse, MemoryItem, UserMemoriesResponse
from app.service import ChatService


def create_app(
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    db = Database(cfg.db_path)
    llm = llm_provider or build_llm_provider(cfg)
    chat_service = ChatService(db=db, llm=llm, settings=cfg)

    app = FastAPI(
        title="MyNaksh Personalized AI Chat",
        description="Lightweight personalized AI chat service with session history and long-term user memory.",
        version="1.0.0",
    )

    # Expose dependencies on app.state for testing/inspection
    app.state.settings = cfg
    app.state.db = db
    app.state.llm = llm
    app.state.chat_service = chat_service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_provider=cfg.llm_provider,
            db_path=cfg.db_path,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        return chat_service.handle_chat(payload)

    @app.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
    def list_user_memories(user_id: str) -> UserMemoriesResponse:
        rows = db.get_user_memories(user_id)
        items = [
            MemoryItem(
                memory_key=str(r["memory_key"]),
                category=str(r["category"]),
                content=str(r["content"]),
                importance=float(r["importance"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]
        return UserMemoriesResponse(user_id=user_id, memories=items)

    return app


app = create_app()
