import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DB_PATH", "mynaksh_chat.db")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20.0"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
    top_k_memories: int = int(os.getenv("TOP_K_MEMORIES", "5"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.2"))


def get_settings() -> Settings:
    return Settings()
