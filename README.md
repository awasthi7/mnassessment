# MyNaksh — Personalized AI Chat Service

A lightweight, modular FastAPI + SQLite chat service that maintains bounded recent conversation history per session and extracts, stores, updates, and selectively retrieves long-term user memories to personalize LLM responses.

## Architecture Overview

```
POST /chat
   │
   ▼
ChatService (app/service.py)
   ├── 1. Memory Extraction (app/memory.py) ──► SQLite `memories` (upsert by user_id + memory_key)
   ├── 2. Relevance Retrieval (app/memory.py) ──► Selects top-K relevant memories (excludes unrelated ones)
   ├── 3. Bounded History (app/database.py) ──► Last N messages for (user_id, session_id)
   └── 4. LLM Generation (app/llm.py) ──► Swappable LLMProvider (OpenAI / Gemini / Mock + Fallback)
```

### Key Design Decisions & Trade-offs

1. **Hybrid Rule + Intent Memory Extraction (`app/memory.py`)**:
   - Extracts durable facts (`career_goal`, `language_preference`, `relationship_goal`, `health_goal`, `astrology_profile`, `personal_fact`, `interest`) while filtering out pure questions and ephemeral chatter.
   - **Trade-off**: Deterministic regex/keyword extraction adds zero extra LLM latency or API cost and works reliably offline/in tests. In production, it can be paired with an async LLM extraction pass for nuanced phrasing.
2. **Upsert for Mutable Preferences & Goals (`app/database.py`)**:
   - Memories are keyed by `(user_id, memory_key)` with `ON CONFLICT DO UPDATE`. When a user updates a preference (e.g., switching from Hindi to English), the existing record is updated in place rather than creating conflicting duplicates.
3. **Selective Relevance Retrieval (`retrieve_relevant_memories`)**:
   - Combines query tokenization (with stop-word removal), domain intent matching (e.g., career, relationship, health, astrology), and importance weighting, filtered by `MIN_RELEVANCE_SCORE` and capped at `TOP_K_MEMORIES`.
   - Unrelated memories (e.g., hobbies when asking about career) fall below the threshold and are excluded from the LLM prompt. `language_preference` is included as a global response-formatting preference when present.
   - **Trade-off**: Avoids vector database / embedding overhead as requested, while remaining easy to swap for hybrid BM25 + vector search later.
4. **Bounded Session History**:
   - Only the most recent `MAX_HISTORY_MESSAGES` (default `6`, i.e. 3 user-assistant turns) for `(user_id, session_id)` are fetched and passed to the LLM, keeping token usage bounded while supporting follow-ups like `"Why do you say that?"`.
5. **Swappable & Resilient LLM Layer (`app/llm.py`)**:
   - `LLMProvider` abstracts `OpenAILLMProvider` (any OpenAI-compatible endpoint via `httpx`), `GeminiLLMProvider`, and `MockLLMProvider`.
   - External providers are wrapped in `ResilientLLMProvider` so transient API errors or timeouts degrade gracefully instead of crashing the endpoint.

## Installation & Running Locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional; defaults to SQLite + MockLLMProvider)
cp .env.example .env

# 4. Run the FastAPI server
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Using a Real LLM Provider

Set environment variables before starting `uvicorn`:

```bash
# OpenAI (or any OpenAI-compatible API)
export LLM_PROVIDER=openai
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini

# OR Google Gemini
export LLM_PROVIDER=gemini
export LLM_API_KEY=...
export LLM_MODEL=gemini-2.5-flash
```

## Running Automated Tests

Tests use Python's standard `unittest` + FastAPI's `TestClient` (compatible with both `pytest` and `python3 -m unittest`):

```bash
python3 -m unittest discover -s tests -v
# or
pytest -v
```

## Endpoints

- `POST /chat` — Accepts `user_id`, `session_id`, `message`; returns `response`, `context_used`, and `memories_saved`.
- `GET /users/{user_id}/memories` — Inspects stored long-term memories for a user.
- `GET /health` — Service health and active provider check.

See `samples/sample_requests.md` for example `curl` requests and responses.
