import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Lightweight SQLite persistence for session messages and long-term user memories."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_user_session
                ON messages (user_id, session_id, id DESC);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.7,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (user_id, memory_key)
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories (user_id, updated_at DESC);
                """
            )

    def add_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (user_id, session_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, session_id, role, content, utc_now_iso()),
            )

    def get_recent_messages(
        self, user_id: str, session_id: str, limit: int = 6
    ) -> List[Dict[str, str]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        # Reverse so messages are in chronological order (oldest -> newest)
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def upsert_memory(
        self,
        user_id: str,
        memory_key: str,
        category: str,
        content: str,
        keywords: str,
        importance: float = 0.7,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    user_id, memory_key, category, content, keywords, importance, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) DO UPDATE SET
                    category = excluded.category,
                    content = excluded.content,
                    keywords = excluded.keywords,
                    importance = MAX(memories.importance, excluded.importance),
                    updated_at = excluded.updated_at
                """,
                (user_id, memory_key, category, content, keywords, importance, now, now),
            )

    def get_user_memories(self, user_id: str) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_key, category, content, keywords, importance, created_at, updated_at
                FROM memories
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]
