import json
import os
import tempfile
import unittest
from typing import Dict, List
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import (
    GeminiLLMProvider,
    LLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    ResilientLLMProvider,
)
from app.main import create_app
from app.memory import extract_memories, retrieve_relevant_memories


class SpyLLMProvider(LLMProvider):
    """Test double that records exact system prompts and message windows passed to the LLM."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []
        self._delegate = MockLLMProvider()

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        self.calls.append({"system_prompt": system_prompt, "messages": list(messages)})
        return self._delegate.generate(system_prompt, messages)


class FailingLLMProvider(LLMProvider):
    """Test double that simulates an upstream LLM outage or timeout."""

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        raise RuntimeError("Simulated upstream LLM 503 Service Unavailable")


class TestPersonalizedChatAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_chat.db")
        self.settings = Settings(
            db_path=self.db_path,
            llm_provider="mock",
            max_history_messages=4,
            top_k_memories=5,
            min_relevance_score=0.2,
        )
        self.spy_llm = SpyLLMProvider()
        self.app = create_app(settings=self.settings, llm_provider=self.spy_llm)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_health_and_validation_errors(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["llm_provider"], "mock")

        bad_req = self.client.post(
            "/chat",
            json={"user_id": "", "session_id": "s1", "message": "hi"},
        )
        self.assertEqual(bad_req.status_code, 422)

        empty_msg = self.client.post(
            "/chat",
            json={"user_id": "u1", "session_id": "s1", "message": ""},
        )
        self.assertEqual(empty_msg.status_code, 422)

    def test_unknown_user_memories_returns_empty_list(self) -> None:
        resp = self.client.get("/users/non-existent-user/memories")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"user_id": "non-existent-user", "memories": []})

    def test_ephemeral_messages_are_not_saved_as_long_term_memories(self) -> None:
        for msg in (
            "Hello, how can you help me today?",
            "What should I focus on for my career?",
            "Why do you say that?",
            "Thanks!",
        ):
            resp = self.client.post(
                "/chat",
                json={"user_id": "user-ephemeral", "session_id": "s1", "message": msg},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["memories_saved"], [])

        mem_resp = self.client.get("/users/user-ephemeral/memories")
        self.assertEqual(mem_resp.status_code, 200)
        self.assertEqual(mem_resp.json()["memories"], [])

    def test_memory_saved_and_retrieved_across_sessions(self) -> None:
        resp1 = self.client.post(
            "/chat",
            json={
                "user_id": "user-123",
                "session_id": "session-1",
                "message": "I'm planning to switch jobs next year, and I prefer advice in Hindi.",
            },
        )
        self.assertEqual(resp1.status_code, 200)
        saved = set(resp1.json()["memories_saved"])
        self.assertEqual(saved, {"career_goal", "language_preference"})

        resp2 = self.client.post(
            "/chat",
            json={
                "user_id": "user-123",
                "session_id": "session-2",
                "message": "What do you remember about my career plans?",
            },
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertIn("career_goal", data2["context_used"])
        self.assertIn("switch jobs next year", data2["response"].lower())

        # Session 2 must NOT receive Session 1's raw conversation history
        session2_messages = self.spy_llm.calls[-1]["messages"]
        self.assertEqual(len(session2_messages), 1)
        self.assertEqual(session2_messages[0]["content"], "What do you remember about my career plans?")

    def test_user_isolation_prevents_cross_user_memory_leakage(self) -> None:
        self.client.post(
            "/chat",
            json={
                "user_id": "alice",
                "session_id": "s1",
                "message": "I'm planning to switch jobs next year.",
            },
        )
        resp_bob = self.client.post(
            "/chat",
            json={
                "user_id": "bob",
                "session_id": "s1",
                "message": "What do you remember about my career plans?",
            },
        )
        self.assertEqual(resp_bob.status_code, 200)
        self.assertEqual(resp_bob.json()["context_used"], [])
        self.assertNotIn("switch jobs", str(self.spy_llm.calls[-1]["system_prompt"]).lower())

    def test_sliding_window_history_drops_oldest_turns_and_preserves_order(self) -> None:
        # max_history_messages=4 (2 prior turns). Send 4 turns so Turn 1 must be evicted.
        turns = [
            "Turn 1 unique marker alpha",
            "Turn 2 unique marker beta",
            "Turn 3 unique marker gamma",
            "Why do you say that?",
        ]
        for msg in turns:
            resp = self.client.post(
                "/chat",
                json={"user_id": "user-window", "session_id": "sess-w", "message": msg},
            )
            self.assertEqual(resp.status_code, 200)

        sent_messages = self.spy_llm.calls[-1]["messages"]
        # 4 history messages (Turn 2 user/assistant + Turn 3 user/assistant) + 1 current = 5
        self.assertEqual(len(sent_messages), 5)
        self.assertEqual([m["role"] for m in sent_messages], ["user", "assistant", "user", "assistant", "user"])

        joined_contents = " ".join(m["content"] for m in sent_messages)
        self.assertNotIn("alpha", joined_contents)
        self.assertIn("beta", joined_contents)
        self.assertIn("gamma", joined_contents)
        self.assertEqual(sent_messages[-1]["content"], "Why do you say that?")

    def test_irrelevant_memories_excluded_from_llm_context(self) -> None:
        self.client.post(
            "/chat",
            json={
                "user_id": "user-3",
                "session_id": "sess-1",
                "message": "I love gardening and baking sourdough bread.",
            },
        )
        self.client.post(
            "/chat",
            json={
                "user_id": "user-3",
                "session_id": "sess-1",
                "message": "I'm planning to switch jobs next year.",
            },
        )

        resp = self.client.post(
            "/chat",
            json={
                "user_id": "user-3",
                "session_id": "sess-2",
                "message": "What should I focus on for my career?",
            },
        )
        self.assertEqual(resp.status_code, 200)
        context_used = resp.json()["context_used"]
        self.assertIn("career_goal", context_used)
        self.assertFalse(any(k.startswith("interest_") for k in context_used))

        last_system_prompt = str(self.spy_llm.calls[-1]["system_prompt"]).lower()
        self.assertIn("switch jobs next year", last_system_prompt)
        self.assertNotIn("gardening", last_system_prompt)
        self.assertNotIn("sourdough", last_system_prompt)

    def test_memory_update_on_user_correction(self) -> None:
        self.client.post(
            "/chat",
            json={"user_id": "user-4", "session_id": "s1", "message": "I prefer advice in Hindi."},
        )
        self.client.post(
            "/chat",
            json={"user_id": "user-4", "session_id": "s1", "message": "Actually, I prefer advice in English now."},
        )

        memories = self.client.get("/users/user-4/memories").json()["memories"]
        lang_memories = [m for m in memories if m["memory_key"] == "language_preference"]
        self.assertEqual(len(lang_memories), 1)
        self.assertIn("English", lang_memories[0]["content"])

    def test_resilient_llm_fallback_and_no_orphaned_turn_on_unhandled_failure(self) -> None:
        # 1. ResilientLLMProvider falls back cleanly when primary raises
        resilient = ResilientLLMProvider(primary=FailingLLMProvider(), fallback=MockLLMProvider())
        resilient_app = create_app(settings=self.settings, llm_provider=resilient)
        resilient_client = TestClient(resilient_app)

        ok_resp = resilient_client.post(
            "/chat",
            json={"user_id": "u-res", "session_id": "s1", "message": "I'm planning to switch jobs next year."},
        )
        self.assertEqual(ok_resp.status_code, 200)
        self.assertIn("switch jobs next year", ok_resp.json()["response"].lower())

        # 2. If raw LLM fails without wrapper, no orphaned user message is persisted
        raw_fail_app = create_app(settings=self.settings, llm_provider=FailingLLMProvider())
        raw_fail_client = TestClient(raw_fail_app, raise_server_exceptions=False)
        fail_resp = raw_fail_client.post(
            "/chat",
            json={"user_id": "u-fail", "session_id": "s1", "message": "Hello"},
        )
        self.assertEqual(fail_resp.status_code, 500)
        self.assertEqual(raw_fail_app.state.db.get_recent_messages("u-fail", "s1", limit=10), [])


class TestMemoryEngineUnit(unittest.TestCase):
    def test_extract_astrology_and_location_facts(self) -> None:
        extracted = extract_memories("I live in Bengaluru, and my zodiac sign is Leo.")
        keys = {m.memory_key for m in extracted}
        self.assertIn("location", keys)
        self.assertIn("astrology_profile", keys)

    def test_top_k_cap_and_relevance_ranking(self) -> None:
        memories = [
            {
                "memory_key": "career_goal",
                "category": "career_goal",
                "content": "Planning to switch jobs next year.",
                "keywords": "career job switch next year",
                "importance": 0.9,
            },
            {
                "memory_key": "health_goal",
                "category": "health_goal",
                "content": "Want to focus on running and diet.",
                "keywords": "health fitness running diet",
                "importance": 0.8,
            },
            {
                "memory_key": "interest_music",
                "category": "interest",
                "content": "I enjoy classical guitar.",
                "keywords": "interest hobby classical guitar",
                "importance": 0.65,
            },
        ]
        selected, keys = retrieve_relevant_memories(
            query="How should I prepare for my job switch?",
            memories=memories,
            top_k=1,
            min_score=0.2,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(keys, ["career_goal"])

    def test_generic_recall_vs_scoped_recall(self) -> None:
        memories = [
            {
                "memory_key": "career_goal",
                "category": "career_goal",
                "content": "Planning to switch jobs next year.",
                "keywords": "career job switch next year",
                "importance": 0.9,
            }
        ]
        # Generic recall ("What do you remember about me?") returns stored memories
        _, generic_keys = retrieve_relevant_memories(
            query="What do you remember about me?",
            memories=memories,
            top_k=5,
            min_score=0.2,
        )
        self.assertEqual(generic_keys, ["career_goal"])

        # Scoped recall for an unrelated topic excludes career_goal
        _, scoped_keys = retrieve_relevant_memories(
            query="What do you remember about my health?",
            memories=memories,
            top_k=5,
            min_score=0.2,
        )
        self.assertEqual(scoped_keys, [])

    def test_language_preference_preserved_when_top_k_saturated(self) -> None:
        memories = [
            {
                "memory_key": "career_goal",
                "category": "career_goal",
                "content": "Planning to switch jobs next year.",
                "keywords": "career job switch",
                "importance": 0.9,
            },
            {
                "memory_key": "general_goal",
                "category": "goal",
                "content": "Goal is to get a promotion at work.",
                "keywords": "goal promotion work career",
                "importance": 0.8,
            },
            {
                "memory_key": "language_preference",
                "category": "language_preference",
                "content": "User prefers responses/advice in Hindi.",
                "keywords": "language preference hindi",
                "importance": 0.95,
            },
        ]
        _, keys = retrieve_relevant_memories(
            query="What should I focus on for my career?",
            memories=memories,
            top_k=1,
            min_score=0.2,
        )
        self.assertIn("career_goal", keys)
        self.assertIn("language_preference", keys)


class TestLLMProvidersUnit(unittest.TestCase):
    def test_openai_provider_payload_and_error_handling(self) -> None:
        captured: Dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Focus on system design."}}]},
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        with patch("app.llm.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)):
            provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini", base_url="https://api.example.com/v1")
            reply = provider.generate("System instructions", [{"role": "user", "content": "Hi"}])

        self.assertEqual(reply, "Focus on system design.")
        self.assertEqual(captured["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer sk-test")
        self.assertEqual(
            captured["body"]["messages"],
            [{"role": "system", "content": "System instructions"}, {"role": "user", "content": "Hi"}],
        )

        # Verify non-2xx raises HTTPStatusError
        err_transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "invalid_api_key"}))
        with patch("app.llm.httpx.Client", side_effect=lambda **kw: real_client(transport=err_transport, **kw)):
            with self.assertRaises(httpx.HTTPStatusError):
                provider.generate("System", [{"role": "user", "content": "Hi"}])

    def test_gemini_provider_payload_and_role_mapping(self) -> None:
        captured: Dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "नमस्ते! करियर पर ध्यान दें।"}]}}]},
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        with patch("app.llm.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)):
            provider = GeminiLLMProvider(api_key="AIza-test", model="gemini-3.8-flash")
            reply = provider.generate(
                "Respond in Hindi.",
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Career advice?"},
                ],
            )

        self.assertEqual(reply, "नमस्ते! करियर पर ध्यान दें।")
        self.assertIn("models/gemini-3.8-flash:generateContent", str(captured["url"]))
        self.assertIn("key=AIza-test", str(captured["url"]))
        self.assertEqual(captured["body"]["system_instruction"], {"parts": [{"text": "Respond in Hindi."}]})
        self.assertEqual(
            [c["role"] for c in captured["body"]["contents"]],
            ["user", "model", "user"],
        )


if __name__ == "__main__":
    unittest.main()