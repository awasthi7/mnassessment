import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for swappable LLM backends."""

    @abstractmethod
    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Generate an assistant response given a system prompt and recent message history."""


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible Chat Completions provider (works with OpenAI, Groq, Together, Ollama)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class GeminiLLMProvider(LLMProvider):
    """Google Gemini REST API provider via httpx."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.model = model or "gemini-2.5-flash"
        self.timeout_seconds = timeout_seconds

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.4},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, params={"key": self.api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


class MockLLMProvider(LLMProvider):
    """
    Deterministic, context-aware mock LLM for automated tests and offline local runs.
    Reads only the memories and history explicitly passed into the prompt.
    """

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        latest_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                latest_user_msg = msg["content"]
                break

        memories = self._parse_memories(system_prompt)
        lang_pref = memories.get("language_preference", "")
        prefer_hindi = "hindi" in lang_pref.lower()

        lower_q = latest_user_msg.lower()

        # Handle follow-up questions using recent conversation history
        if any(phrase in lower_q for phrase in ("why do you say that", "explain that", "what do you mean")):
            prior_assistant = [m["content"] for m in messages[:-1] if m["role"] == "assistant"]
            if prior_assistant:
                last_reply = prior_assistant[-1]
                if prefer_hindi:
                    return (
                        f"पिछली बात के आधार पर ({last_reply[:120]}), यह सुझाव आपकी वर्तमान स्थिति "
                        "और प्राथमिकताओं को ध्यान में रखकर दिया गया है।"
                    )
                return (
                    f"I said that because of my previous point: \"{last_reply[:140]}\" — "
                    "it connects directly to what you shared earlier in this conversation."
                )

        # Recall queries ("What do you remember about...")
        if any(kw in lower_q for kw in ("remember", "know about me", "what did i tell")):
            relevant_items = [v for k, v in memories.items() if k != "language_preference"]
            if not relevant_items:
                return (
                    "अभी मेरे पास आपके बारे में कोई सहेजी गई जानकारी नहीं है।"
                    if prefer_hindi
                    else "I don't have any saved memories matching that yet."
                )
            joined = "; ".join(relevant_items)
            if prefer_hindi:
                return f"मुझे याद है कि आपने बताया था: {joined}"
            return f"Here is what I remember from our earlier conversations: {joined}"

        # Personalized guidance using retrieved memories
        non_lang_memories = [(k, v) for k, v in memories.items() if k != "language_preference"]
        if non_lang_memories:
            mem_summary = "; ".join(v for _, v in non_lang_memories)
            if prefer_hindi:
                return (
                    f"आपकी सहेजी गई जानकारी ({mem_summary}) को ध्यान में रखते हुए, "
                    "अपने मुख्य लक्ष्य के लिए सही समय, कौशल निर्माण और निरंतर प्रयास पर ध्यान दें।"
                )
            return (
                f"Based on what you've shared ({mem_summary}), focus on clear milestones, "
                "skill alignment, and steady preparation toward your goal."
            )

        if prefer_hindi:
            return "मैं आपकी मदद के लिए तैयार हूँ। कृपया अपनी स्थिति या प्रश्न के बारे में थोड़ा और बताएं।"
        return "I'm here to help! Share a bit more about what you're looking for, and we can explore it together."

    @staticmethod
    def _parse_memories(system_prompt: str) -> Dict[str, str]:
        memories: Dict[str, str] = {}
        for line in system_prompt.splitlines():
            match = re.match(r"^- \[([^\]]+)\]:\s*(.+)$", line.strip())
            if match:
                memories[match.group(1)] = match.group(2)
        return memories


class ResilientLLMProvider(LLMProvider):
    """Wraps a primary LLM provider with a safe fallback if the upstream API fails."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        try:
            return self.primary.generate(system_prompt, messages)
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            logger.warning("Primary LLM failed (%s); falling back to safe provider.", exc)
            return self.fallback.generate(system_prompt, messages)


def build_llm_provider(settings: Settings) -> LLMProvider:
    fallback = MockLLMProvider()
    provider_name = settings.llm_provider

    if provider_name == "openai" and settings.llm_api_key:
        primary: LLMProvider = OpenAILLMProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gpt-4o-mini",
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        return ResilientLLMProvider(primary=primary, fallback=fallback)

    if provider_name == "gemini" and settings.llm_api_key:
        primary = GeminiLLMProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model or "gemini-2.5-flash",
            timeout_seconds=settings.llm_timeout_seconds,
        )
        return ResilientLLMProvider(primary=primary, fallback=fallback)

    return fallback
