import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "what", "which", "who", "whom", "this", "that", "these",
    "those", "do", "does", "did", "doing", "have", "has", "had", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against", "during", "without",
    "before", "under", "around", "among", "and", "or", "but", "if", "because",
    "so", "than", "too", "very", "just", "now", "then", "there", "here", "when",
    "where", "why", "how", "all", "any", "both", "each", "more", "most", "some",
    "no", "not", "only", "own", "same", "also", "please", "tell", "say", "said",
}

INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "career_goal": {
        "career", "job", "jobs", "work", "working", "switch", "switching",
        "promotion", "promote", "salary", "hike", "interview", "office",
        "business", "startup", "company", "role", "profession", "professional",
        "resign", "offer", "layoff",
    },
    "relationship_goal": {
        "relationship", "marriage", "marry", "married", "partner", "love",
        "dating", "spouse", "wedding", "wife", "husband", "girlfriend",
        "boyfriend", "compatibility",
    },
    "health_goal": {
        "health", "fitness", "workout", "gym", "diet", "weight", "sleep",
        "stress", "mental", "physical", "wellness", "yoga",
    },
    "astrology_profile": {
        "astrology", "horoscope", "kundli", "birth", "born", "zodiac", "rashi",
        "nakshatra", "lagna", "ascendant", "chart", "planet", "transit",
        "dasha", "sign", "sun", "moon",
    },
    "personal_fact": {
        "live", "living", "city", "location", "home", "based", "age", "name",
        "family", "background",
    },
    "interest": {
        "interest", "interested", "hobby", "hobbies", "enjoy", "love", "like",
        "passion", "favorite", "favourite", "free", "time",
    },
}

RECALL_TRIGGERS: Set[str] = {
    "remember", "recall", "know", "saved", "memory", "memories", "told", "mentioned",
}

SUPPORTED_LANGUAGES: List[str] = [
    "hindi", "english", "hinglish", "tamil", "telugu", "kannada", "malayalam",
    "marathi", "bengali", "gujarati", "punjabi", "spanish", "french", "german",
]


@dataclass(frozen=True)
class ExtractedMemory:
    memory_key: str
    category: str
    content: str
    keywords: str
    importance: float = 0.7


def tokenize(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}


def _split_clauses(message: str) -> List[str]:
    parts = re.split(r"(?:[.!?\n]+|,\s+and\s+|\s+and\s+(?=i\s|my\s))", message, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def _looks_like_pure_question(clause: str, original_message: str) -> bool:
    stripped = clause.strip().lower()
    question_starters = (
        "what ", "why ", "how ", "when ", "where ", "who ", "which ",
        "can you", "could you", "should i", "do you", "did i", "tell me",
    )
    if stripped.startswith(question_starters):
        return True
    if original_message.strip().endswith("?") and not any(
        marker in stripped
        for marker in ("i'm ", "i am ", "i prefer", "i want ", "i plan", "my goal", "i live", "i was born")
    ):
        return True
    return False


def extract_memories(message: str) -> List[ExtractedMemory]:
    """
    Extract durable user facts, goals, preferences, and interests from a message.
    Questions, greetings, and ephemeral remarks are intentionally ignored.
    """
    if not message or len(message.strip()) < 4:
        return []

    extracted: Dict[str, ExtractedMemory] = {}

    for clause in _split_clauses(message):
        lower = clause.lower()
        if _looks_like_pure_question(clause, message):
            continue

        # 1. Language preference (supports updates/corrections via same memory_key)
        lang_match = re.search(
            r"(?:prefer|want|give|speak|reply|respond|advice|talk|chat)\b[^.?!]*?\bin\s+("
            + "|".join(SUPPORTED_LANGUAGES)
            + r")\b",
            lower,
        )
        if not lang_match:
            lang_match = re.search(
                r"\b(" + "|".join(SUPPORTED_LANGUAGES) + r")\s+(?:is\s+my\s+preferred\s+language|preferred)\b",
                lower,
            )
        if lang_match:
            lang = lang_match.group(1).capitalize()
            extracted["language_preference"] = ExtractedMemory(
                memory_key="language_preference",
                category="language_preference",
                content=f"User prefers responses/advice in {lang}.",
                keywords=f"language preference {lang.lower()} speak reply advice",
                importance=0.95,
            )
            continue

        # 2. Career goals & plans
        if any(
            kw in lower
            for kw in (
                "switch job", "switch jobs", "change job", "new job", "job hunt",
                "career", "promotion", "startup", "business", "interview", "resign",
                "get promoted", "tech lead", "engineer",
            )
        ) and any(
            intent in lower
            for intent in (
                "plan", "planning", "want", "goal", "aim", "trying", "hope",
                "looking to", "decided", "going to", "preparing", "focus",
            )
        ):
            extracted["career_goal"] = ExtractedMemory(
                memory_key="career_goal",
                category="career_goal",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"career", "job", "goal", "plan"})),
                importance=0.9,
            )
            continue

        # 3. Relationship goals
        if any(kw in lower for kw in ("marriage", "marry", "partner", "relationship", "wedding")) and any(
            intent in lower for intent in ("plan", "planning", "want", "goal", "looking", "getting")
        ):
            extracted["relationship_goal"] = ExtractedMemory(
                memory_key="relationship_goal",
                category="relationship_goal",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"relationship", "marriage", "goal"})),
                importance=0.85,
            )
            continue

        # 4. Health & wellness goals
        if any(kw in lower for kw in ("health", "fitness", "weight", "workout", "diet", "yoga")) and any(
            intent in lower for intent in ("plan", "want", "goal", "trying", "focus", "starting")
        ):
            extracted["health_goal"] = ExtractedMemory(
                memory_key="health_goal",
                category="health_goal",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"health", "fitness", "goal"})),
                importance=0.8,
            )
            continue

        # 5. Astrology profile details (DOB, zodiac/rashi, birth place/time)
        if any(
            kw in lower
            for kw in ("born on", "birth date", "dob", "zodiac", "rashi", "nakshatra", "ascendant", "lagna")
        ):
            extracted["astrology_profile"] = ExtractedMemory(
                memory_key="astrology_profile",
                category="astrology_profile",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"astrology", "birth", "zodiac", "rashi"})),
                importance=0.9,
            )
            continue

        # 6. Personal facts (location, occupation, name)
        if re.search(r"\b(?:i live in|i'm from|i am from|i'm based in|i work as|my name is)\b", lower):
            key = "location" if any(w in lower for w in ("live", "from", "based")) else "personal_fact"
            extracted[key] = ExtractedMemory(
                memory_key=key,
                category="personal_fact",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"personal", "fact", key})),
                importance=0.8,
            )
            continue

        # 7. Interests & hobbies
        interest_match = re.search(
            r"\b(?:i love|i enjoy|i'm interested in|i am interested in|my hobby is|my hobbies are)\s+([a-z0-9\s,&-]+)",
            lower,
        )
        if interest_match:
            topic_tokens = sorted(tokenize(interest_match.group(1)))
            suffix = "_".join(topic_tokens[:2]) if topic_tokens else "general"
            memory_key = f"interest_{suffix}"
            extracted[memory_key] = ExtractedMemory(
                memory_key=memory_key,
                category="interest",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"interest", "hobby"})),
                importance=0.65,
            )
            continue

        # 8. Generic explicit goal/plan fallback
        if re.search(r"\b(?:my goal is|i'm planning to|i am planning to|i plan to)\b", lower):
            extracted["general_goal"] = ExtractedMemory(
                memory_key="general_goal",
                category="goal",
                content=clause,
                keywords=" ".join(sorted(tokenize(clause) | {"goal", "plan"})),
                importance=0.75,
            )

    return list(extracted.values())


def retrieve_relevant_memories(
    query: str,
    memories: List[Dict[str, object]],
    top_k: int = 5,
    min_score: float = 0.2,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Score and select only memories relevant to the current query.
    Global response preferences (like language_preference) are included when present.
    """
    if not memories:
        return [], []

    query_tokens = tokenize(query)
    is_broad_recall = bool(query_tokens & RECALL_TRIGGERS)

    # Detect matching domain intents from the query
    matched_categories: Set[str] = set()
    for category, kw_set in INTENT_KEYWORDS.items():
        if query_tokens & kw_set:
            matched_categories.add(category)

    scored: List[Tuple[float, Dict[str, object]]] = []
    always_include: List[Dict[str, object]] = []

    for mem in memories:
        key = str(mem["memory_key"])
        category = str(mem["category"])
        mem_tokens = tokenize(f"{key} {mem['keywords']} {mem['content']}")
        importance = float(mem.get("importance", 0.7))

        # Language preference governs response style across queries
        if key == "language_preference":
            always_include.append(mem)
            continue

        overlap = len(query_tokens & mem_tokens)
        union = len(query_tokens | mem_tokens) or 1
        lexical_score = overlap / len(query_tokens) if query_tokens else 0.0
        jaccard_score = overlap / union

        category_boost = 0.55 if category in matched_categories else 0.0

        # If user asks "What do you remember about me?" with no specific topic,
        # allow high-importance memories; if they ask "What do you remember about my career?",
        # require topic/category match so unrelated memories stay excluded.
        topic_tokens = query_tokens - RECALL_TRIGGERS
        if is_broad_recall and not topic_tokens:
            recall_boost = 0.35
        elif is_broad_recall and (overlap > 0 or category in matched_categories):
            recall_boost = 0.25
        else:
            recall_boost = 0.0

        total_score = (
            (0.45 * lexical_score)
            + (0.15 * jaccard_score)
            + category_boost
            + recall_boost
            + (0.1 * importance if (overlap > 0 or category_boost > 0 or recall_boost > 0) else 0.0)
        )

        if total_score >= min_score:
            scored.append((total_score, mem))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [mem for _, mem in scored[:top_k]]

    # Append global language preference if space permits (or include it alongside top matches)
    for pref in always_include:
        if len(selected) < top_k or pref not in selected:
            selected.append(pref)

    # Deduplicate preserving order
    seen_keys: Set[str] = set()
    final_memories: List[Dict[str, object]] = []
    for mem in selected:
        k = str(mem["memory_key"])
        if k not in seen_keys:
            seen_keys.add(k)
            final_memories.append(mem)

    context_used = [str(m["memory_key"]) for m in final_memories]
    return final_memories, context_used
