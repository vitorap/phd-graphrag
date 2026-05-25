from __future__ import annotations

import re
import unicodedata


STOPWORDS = {
    "a",
    "about",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "for",
    "from",
    "is",
    "o",
    "os",
    "para",
    "por",
    "qual",
    "que",
    "the",
    "to",
    "what",
    "with",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    normalized = normalize(value)
    return [token for token in normalized.split() if len(token) > 2 and token not in STOPWORDS]


def snippet(text: str, query_tokens: list[str], max_chars: int = 760) -> str:
    if len(text) <= max_chars:
        return text
    lower = normalize(text)
    first_hit = -1
    for token in query_tokens:
        first_hit = lower.find(token)
        if first_hit >= 0:
            break
    if first_hit < 0:
        return text[:max_chars].strip() + "..."
    start = max(0, first_hit - max_chars // 3)
    end = min(len(text), start + max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
