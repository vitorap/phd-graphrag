from __future__ import annotations

import requests

from app.config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.num_ctx = settings.ollama_num_ctx
        self.timeout = settings.ollama_timeout

    def chat(self, messages: list[dict[str, str]], model: str | None = None, timeout: int | None = None) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": self.num_ctx,
                "num_predict": 320,
            },
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout or self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
