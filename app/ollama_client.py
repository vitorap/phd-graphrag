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

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        embed_model = model or settings.ollama_embed_model
        payload = {"model": embed_model, "input": texts}
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=timeout or max(self.timeout, 120),
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list):
                return embeddings
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code not in {400, 404}:
                raise

        embeddings: list[list[float]] = []
        for text in texts:
            legacy = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": embed_model, "prompt": text},
                timeout=timeout or max(self.timeout, 120),
            )
            legacy.raise_for_status()
            data = legacy.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Ollama nao retornou embedding para modelo {embed_model}")
            embeddings.append(embedding)
        return embeddings

    def list_models(self, timeout: int = 5) -> list[dict[str, str | int | None]]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        models: list[dict[str, str | int | None]] = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            name = item.get("name") or item.get("model")
            if not name:
                continue
            models.append(
                {
                    "name": name,
                    "size": item.get("size"),
                    "family": details.get("family"),
                    "parameterSize": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                }
            )
        models.sort(key=lambda model: (model["name"] != self.model, str(model["name"])))
        return models
