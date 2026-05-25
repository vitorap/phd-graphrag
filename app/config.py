from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "graphrag-lotr")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.6:latest")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    vector_dir: str = os.getenv("VECTOR_DIR", "")


settings = Settings()
