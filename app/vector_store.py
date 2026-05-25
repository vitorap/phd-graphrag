from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.text_utils import snippet, tokenize
from app.ollama_client import OllamaClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DIR = ROOT / "data" / "vector_store"


def vector_dir() -> Path:
    return Path(settings.vector_dir) if settings.vector_dir else DEFAULT_VECTOR_DIR


class VectorStore:
    def __init__(self, path: Path | None = None, collection: str = "lotr_retrieval_documents") -> None:
        self.path = path or vector_dir()
        self.collection = collection
        self.vectors_path = self.path / f"{collection}.npz"
        self.metadata_path = self.path / f"{collection}.metadata.json"
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None

    def exists(self) -> bool:
        return self.vectors_path.exists() and self.metadata_path.exists()

    def stats(self) -> dict[str, Any]:
        if not self.exists():
            return {
                "ready": False,
                "documents": 0,
                "dimensions": 0,
                "collection": self.collection,
                "path": str(self.path),
            }
        vectors, metadata = self.load()
        build = metadata[0].get("_index") if metadata and metadata[0].get("_index") else {}
        return {
            "ready": True,
            "documents": int(vectors.shape[0]),
            "dimensions": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            "collection": self.collection,
            "path": str(self.path),
            "embeddingModel": build.get("embeddingModel"),
            "builtAt": build.get("builtAt"),
        }

    def load(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self._vectors is None or self._metadata is None:
            if not self.exists():
                raise FileNotFoundError(
                    f"indice vetorial ausente em {self.path}. Rode `make vectors`."
                )
            data = np.load(self.vectors_path)
            vectors = data["vectors"].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1.0e-12, None)
            self._vectors = vectors
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._vectors, self._metadata

    def build(
        self,
        documents: list[dict[str, Any]],
        ollama: OllamaClient,
        model: str,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        if not documents:
            raise RuntimeError("nenhum RetrievalDocument encontrado no Neo4j")
        self.path.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            texts = [document_text(doc) for doc in batch]
            embeddings = ollama.embed(texts, model=model)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama retornou {len(embeddings)} embeddings para lote de {len(batch)} textos"
                )
            vectors.extend(embeddings)
            for doc in batch:
                rows.append(sanitized_metadata(doc))

        matrix = np.array(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(rows):
            raise RuntimeError("matriz de embeddings invalida")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1.0e-12, None)

        from datetime import datetime, timezone

        rows[0]["_index"] = {
            "embeddingModel": model,
            "builtAt": datetime.now(timezone.utc).isoformat(),
            "documents": len(rows),
            "dimensions": int(matrix.shape[1]),
        }
        np.savez_compressed(self.vectors_path, vectors=matrix)
        self.metadata_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._vectors = matrix
        self._metadata = rows
        return self.stats()

    def search(
        self,
        query: str,
        ollama: OllamaClient,
        model: str,
        limit: int = 8,
        seed_entities: list[str] | None = None,
        graph_entities: list[str] | None = None,
        source_type: str | None = None,
        apply_boost: bool = False,
    ) -> list[dict[str, Any]]:
        vectors, metadata = self.load()
        query_embedding = np.array(ollama.embed([query], model=model)[0], dtype=np.float32)
        query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1.0e-12)
        scores = vectors @ query_embedding
        seed_set = set(seed_entities or [])
        graph_set = set(graph_entities or [])
        candidates: list[dict[str, Any]] = []

        for idx, base_score in enumerate(scores):
            meta = metadata[idx]
            if source_type and meta.get("sourceType") != source_type:
                continue
            mentions = set(meta.get("mentions") or [])
            seed_hits = mentions & seed_set
            graph_hits = mentions & graph_set
            graph_boost = 0.0
            if apply_boost:
                graph_boost = len(seed_hits) * 0.08 + min(len(graph_hits), 8) * 0.015
                if len(seed_hits) >= 2:
                    graph_boost += 0.08
            final_score = float(base_score) + graph_boost
            doc = dict(meta)
            doc.pop("_index", None)
            doc["score"] = final_score
            doc["vectorScore"] = float(base_score)
            doc["graphBoost"] = graph_boost
            doc["retrievalMethod"] = "vector"
            doc["snippet"] = snippet(str(doc.get("text") or ""), tokenize(query), max_chars=780)
            candidates.append(doc)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:limit]


def document_text(doc: dict[str, Any]) -> str:
    source = doc.get("sourceTitle") or doc.get("sourceType") or "texto"
    chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
    speaker = f" / fala de {doc['speaker']}" if doc.get("speaker") else ""
    return f"{source}{chapter}{speaker}\n{doc.get('text') or ''}"


def sanitized_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "labels",
        "text",
        "sourceType",
        "sourceTitle",
        "chapterTitle",
        "sequence",
        "speaker",
        "lineNumber",
        "mentions",
    ]
    clean = {key: doc.get(key) for key in keys}
    clean["mentions"] = sorted(set(clean.get("mentions") or []))
    return clean
