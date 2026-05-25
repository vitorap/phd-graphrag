from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graphrag import GraphRAG
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="GraphRAG em Middle-earth")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    hops: int = Field(2, ge=1, le=4)
    mode: str = Field("hybrid", pattern="^(rag|graph|hybrid|baseline)$")
    model: str | None = None
    use_llm: bool = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    client = Neo4jClient()
    try:
        ok = client.ping()
    finally:
        client.close()
    return {"ok": ok}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.stats()
    finally:
        client.close()


@app.get("/api/graph")
def graph(
    center: str | None = None,
    hops: int = Query(1, ge=1, le=4),
    limit: int = Query(160, ge=20, le=260),
) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.graph(center=center, hops=hops, limit=limit)
    finally:
        client.close()


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.answer(
            payload.question,
            hops=payload.hops,
            mode=payload.mode,
            model=payload.model,
            use_llm=payload.use_llm,
        )
    finally:
        client.close()


@app.post("/api/compare")
def compare(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.compare(
            payload.question,
            hops=payload.hops,
            model=payload.model,
            use_llm=payload.use_llm,
        )
    finally:
        client.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
