from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.graphrag import GraphRAG
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="GraphRAG em Middle-earth")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    hops: int = Field(2, ge=1, le=4)
    top_k: int = Field(8, ge=1, le=24)
    mode: str = Field("hybrid", pattern="^(rag|graph|hybrid|baseline)$")
    model: str | None = None
    use_llm: bool = False


class VectorSearchRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(8, ge=1, le=24)
    mode: str = Field("rag", pattern="^(rag|hybrid)$")
    hops: int = Field(2, ge=1, le=4)
    source_type: str | None = Field(None, pattern="^(book|dialogue)$")


class CypherRequest(BaseModel):
    query: str = Field(..., min_length=3)
    limit: int = Field(100, ge=1, le=300)


CYPHER_EXAMPLES = [
    {
        "id": "frodo-neighbors",
        "title": "Vizinhos de Frodo",
        "explain": "Mostra entidades conectadas diretamente a Frodo e o tipo de relacao.",
        "query": """
MATCH (:Entity {name: 'Frodo'})-[r]-(n:Entity)
RETURN n.name AS entidade, labels(n) AS labels, type(r) AS relacao,
       coalesce(r.weight, r.confidence, 1) AS peso
ORDER BY peso DESC, entidade
LIMIT 20
""".strip(),
    },
    {
        "id": "frodo-sauron-path",
        "title": "Caminho Frodo-Sauron",
        "explain": "Mostra o menor caminho estrutural entre a pergunta classica da demo.",
        "query": """
MATCH p = shortestPath((a:Entity {name: 'Frodo'})-[*..5]-(b:Entity {name: 'Sauron'}))
RETURN [n IN nodes(p) | n.name] AS caminho,
       [r IN relationships(p) | type(r)] AS relacoes,
       length(p) AS saltos
""".strip(),
    },
    {
        "id": "pagerank",
        "title": "Top PageRank",
        "explain": "Centralidade estrutural calculada no grafo personagem-personagem.",
        "query": """
MATCH (c:Character)
RETURN c.name AS personagem, c.race AS raca,
       round(coalesce(c.pagerank, 0), 5) AS pagerank,
       round(coalesce(c.weightedDegree, 0), 1) AS grau_ponderado,
       c.community AS comunidade
ORDER BY pagerank DESC
LIMIT 15
""".strip(),
    },
    {
        "id": "relationship-types",
        "title": "Tipos de relacao",
        "explain": "Ajuda a explicar que o grafo mistura ontologia, coocorrencia, falas e texto.",
        "query": """
MATCH ()-[r]->()
RETURN type(r) AS relacao, count(r) AS total
ORDER BY total DESC
LIMIT 20
""".strip(),
    },
    {
        "id": "strong-cooccurrence",
        "title": "Coocorrencias fortes",
        "explain": "Mostra as arestas ponderadas usadas como estrutura social/narrativa.",
        "query": """
MATCH (a:Character)-[r:CO_OCCURS_WITH]-(b:Character)
WHERE a.name < b.name
RETURN a.name AS origem, b.name AS destino, r.weight AS peso
ORDER BY peso DESC
LIMIT 20
""".strip(),
    },
]


LECTURE_STEPS = [
    {
        "id": "opening",
        "title": "Abertura: por que LOTR GraphRAG?",
        "duration": "3 min",
        "mode": "overview",
        "question": "Como Frodo se conecta a Sauron?",
        "talkingPoints": [
            "O corpus mistura texto completo, scripts, ontologia e redes de personagens.",
            "A pergunta Frodo-Sauron e boa porque exige narrativa e estrutura.",
        ],
        "quiz": {
            "question": "Qual informacao o texto sabe melhor que o grafo?",
            "answer": "A explicacao narrativa: missao, Anel, Mordor e motivacoes.",
        },
    },
    {
        "id": "rag",
        "title": "RAG vetorial: similaridade sem estrutura explicita",
        "duration": "7 min",
        "mode": "rag",
        "question": "Por que o Anel importa para Frodo?",
        "talkingPoints": [
            "Chunks e falas viram vetores de embedding.",
            "A pergunta tambem vira vetor; a busca usa similaridade coseno.",
            "RAG encontra passagens boas, mas nao sabe caminhos estruturais por si so.",
        ],
        "quiz": {
            "question": "O que muda entre BM25 e embedding retrieval?",
            "answer": "BM25 depende de termos; embedding retrieval aproxima significado no espaco vetorial.",
        },
    },
    {
        "id": "graph",
        "title": "Graph: vizinhanca, caminhos e centralidade",
        "duration": "8 min",
        "mode": "graph",
        "question": "Qual o caminho estrutural entre Frodo e Sauron?",
        "talkingPoints": [
            "k-hop define o campo receptivo, como em message passing.",
            "PageRank e comunidades resumem posicao estrutural.",
            "O grafo explica conexoes, mas nao substitui a narrativa textual.",
        ],
        "quiz": {
            "question": "Por que aumentar hops pode piorar a resposta?",
            "answer": "Mais hops aumentam cobertura, mas tambem trazem ruido e entidades pouco relevantes.",
        },
    },
    {
        "id": "graphrag",
        "title": "GraphRAG: estrutura guia a recuperacao textual",
        "duration": "12 min",
        "mode": "hybrid",
        "question": "O que Frodo e de Sauron?",
        "talkingPoints": [
            "Primeiro detectamos entidades.",
            "Depois expandimos k-hop no grafo.",
            "Por fim, buscamos evidencias vetoriais com boost do subgrafo.",
        ],
        "quiz": {
            "question": "Onde a ideia de GNN aparece no GraphRAG?",
            "answer": "Na expansao k-hop: informacao de vizinhos e caminhos vira contexto para a resposta.",
        },
    },
    {
        "id": "compare",
        "title": "Comparacao final: RAG vs Graph vs GraphRAG",
        "duration": "10 min",
        "mode": "compare",
        "question": "Qual a relacao de Frodo com Sauron?",
        "talkingPoints": [
            "RAG textual recupera narrativa.",
            "Graph recupera estrutura.",
            "GraphRAG combina as duas evidencias.",
        ],
        "quiz": {
            "question": "Qual modo voce usaria para pergunta causal/narrativa com entidades conhecidas?",
            "answer": "GraphRAG, porque usa o grafo para selecionar contexto e o texto para explicar.",
        },
    },
]


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


@app.get("/api/vector/status")
def vector_status() -> dict[str, Any]:
    stats = VectorStore().stats()
    stats["defaultEmbeddingModel"] = settings.ollama_embed_model
    return stats


@app.get("/api/models")
def models() -> dict[str, Any]:
    ollama = OllamaClient()
    try:
        return {
            "ok": True,
            "defaultModel": ollama.model,
            "models": ollama.list_models(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "defaultModel": ollama.model,
            "models": [{"name": ollama.model}],
            "error": str(exc),
        }


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


@app.post("/api/vector-search")
def vector_search(payload: VectorSearchRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient())
        entities = rag.resolve_entities(payload.question)
        graph = (
            client.subgraph_for_seeds(entities, hops=payload.hops, limit=180)
            if payload.mode == "hybrid" and entities
            else {}
        )
        graph_names = [
            node["name"]
            for node in graph.get("nodes", [])
            if node.get("name") and node.get("name") not in entities
        ]
        docs = VectorStore().search(
            payload.question,
            OllamaClient(),
            model=settings.ollama_embed_model,
            limit=payload.top_k,
            seed_entities=entities,
            graph_entities=graph_names,
            source_type=payload.source_type,
        )
        return {
            "question": payload.question,
            "entities": entities,
            "graphEntities": graph_names,
            "documents": docs,
            "embeddingModel": settings.ollama_embed_model,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        client.close()


@app.get("/api/cypher/examples")
def cypher_examples() -> dict[str, Any]:
    return {"examples": CYPHER_EXAMPLES}


@app.post("/api/cypher/run")
def cypher_run(payload: CypherRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.run_readonly_cypher(payload.query, limit=payload.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        client.close()


@app.get("/api/lecture")
def lecture() -> dict[str, Any]:
    return {"steps": LECTURE_STEPS}


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.answer(
            payload.question,
            hops=payload.hops,
            top_k=payload.top_k,
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
            top_k=payload.top_k,
            model=payload.model,
            use_llm=payload.use_llm,
        )
    finally:
        client.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
