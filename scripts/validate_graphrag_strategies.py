from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graphrag import GRAPH_RAG_STRATEGY_ORDER, GraphRAG
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.vector_store import VectorStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def trace_for(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("trace") or {}


def retrieval_for(result: dict[str, Any]) -> dict[str, Any]:
    return trace_for(result).get("retrieval") or {}


def grounding_for(result: dict[str, Any]) -> dict[str, Any]:
    return trace_for(result).get("grounding") or {}


def runtime_for(result: dict[str, Any]) -> dict[str, Any]:
    return ((trace_for(result).get("strategy") or {}).get("runtime")) or {}


def validate(question: str, hops: int, top_k: int, as_json: bool) -> int:
    vector_store = VectorStore()
    require(
        vector_store.exists(),
        f"indice vetorial ausente em {vector_store.path}; rode `make vectors` antes de `make smoke-strategies`",
    )

    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(), vector_store=vector_store)
        results = {
            strategy: rag.answer(
                question,
                hops=hops,
                top_k=top_k,
                mode="hybrid",
                use_llm=False,
                graph_rag_strategy=strategy,
            )
            for strategy in GRAPH_RAG_STRATEGY_ORDER
        }
    finally:
        client.close()

    methods: dict[str, str] = {}
    summary: dict[str, Any] = {}
    for strategy, result in results.items():
        trace = trace_for(result)
        retrieval = retrieval_for(result)
        grounding = grounding_for(result)
        runtime = runtime_for(result)
        top_docs = retrieval.get("topDocuments") or []
        method = str(retrieval.get("method") or "")
        methods[strategy] = method
        summary[strategy] = {
            "method": method,
            "scoreMode": retrieval.get("scoreMode"),
            "documents": retrieval.get("documents"),
            "nodes": (trace.get("graph") or {}).get("nodeCount"),
            "edges": (trace.get("graph") or {}).get("edgeCount"),
            "degraded": bool(runtime.get("degraded")),
            "fallbackStrategy": runtime.get("fallbackStrategy"),
        }

        require((trace.get("variant") or {}).get("id") == strategy, f"{strategy}: trace variant incorreta")
        require((retrieval.get("documents") or 0) > 0, f"{strategy}: nenhum documento recuperado")
        require(strategy in method, f"{strategy}: metodo nao identifica a variante: {method}")

        if strategy == "kg_index":
            require(not runtime.get("degraded"), "kg_index: nao deveria degradar")
            require(retrieval.get("scoreMode") == "cosine+graph-boost", "kg_index: deveria usar cosine+graph-boost")
            require((retrieval.get("boostedDocuments") or 0) > 0, "kg_index: deveria aplicar boost estrutural")

        if strategy == "vector_first":
            require(retrieval.get("scoreMode") == "cosine", "vector_first: deveria manter cosine puro")
            require((retrieval.get("boostedDocuments") or 0) == 0, "vector_first: nao deveria aplicar graph boost")
            require(grounding.get("derivedEntities"), "vector_first: deveria derivar sementes dos hits vetoriais")

        if strategy == "graph_filter":
            require(retrieval.get("scoreMode") == "cosine+graph-boost", "graph_filter: deveria ranquear candidatos filtrados")
            require(
                all((doc.get("seedHits") or doc.get("graphHits")) for doc in top_docs),
                "graph_filter: todo top doc deve mencionar seed ou no do subgrafo",
            )

        if strategy == "path":
            graph_trace = trace.get("graph") or {}
            require(not runtime.get("degraded"), "path: pergunta padrao deveria ter caminho/conector real")
            require(
                graph_trace.get("paths") or graph_trace.get("connectors"),
                "path: deveria retornar caminho curto ou conector",
            )
            require(
                any((doc.get("strategyFocusHits") or doc.get("seedHits")) for doc in top_docs),
                "path: top docs deveriam carregar hits de caminho ou de seeds",
            )

        if strategy == "community":
            require(not runtime.get("degraded"), "community: dataset padrao deveria ter comunidade")
            require(grounding.get("communityEntities"), "community: deveria expor entidades da comunidade")

        if strategy == "cypher":
            graph_trace = trace.get("graph") or {}
            require(retrieval.get("scoreMode") == "symbolic entity hits", "cypher: deveria usar score simbolico")
            require(grounding.get("queryEntities"), "cypher: deveria expor entidades da query")
            require((graph_trace.get("relationshipTypes") or {}).get("MENTIONS", 0) > 0, "cypher: grafo deveria vir da query MENTIONS")
            require(not graph_trace.get("paths"), "cypher: nao deveria injetar shortest paths globais no trace")
            require(not graph_trace.get("connectors"), "cypher: nao deveria tratar documentos como conectores 2-hop")
            require(all(doc.get("vectorScore") is None for doc in top_docs), "cypher: ranking principal nao deveria ser vetorial")

    require(len(set(methods.values())) >= 5, "estrategias estao colapsando em poucos metodos de retrieval")

    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print("GraphRAG strategy invariants: OK")
    for strategy in GRAPH_RAG_STRATEGY_ORDER:
        item = summary[strategy]
        degraded = f", fallback={item['fallbackStrategy']}" if item["degraded"] else ""
        print(
            f"- {strategy}: {item['method']} | {item['scoreMode']} | "
            f"docs={item['documents']} graph={item['nodes']}/{item['edges']}{degraded}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida invariantes das variantes GraphRAG.")
    parser.add_argument("question", nargs="?", default="Como Frodo se conecta a Sauron?")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        return validate(args.question, hops=args.hops, top_k=args.top_k, as_json=args.json)
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
