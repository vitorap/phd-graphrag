from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graphrag import GraphRAG
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara RAG vetorial, Graph e GraphRAG hibrido.")
    parser.add_argument("question", help="pergunta em linguagem natural")
    parser.add_argument("--hops", type=int, default=2, help="profundidade k-hop")
    parser.add_argument("--top-k", type=int, default=8, help="numero de evidencias textuais recuperadas")
    parser.add_argument("--llm", action="store_true", help="usa Ollama nas tres respostas")
    parser.add_argument("--json", action="store_true", help="imprime JSON completo")
    args = parser.parse_args()

    try:
        client = Neo4jClient()
        rag = GraphRAG(client, OllamaClient())
        result = rag.compare(args.question, hops=args.hops, top_k=args.top_k, use_llm=args.llm)
        client.close()
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("Pergunta")
    print("========")
    print(args.question)
    for mode, payload in result["results"].items():
        print(f"\n{mode.upper()}")
        print("=" * len(mode))
        print(payload["answer"])
        docs = payload.get("documents") or []
        if docs:
            print("\nEvidencias textuais:")
            for doc in docs[:3]:
                source = doc.get("sourceTitle") or doc.get("sourceType")
                chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
                speaker = f" / {doc['speaker']}" if doc.get("speaker") else ""
                print(f"- {source}{chapter}{speaker} (score={doc.get('score', 0):.2f})")
        graph = payload.get("graph") or {}
        if graph.get("edges"):
            print(f"\nSubgrafo: {len(graph.get('nodes', []))} nos, {len(graph.get('edges', []))} arestas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
