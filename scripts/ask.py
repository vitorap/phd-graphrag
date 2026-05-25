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
    parser = argparse.ArgumentParser(description="Pergunta ao GraphRAG via CLI.")
    parser.add_argument("question", help="pergunta em linguagem natural")
    parser.add_argument("--hops", type=int, default=2, help="profundidade k-hop")
    parser.add_argument("--mode", default="graph", choices=["graph", "baseline"], help="modo de retrieval")
    parser.add_argument("--no-llm", action="store_true", help="mostra so contexto recuperado")
    parser.add_argument("--json", action="store_true", help="imprime JSON completo")
    args = parser.parse_args()

    try:
        client = Neo4jClient()
        rag = GraphRAG(client, OllamaClient())
        result = rag.answer(args.question, hops=args.hops, mode=args.mode, use_llm=not args.no_llm)
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
    print("\nEntidades detectadas")
    print("====================")
    print(", ".join(result["entities"]) if result["entities"] else "nenhuma")
    print("\nContexto recuperado")
    print("===================")
    print(result["context"])
    print("\nResposta")
    print("========")
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
