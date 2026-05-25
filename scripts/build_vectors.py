from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o indice vetorial dos RetrievalDocument.")
    parser.add_argument("--model", default=settings.ollama_embed_model, help="modelo Ollama de embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="tamanho do lote enviado ao Ollama")
    parser.add_argument("--limit", type=int, default=6000, help="maximo de documentos indexados")
    parser.add_argument("--json", action="store_true", help="imprime JSON com estatisticas do indice")
    args = parser.parse_args()

    client = Neo4jClient()
    try:
        docs = client.retrieval_documents(limit=args.limit)
    finally:
        client.close()

    store = VectorStore()
    try:
        stats = store.build(
            docs,
            OllamaClient(),
            model=args.model,
            batch_size=max(1, args.batch_size),
        )
    except Exception as exc:
        print(f"erro ao gerar embeddings: {exc}", file=sys.stderr)
        print(
            f"verifique se o Ollama esta rodando e se o modelo de embedding existe: ollama pull {args.model}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(
            "indice vetorial pronto: "
            f"{stats['documents']} documentos, {stats['dimensions']} dimensoes, modelo {stats.get('embeddingModel')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
