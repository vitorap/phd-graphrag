from __future__ import annotations

import os
import sys

from neo4j import GraphDatabase


def main() -> int:
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "graphrag-lotr")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (n:Entity)
                WITH count(n) AS entities
                MATCH ()-[r]->()
                WITH entities, count(r) AS relationships
                MATCH (c:Character)
                WITH entities, relationships, count(c) AS characters
                MATCH (d:RetrievalDocument)
                WITH entities, relationships, characters, count(d) AS retrievalDocuments
                MATCH (t:TextChunk)
                WITH entities, relationships, characters, retrievalDocuments, count(t) AS textChunks
                MATCH (l:DialogueLine)
                WITH entities, relationships, characters, retrievalDocuments, textChunks, count(l) AS dialogueLines
                MATCH (c:Character)
                RETURN entities, relationships, characters, retrievalDocuments, textChunks, dialogueLines,
                       collect(c.name)[0..10] AS sample
                """
            ).single()
            rel_rows = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC, type
                """
            )
            top_rows = session.run(
                """
                MATCH (c:Character)
                RETURN c.name AS name, c.race AS race, c.pagerank AS pagerank,
                       c.weightedDegree AS weightedDegree, c.community AS community
                ORDER BY coalesce(c.pagerank, 0) DESC
                LIMIT 10
                """
            )

            print("Grafo LOTR")
            print("==========")
            print(f"Entidades: {rows['entities']}")
            print(f"Relacoes: {rows['relationships']}")
            print(f"Personagens: {rows['characters']}")
            print(f"Documentos RAG: {rows['retrievalDocuments']}")
            print(f"Text chunks: {rows['textChunks']}")
            print(f"Falas de script: {rows['dialogueLines']}")
            print(f"Amostra: {', '.join(rows['sample'])}")
            print("\nRelacoes por tipo:")
            for row in rel_rows:
                print(f"- {row['type']}: {row['count']}")
            print("\nTop PageRank:")
            for row in top_rows:
                pagerank = row["pagerank"] or 0
                degree = row["weightedDegree"] or 0
                race = row["race"] or "-"
                community = row["community"]
                print(f"- {row['name']}: pr={pagerank:.4f}, wdeg={degree:.0f}, race={race}, c={community}")
        driver.close()
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
