from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", value).strip()


class GraphRAG:
    def __init__(self, neo4j: Neo4jClient, ollama: OllamaClient) -> None:
        self.neo4j = neo4j
        self.ollama = ollama

    def answer(
        self,
        question: str,
        hops: int = 2,
        mode: str = "graph",
        model: str | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        entities = self.resolve_entities(question)
        graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180) if mode == "graph" else {}
        context = self.build_context(question, entities, graph, hops=hops, mode=mode)

        if not use_llm:
            answer = self.extractive_answer(question, entities, graph, llm_error=None)
            return {
                "question": question,
                "entities": entities,
                "hops": hops,
                "mode": mode,
                "context": context,
                "answer": answer,
                "graph": graph,
                "model": model or self.ollama.model,
                "llmStatus": "disabled",
            }

        try:
            answer = self.ollama.chat(self.messages(question, context), model=model)
            if not answer.strip():
                answer = self.extractive_answer(question, entities, graph, llm_error="resposta vazia do Ollama")
                status = "fallback: resposta vazia do Ollama"
            else:
                status = "ok"
        except Exception as exc:
            answer = self.extractive_answer(question, entities, graph, llm_error=str(exc))
            status = f"fallback: {exc}"

        return {
            "question": question,
            "entities": entities,
            "hops": hops,
            "mode": mode,
            "context": context,
            "answer": answer,
            "graph": graph,
            "model": model or self.ollama.model,
            "llmStatus": status,
        }

    def resolve_entities(self, question: str, limit: int = 4) -> list[str]:
        question_norm = f" {normalize(question)} "
        candidates = []
        for entity in self.neo4j.list_entities():
            names = [entity["name"], *(entity.get("aliases") or [])]
            best_match = ""
            for name in names:
                name_norm = normalize(name)
                if len(name_norm) < 3:
                    continue
                if re.search(rf"(?<!\w){re.escape(name_norm)}(?!\w)", question_norm):
                    if len(name_norm) > len(best_match):
                        best_match = name_norm
            if best_match:
                candidates.append(
                    {
                        "name": entity["name"],
                        "score": len(best_match) + float(entity.get("pagerank") or 0) * 100,
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        seen: set[str] = set()
        result: list[str] = []
        for candidate in candidates:
            if candidate["name"] not in seen:
                seen.add(candidate["name"])
                result.append(candidate["name"])
            if len(result) >= limit:
                break
        return result

    def build_context(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        hops: int,
        mode: str,
    ) -> str:
        lines: list[str] = []
        lines.append(f"Pergunta: {question}")
        lines.append(f"Modo de retrieval: {mode}")
        lines.append(f"Profundidade k-hop: {hops}")

        if entities:
            lines.append("Entidades detectadas: " + ", ".join(entities))
        else:
            fallback = self.neo4j.top_characters(limit=8)
            entities = [row["name"] for row in fallback]
            lines.append("Entidades detectadas: nenhuma; usando personagens centrais como contexto global.")

        lines.append("")
        lines.append("Perfis dos personagens centrais:")
        for name in entities[:4]:
            neighbors = self.neo4j.top_neighbors(name, limit=8)
            if neighbors:
                neighbor_text = "; ".join(
                    f"{row['name']} ({row['relation']}, peso={row['weight']})"
                    for row in neighbors
                )
                lines.append(f"- {name}: vizinhos principais: {neighbor_text}")

        if len(entities) >= 2:
            lines.append("")
            lines.append("Caminhos curtos entre entidades detectadas:")
            for idx, source in enumerate(entities):
                for target in entities[idx + 1 :]:
                    path = self.neo4j.shortest_path(source, target, max_depth=5)
                    if path:
                        lines.append(f"- {source} -> {target}: " + " -> ".join(path))

        if graph.get("edges"):
            lines.append("")
            lines.append("Arestas recuperadas do subgrafo:")
            for edge in sorted(
                graph["edges"],
                key=lambda item: (item["type"] != "INTERACTS_WITH", -(item.get("weight") or 1)),
            )[:45]:
                weight = edge.get("weight") or 1
                dataset = edge.get("sourceDataset") or "graph"
                lines.append(
                    f"- {edge['sourceName']} -[{edge['type']}, peso={weight}, fonte={dataset}]-> {edge['targetName']}"
                )

        if not graph.get("edges"):
            lines.append("")
            lines.append("Nenhum subgrafo especifico foi recuperado. Responda apenas com a incerteza apropriada.")

        return "\n".join(lines)

    @staticmethod
    def messages(question: str, context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Voce e um assistente para uma aula de doutorado sobre GNN, Knowledge Graphs, LLMs e GraphRAG. "
                    "Nao exponha raciocinio interno, passos ocultos ou texto de thinking. "
                    "Responda em portugues brasileiro. Use apenas o contexto recuperado do grafo. "
                    "Explique a resposta de forma curta, cite as entidades e relacoes relevantes, e quando fizer sentido "
                    "conecte a ideia com vizinhanca k-hop/message passing."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\nContexto recuperado do grafo:\n{context}\n\nPergunta: {question}",
            },
        ]

    def extractive_answer(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        llm_error: str | None,
    ) -> str:
        prefix = ""
        if llm_error:
            prefix = f"O Ollama nao respondeu ({llm_error}). "
        if not entities:
            return (
                f"{prefix}Nao detectei uma entidade especifica na pergunta. Use o grafo global, PageRank e "
                "comunidades para escolher pontos de entrada e depois reduza para uma vizinhanca k-hop."
            )

        edges = graph.get("edges") or []
        edge_text = self.direct_edge_summary(entities, edges)
        connector_text = self.connector_summary(entities, edges)
        entity_text = ", ".join(entities)

        parts = [
            f"{prefix}Pelo subgrafo recuperado, a pergunta envolve {entity_text}.",
        ]
        if edge_text:
            parts.append(edge_text)
        if connector_text:
            parts.append(connector_text)
        parts.append(
            "Para conectar com GNN: `hops=1` mostra relacoes diretas; `hops=2` inclui intermediarios "
            "que funcionam como campo receptivo; `hops=3+` aumenta cobertura, mas tambem traz ruido."
        )
        return "\n\n".join(parts)

    @staticmethod
    def direct_edge_summary(entities: list[str], edges: list[dict[str, Any]]) -> str:
        if len(entities) < 2:
            return ""
        wanted = set(entities)
        direct = []
        for edge in edges:
            if {edge["sourceName"], edge["targetName"]} <= wanted:
                direct.append(edge)
        if not direct:
            return ""
        fragments = []
        for edge in sorted(direct, key=lambda item: (item["type"], -(item.get("weight") or 1)))[:6]:
            fragments.append(
                f"{edge['sourceName']} -[{edge['type']}, peso={edge.get('weight', 1)}]-> {edge['targetName']}"
            )
        return "Ha relacoes diretas importantes: " + "; ".join(fragments) + "."

    @staticmethod
    def connector_summary(entities: list[str], edges: list[dict[str, Any]]) -> str:
        if len(entities) < 2:
            return ""
        left, right = entities[0], entities[1]
        neighbors: dict[str, dict[str, float]] = {left: {}, right: {}}
        for edge in edges:
            source = edge["sourceName"]
            target = edge["targetName"]
            weight = float(edge.get("weight") or 1)
            for seed in [left, right]:
                if source == seed and target != seed:
                    neighbors[seed][target] = max(neighbors[seed].get(target, 0), weight)
                if target == seed and source != seed:
                    neighbors[seed][source] = max(neighbors[seed].get(source, 0), weight)
        common = set(neighbors[left]) & set(neighbors[right])
        if not common:
            return ""
        ranked = sorted(
            common,
            key=lambda name: -(neighbors[left].get(name, 0) + neighbors[right].get(name, 0)),
        )[:5]
        formatted = ", ".join(
            f"{name} (peso combinado={neighbors[left].get(name, 0) + neighbors[right].get(name, 0):.0f})"
            for name in ranked
        )
        return f"Em 2-hop, os conectores mais fortes entre {left} e {right} incluem: {formatted}."
