from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import log
from typing import Any

from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient


STOPWORDS = {
    "a",
    "about",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "for",
    "from",
    "is",
    "o",
    "os",
    "para",
    "por",
    "qual",
    "que",
    "the",
    "to",
    "what",
    "with",
}


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
        mode = self.normalize_mode(mode)
        entities = self.resolve_entities(question)
        graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180) if mode in {"graph", "hybrid"} else {}
        documents = self.retrieve_text(question, entities, graph, mode) if mode in {"rag", "hybrid"} else []
        context = self.build_context(question, entities, graph, documents, hops=hops, mode=mode)

        if not use_llm:
            answer = self.extractive_answer(question, entities, graph, documents, mode, llm_error=None)
            return {
                "question": question,
                "entities": entities,
                "hops": hops,
                "mode": mode,
                "context": context,
                "answer": answer,
                "graph": graph,
                "documents": documents,
                "model": model or self.ollama.model,
                "llmStatus": "disabled",
            }

        try:
            answer = self.ollama.chat(self.messages(question, context), model=model)
            if not answer.strip():
                answer = self.extractive_answer(
                    question,
                    entities,
                    graph,
                    documents,
                    mode,
                    llm_error="resposta vazia do Ollama",
                )
                status = "fallback: resposta vazia do Ollama"
            else:
                status = "ok"
        except Exception as exc:
            answer = self.extractive_answer(question, entities, graph, documents, mode, llm_error=str(exc))
            status = f"fallback: {exc}"

        return {
            "question": question,
            "entities": entities,
            "hops": hops,
            "mode": mode,
            "context": context,
            "answer": answer,
            "graph": graph,
            "documents": documents,
            "model": model or self.ollama.model,
            "llmStatus": status,
        }

    def compare(
        self,
        question: str,
        hops: int = 2,
        model: str | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        modes = ["rag", "graph", "hybrid"]
        return {
            "question": question,
            "hops": hops,
            "results": {
                mode: self.answer(question, hops=hops, mode=mode, model=model, use_llm=use_llm)
                for mode in modes
            },
        }

    @staticmethod
    def normalize_mode(mode: str) -> str:
        if mode == "baseline":
            return "rag"
        if mode not in {"rag", "graph", "hybrid"}:
            return "graph"
        return mode

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
        documents: list[dict[str, Any]],
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
                key=lambda item: (
                    item["type"] not in {"INTERACTS_WITH", "CO_OCCURS_WITH"},
                    -(item.get("weight") or item.get("confidence") or 1),
                ),
            )[:45]:
                weight = edge.get("weight") or edge.get("confidence") or 1
                dataset = edge.get("sourceDataset") or "graph"
                method = f", metodo={edge['method']}" if edge.get("method") else ""
                lines.append(
                    f"- {edge['sourceName']} -[{edge['type']}, peso={weight}{method}, fonte={dataset}]-> {edge['targetName']}"
                )

        if documents:
            lines.append("")
            lines.append("Evidencias textuais recuperadas:")
            for doc in documents[:8]:
                mentions = ", ".join(doc.get("mentions") or [])
                source = doc.get("sourceTitle") or doc.get("sourceType") or "texto"
                chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
                speaker = f" / fala de {doc['speaker']}" if doc.get("speaker") else ""
                score = float(doc.get("score") or 0.0)
                lines.append(f"- [{source}{chapter}{speaker}; score={score:.2f}; mencoes={mentions}]")
                lines.append(f"  {doc.get('snippet') or doc.get('text') or ''}")

        if not graph.get("edges") and mode in {"graph", "hybrid"}:
            lines.append("")
            lines.append("Nenhum subgrafo especifico foi recuperado. Responda apenas com a incerteza apropriada.")
        if not documents and mode in {"rag", "hybrid"}:
            lines.append("")
            lines.append("Nenhuma evidencia textual especifica foi recuperada.")

        return "\n".join(lines)

    @staticmethod
    def messages(question: str, context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Voce e um assistente para uma aula de doutorado sobre GNN, Knowledge Graphs, LLMs e GraphRAG. "
                    "Nao exponha raciocinio interno, passos ocultos ou texto de thinking. "
                    "Responda em portugues brasileiro. Use apenas o contexto recuperado. "
                    "Diferencie evidencia textual de evidencia estrutural quando isso ajudar. "
                    "Explique a resposta de forma curta, cite entidades, relacoes, chunks ou falas relevantes, "
                    "e conecte a ideia com vizinhanca k-hop/message passing quando fizer sentido."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\nContexto recuperado:\n{context}\n\nPergunta: {question}",
            },
        ]

    def retrieve_text(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        mode: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        docs = self.neo4j.retrieval_documents()
        if not docs:
            return []

        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        graph_names = {
            node["name"]
            for node in graph.get("nodes", [])
            if node.get("name") and node.get("name") not in entities
        }
        seed_set = set(entities)
        graph_set = graph_names if mode == "hybrid" else set()

        tokenized_docs = [tokenize(str(doc.get("text") or "")) for doc in docs]
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))
        avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
        total_docs = len(docs)

        scored: list[dict[str, Any]] = []
        for doc, tokens in zip(docs, tokenized_docs):
            if not tokens:
                continue
            token_counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                tf = token_counts.get(token, 0)
                if tf == 0:
                    continue
                df = doc_freq.get(token, 0)
                idf = log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(avg_len, 1))
                score += idf * (tf * 2.5) / denom

            mentions = set(doc.get("mentions") or [])
            seed_hits = mentions & seed_set
            graph_hits = mentions & graph_set
            score += len(seed_hits) * 1.65
            score += min(len(graph_hits), 6) * 0.18
            if doc.get("sourceType") == "book":
                score += 0.2
            if len(seed_hits) >= 2:
                score += 1.5

            if score <= 0:
                continue
            enriched = dict(doc)
            enriched["score"] = score
            enriched["snippet"] = snippet(str(doc.get("text") or ""), query_tokens, max_chars=780)
            scored.append(enriched)

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def extractive_answer(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        documents: list[dict[str, Any]],
        mode: str,
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
        evidence_text = self.text_evidence_summary(documents)
        entity_text = ", ".join(entities)

        parts = [
            f"{prefix}Modo `{mode}`: a pergunta envolve {entity_text}.",
        ]
        if {"Frodo", "Sauron"} <= set(entities) and documents:
            if mode == "hybrid":
                parts.append(
                    "A leitura hibrida fica mais forte aqui: Frodo aparece como o portador que precisa levar o Anel "
                    "ate Mordor, enquanto Sauron aparece como o poder antagonista ligado ao Um Anel. O grafo mostra "
                    "a conexao estrutural; o texto explica a natureza narrativa dessa conexao."
                )
            else:
                parts.append(
                    "O RAG textual recupera a dimensao narrativa: Frodo aparece ligado ao Anel e a missao rumo a "
                    "Mordor, enquanto Sauron aparece como a forca que busca recuperar esse poder."
                )
        if edge_text:
            parts.append(edge_text)
        if connector_text:
            parts.append(connector_text)
        if evidence_text:
            parts.append(evidence_text)
        parts.append(
            "Para conectar com GNN: `hops=1` mostra relacoes diretas; `hops=2` inclui intermediarios "
            "que funcionam como campo receptivo; `hops=3+` aumenta cobertura, mas tambem traz ruido."
        )
        return "\n\n".join(parts)

    @staticmethod
    def text_evidence_summary(documents: list[dict[str, Any]]) -> str:
        if not documents:
            return ""
        sources = []
        for doc in documents[:4]:
            source = doc.get("sourceTitle") or doc.get("sourceType") or "texto"
            chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
            speaker = f" / fala de {doc['speaker']}" if doc.get("speaker") else ""
            sources.append(f"{source}{chapter}{speaker}")
        return "O RAG textual recuperou evidencias em: " + "; ".join(sources) + "."

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
                f"{edge['sourceName']} -[{edge['type']}, peso={edge.get('weight') or edge.get('confidence') or 1}]-> {edge['targetName']}"
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


def tokenize(value: str) -> list[str]:
    normalized = normalize(value)
    return [token for token in normalized.split() if len(token) > 2 and token not in STOPWORDS]


def snippet(text: str, query_tokens: list[str], max_chars: int = 760) -> str:
    if len(text) <= max_chars:
        return text
    lower = normalize(text)
    first_hit = -1
    for token in query_tokens:
        first_hit = lower.find(token)
        if first_hit >= 0:
            break
    if first_hit < 0:
        return text[:max_chars].strip() + "..."
    start = max(0, first_hit - max_chars // 3)
    end = min(len(text), start + max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
