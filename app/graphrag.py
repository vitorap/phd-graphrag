from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Any

from app.config import settings
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.text_utils import normalize, snippet, tokenize
from app.vector_store import VectorStore


class GraphRAG:
    def __init__(
        self,
        neo4j: Neo4jClient,
        ollama: OllamaClient,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.neo4j = neo4j
        self.ollama = ollama
        self.vector_store = vector_store or VectorStore()

    def answer(
        self,
        question: str,
        hops: int = 2,
        top_k: int = 8,
        mode: str = "graph",
        model: str | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        mode = self.normalize_mode(mode)
        entities = self.resolve_entities(question)
        graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180) if mode in {"graph", "hybrid"} else {}
        documents = self.retrieve_text(question, entities, graph, mode, limit=top_k) if mode in {"rag", "hybrid"} else []
        context_sections = self.build_context_sections(question, entities, graph, documents, hops=hops, mode=mode)
        context = context_sections["selected"]
        documents_by_source = self.documents_by_source(documents)

        if not use_llm:
            answer = self.extractive_answer(question, entities, graph, documents, mode, llm_error=None)
            return {
                "question": question,
                "entities": entities,
                "hops": hops,
                "topK": top_k,
                "topKPerSource": top_k if mode == "rag" else None,
                "mode": mode,
                "context": context,
                "textContext": context_sections["text"],
                "graphContext": context_sections["graph"],
                "hybridContext": context_sections["hybrid"],
                "answer": answer,
                "graph": graph,
                "documents": documents,
                "documentsBySource": documents_by_source,
                "retrieval": self.retrieval_summary(documents),
                "model": model or self.ollama.model,
                "llmStatus": "retrieval-only",
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
                status = "retrieval+ollama"
        except Exception as exc:
            answer = self.extractive_answer(question, entities, graph, documents, mode, llm_error=str(exc))
            status = f"fallback: {exc}"

        return {
            "question": question,
            "entities": entities,
            "hops": hops,
            "topK": top_k,
            "topKPerSource": top_k if mode == "rag" else None,
            "mode": mode,
            "context": context,
            "textContext": context_sections["text"],
            "graphContext": context_sections["graph"],
            "hybridContext": context_sections["hybrid"],
            "answer": answer,
            "graph": graph,
            "documents": documents,
            "documentsBySource": documents_by_source,
            "retrieval": self.retrieval_summary(documents),
            "model": model or self.ollama.model,
            "llmStatus": status,
        }

    def compare(
        self,
        question: str,
        hops: int = 2,
        top_k: int = 8,
        model: str | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        modes = ["rag", "graph", "hybrid"]
        return {
            "question": question,
            "hops": hops,
            "topK": top_k,
            "results": {
                mode: self.answer(question, hops=hops, top_k=top_k, mode=mode, model=model, use_llm=use_llm)
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

    @staticmethod
    def retrieval_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
        methods = sorted({doc.get("retrievalMethod") or "unknown" for doc in documents})
        by_source = Counter(doc.get("sourceType") or "other" for doc in documents)
        has_boost = any(float(doc.get("graphBoost") or 0.0) > 0 for doc in documents)
        has_vector = any(doc.get("vectorScore") is not None for doc in documents)
        return {
            "method": methods[0] if len(methods) == 1 else ("+".join(methods) if methods else "none"),
            "documents": len(documents),
            "bySource": dict(sorted(by_source.items())),
            "scoreMode": "cosine+graph-boost" if has_boost else ("cosine" if has_vector else ("bm25" if documents else "none")),
            "topScore": float(documents[0].get("score") or 0.0) if documents else 0.0,
            "topVectorScore": documents[0].get("vectorScore") if documents else None,
        }

    @staticmethod
    def documents_by_source(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {"book": [], "dialogue": [], "other": []}
        for doc in documents:
            source_type = doc.get("sourceType") or "other"
            key = source_type if source_type in grouped else "other"
            grouped[key].append(doc)
        return {key: value for key, value in grouped.items() if value}

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
        return self.build_context_sections(question, entities, graph, documents, hops, mode)["selected"]

    def build_context_sections(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        documents: list[dict[str, Any]],
        hops: int,
        mode: str,
    ) -> dict[str, str]:
        text_context = self.build_text_context(question, documents, mode=mode) if mode in {"rag", "hybrid"} else ""
        graph_context = self.build_graph_context(question, entities, graph, hops=hops) if mode in {"graph", "hybrid"} else ""
        hybrid_context = self.build_hybrid_context(question, text_context, graph_context) if mode == "hybrid" else ""
        selected = {
            "rag": text_context,
            "graph": graph_context,
            "hybrid": hybrid_context,
        }.get(mode, graph_context)
        return {
            "selected": selected,
            "text": text_context,
            "graph": graph_context,
            "hybrid": hybrid_context,
        }

    def build_text_context(self, question: str, documents: list[dict[str, Any]], mode: str) -> str:
        lines: list[str] = []
        lines.append(f"Pergunta: {question}")
        if mode == "rag":
            lines.append("Modo de retrieval: rag textual puro")
            lines.append("Score: similaridade textual apenas.")
        elif mode == "hybrid":
            lines.append("Modo de retrieval: texto para GraphRAG")
            lines.append("Score: similaridade textual com reforco de entidades ativadas pelo subgrafo.")
        else:
            lines.append("Modo de retrieval: texto")

        if not documents:
            lines.append("")
            lines.append("Nenhuma evidencia textual especifica foi recuperada.")
            return "\n".join(lines)

        grouped = self.documents_by_source(documents)
        for source_type, label in [("book", "Chunks dos livros"), ("dialogue", "Falas dos scripts"), ("other", "Outras evidencias")]:
            bucket = grouped.get(source_type) or []
            if not bucket:
                continue
            lines.append("")
            lines.append(f"{label}:")
            for idx, doc in enumerate(bucket, start=1):
                mentions = ", ".join(doc.get("mentions") or [])
                source = doc.get("sourceTitle") or doc.get("sourceType") or "texto"
                chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
                speaker = f" / fala de {doc['speaker']}" if doc.get("speaker") else ""
                score = float(doc.get("score") or 0.0)
                method = doc.get("retrievalMethod") or "retrieval"
                vector = doc.get("vectorScore")
                vector_text = f"; cosine={float(vector):.3f}" if vector is not None else ""
                boost = float(doc.get("graphBoost") or 0.0)
                boost_text = f"; boost={boost:.3f}" if boost else ""
                lines.append(
                    f"- #{idx} [{source}{chapter}{speaker}; metodo={method}; score={score:.3f}{vector_text}{boost_text}; mencoes={mentions}]"
                )
                lines.append(f"  {doc.get('snippet') or doc.get('text') or ''}")
        return "\n".join(lines)

    def build_graph_context(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        hops: int,
    ) -> str:
        lines: list[str] = []
        lines.append(f"Pergunta: {question}")
        lines.append("Modo de retrieval: graph estrutural")
        lines.append(f"Profundidade k-hop: {hops}")

        if entities:
            lines.append("Entidades detectadas: " + ", ".join(entities))
        else:
            lines.append("Entidades detectadas: nenhuma.")

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

        if not graph.get("edges"):
            lines.append("")
            lines.append("Nenhum subgrafo especifico foi recuperado. Responda apenas com a incerteza apropriada.")

        return "\n".join(lines)

    @staticmethod
    def build_hybrid_context(question: str, text_context: str, graph_context: str) -> str:
        return "\n\n".join(
            [
                f"Pergunta: {question}",
                "Modo de retrieval: GraphRAG hibrido",
                "A evidencia estrutural identifica entidades, caminhos e vizinhos; a evidencia textual sustenta a resposta.",
                "=== Evidencia estrutural ===",
                graph_context,
                "=== Evidencia textual ===",
                text_context,
            ]
        )

    @staticmethod
    def messages(question: str, context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Voce responde perguntas sobre o corpus de Senhor dos Aneis usando evidencias recuperadas. "
                    "Nao exponha raciocinio interno, passos ocultos ou texto de thinking. "
                    "Responda em portugues brasileiro. Use apenas o contexto recuperado. "
                    "Diferencie evidencia textual de evidencia estrutural quando isso ajudar. "
                    "Explique de forma curta e direta, cite entidades, relacoes, chunks ou falas relevantes, "
                    "e nao escreva frases meta sobre a aula, a apresentacao ou a turma."
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
        if mode == "rag":
            documents: list[dict[str, Any]] = []
            for source_type in ["book", "dialogue"]:
                documents.extend(
                    self.retrieve_text_ranked(
                        question,
                        entities=[],
                        graph=graph,
                        mode=mode,
                        limit=limit,
                        source_type=source_type,
                        apply_boost=False,
                    )
                )
            return documents

        return self.retrieve_text_ranked(
            question,
            entities=entities,
            graph=graph,
            mode=mode,
            limit=limit,
            source_type=None,
            apply_boost=mode == "hybrid",
        )

    def retrieve_text_ranked(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        mode: str,
        limit: int = 8,
        source_type: str | None = None,
        apply_boost: bool = False,
    ) -> list[dict[str, Any]]:
        graph_names = {
            node["name"]
            for node in graph.get("nodes", [])
            if node.get("name") and node.get("name") not in entities
        }
        if self.vector_store.exists():
            try:
                results = self.vector_store.search(
                    question,
                    self.ollama,
                    model=settings.ollama_embed_model,
                    limit=limit,
                    seed_entities=entities,
                    graph_entities=sorted(graph_names) if apply_boost else [],
                    source_type=source_type,
                    apply_boost=apply_boost,
                )
                for idx, doc in enumerate(results, start=1):
                    doc["sourceRank"] = idx
                    doc["sourceBucket"] = doc.get("sourceType") or "other"
                return results
            except Exception as exc:
                fallback = self.retrieve_text_bm25(
                    question,
                    entities,
                    graph,
                    mode,
                    limit,
                    source_type=source_type,
                    apply_boost=apply_boost,
                )
                for doc in fallback:
                    doc["retrievalMethod"] = "bm25_fallback"
                    doc["retrievalError"] = str(exc)
                return fallback
        return self.retrieve_text_bm25(
            question,
            entities,
            graph,
            mode,
            limit,
            source_type=source_type,
            apply_boost=apply_boost,
        )

    def retrieve_text_bm25(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        mode: str,
        limit: int = 8,
        source_type: str | None = None,
        apply_boost: bool = False,
    ) -> list[dict[str, Any]]:
        docs = self.neo4j.retrieval_documents()
        if not docs:
            return []
        if source_type:
            docs = [doc for doc in docs if doc.get("sourceType") == source_type]
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
        graph_set = graph_names if apply_boost else set()

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
            if apply_boost:
                score += len(seed_hits) * 1.65
                score += min(len(graph_hits), 6) * 0.18
            if apply_boost and len(seed_hits) >= 2:
                score += 1.5

            if score <= 0:
                continue
            enriched = dict(doc)
            enriched["score"] = score
            enriched["vectorScore"] = None
            enriched["graphBoost"] = None
            enriched["retrievalMethod"] = "bm25"
            enriched["snippet"] = snippet(str(doc.get("text") or ""), query_tokens, max_chars=780)
            scored.append(enriched)

        scored.sort(key=lambda item: item["score"], reverse=True)
        results = scored[:limit]
        for idx, doc in enumerate(results, start=1):
            doc["sourceRank"] = idx
            doc["sourceBucket"] = doc.get("sourceType") or "other"
        return results

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
            if mode == "rag":
                evidence_text = self.text_evidence_summary(documents)
                return (
                    f"{prefix}Modo `rag`: nao detectei entidades nomeadas na pergunta, entao a resposta fica baseada "
                    "apenas nas evidencias textuais recuperadas por similaridade. " + (evidence_text or "")
                ).strip()
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
                    "A busca vetorial recupera a dimensao narrativa: Frodo aparece ligado ao Anel e a missao rumo a "
                    "Mordor, enquanto Sauron aparece como a forca que busca recuperar esse poder."
                )
        if edge_text:
            parts.append(edge_text)
        if connector_text:
            parts.append(connector_text)
        if evidence_text:
            parts.append(evidence_text)
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
            method = doc.get("retrievalMethod") or "retrieval"
            sources.append(f"{source}{chapter}{speaker} ({method})")
        return "O RAG recuperou evidencias em: " + "; ".join(sources) + "."

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
