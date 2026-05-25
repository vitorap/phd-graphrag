from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
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
        entity_matches = self.resolve_entity_matches(question)
        entities = [match["name"] for match in entity_matches]
        graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180) if mode in {"graph", "hybrid"} else {}
        documents = self.retrieve_text(question, entities, graph, mode, limit=top_k) if mode in {"rag", "hybrid"} else []
        context_sections = self.build_context_sections(question, entities, graph, documents, hops=hops, mode=mode)
        context = context_sections["selected"]
        documents_by_source = self.documents_by_source(documents)
        trace = self.build_trace(
            question=question,
            entity_matches=entity_matches,
            graph=graph,
            documents=documents,
            context_sections=context_sections,
            hops=hops,
            top_k=top_k,
            mode=mode,
        )

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
                "trace": trace,
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
            "trace": trace,
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

    def build_trace(
        self,
        question: str,
        entity_matches: list[dict[str, Any]],
        graph: dict[str, Any],
        documents: list[dict[str, Any]],
        context_sections: dict[str, str],
        hops: int,
        top_k: int,
        mode: str,
    ) -> dict[str, Any]:
        entities = [match["name"] for match in entity_matches]
        edges = graph.get("edges") or []
        nodes = graph.get("nodes") or []
        relationship_counts = Counter(edge.get("type") or "UNKNOWN" for edge in edges)
        boosted_documents = [doc for doc in documents if float(doc.get("graphBoost") or 0.0) > 0]
        direct_edges = self.direct_edges(entities, edges)
        connectors = self.connector_rows(entities, edges)
        paths = self.shortest_paths(entities) if mode in {"graph", "hybrid"} else []
        context = context_sections.get("selected") or ""

        return {
            "variant": {
                "name": "Hybrid Vector + Graph",
                "subtitle": "entity grounding -> k-hop subgraph -> vector retrieval with graph boost -> LLM synthesis",
                "active": mode == "hybrid",
                "implemented": [
                    "Graph as structural context",
                    "Hybrid vector + graph reranking",
                ],
                "notImplemented": [
                    "LLM-generated Cypher",
                    "LLM-as-KG-builder during question answering",
                ],
            },
            "strategy": {
                "mode": mode,
                "graph": "deterministic_k_hop" if mode in {"graph", "hybrid"} else "disabled",
                "text": "vector_similarity_plus_graph_boost" if mode == "hybrid" else ("pure_vector_similarity" if mode == "rag" else "disabled"),
                "synthesis": "ollama_or_retrieval_only",
                "scoreFormula": "final = cosine + 0.08*seed_mentions + 0.015*subgraph_mentions_capped_8 + 0.08_if_two_seed_mentions",
            },
            "grounding": {
                "question": question,
                "entities": entity_matches,
                "entityCount": len(entity_matches),
            },
            "graph": {
                "hops": hops,
                "nodeCount": len(nodes),
                "edgeCount": len(edges),
                "relationshipTypes": dict(sorted(relationship_counts.items())),
                "paths": paths,
                "directEdges": direct_edges,
                "connectors": connectors,
                "query": self.graph_query_trace(entities, hops),
            },
            "retrieval": {
                **self.retrieval_summary(documents),
                "requestedTopK": top_k,
                "boostedDocuments": len(boosted_documents),
                "topDocuments": [self.trace_document(doc, entities, nodes) for doc in documents[: min(len(documents), 10)]],
            },
            "prompt": {
                "selectedChars": len(context),
                "estimatedTokens": max(1, len(context) // 4) if context else 0,
                "sections": [
                    {
                        "name": "Evidencia estrutural",
                        "chars": len(context_sections.get("graph") or ""),
                        "enabled": bool(context_sections.get("graph")),
                    },
                    {
                        "name": "Evidencia textual",
                        "chars": len(context_sections.get("text") or ""),
                        "enabled": bool(context_sections.get("text")),
                    },
                    {
                        "name": "Prompt hibrido",
                        "chars": len(context_sections.get("hybrid") or ""),
                        "enabled": bool(context_sections.get("hybrid")),
                    },
                ],
                "preview": context[:1800],
            },
            "steps": [
                {"id": "grounding", "label": "Grounding", "value": f"{len(entity_matches)} entidades"},
                {"id": "graph", "label": "Subgrafo k-hop", "value": f"{len(nodes)} nos / {len(edges)} arestas"},
                {"id": "retrieval", "label": "Reranking", "value": f"{len(documents)} docs / {len(boosted_documents)} boosted"},
                {"id": "prompt", "label": "Prompt", "value": f"~{max(1, len(context) // 4) if context else 0} tokens"},
            ],
        }

    @staticmethod
    def trace_document(doc: dict[str, Any], seed_entities: list[str], graph_nodes: list[dict[str, Any]]) -> dict[str, Any]:
        mentions = set(doc.get("mentions") or [])
        graph_names = {node.get("name") for node in graph_nodes if node.get("name")}
        seed_hits = sorted(mentions & set(seed_entities))
        graph_hits = sorted(mentions & graph_names)
        boost = float(doc.get("graphBoost") or 0.0)
        if boost > 0 and seed_hits:
            reason = "mentions seed entity"
        elif boost > 0 and graph_hits:
            reason = "mentions entity from k-hop subgraph"
        elif boost > 0:
            reason = "graph boost applied"
        else:
            reason = "ranked by vector similarity"
        return {
            "id": doc.get("id"),
            "sourceType": doc.get("sourceType") or "other",
            "sourceTitle": doc.get("sourceTitle") or doc.get("sourceType") or "texto",
            "chapterTitle": doc.get("chapterTitle"),
            "speaker": doc.get("speaker"),
            "sourceRank": doc.get("sourceRank"),
            "retrievalMethod": doc.get("retrievalMethod"),
            "vectorScore": doc.get("vectorScore"),
            "graphBoost": boost,
            "score": doc.get("score"),
            "mentions": sorted(mentions),
            "seedHits": seed_hits,
            "graphHits": graph_hits[:8],
            "boostReason": reason,
            "snippet": doc.get("snippet") or doc.get("text") or "",
        }

    @staticmethod
    def graph_query_trace(entities: list[str], hops: int) -> dict[str, Any]:
        if not entities:
            return {
                "label": "global fallback",
                "parameters": {"seeds": [], "hops": hops},
                "cypher": "MATCH (e:Entity) RETURN e ORDER BY coalesce(e.pagerank, 0) DESC LIMIT $limit",
            }
        return {
            "label": "deterministic k-hop expansion",
            "parameters": {"seeds": entities, "hops": hops, "limit": 180},
            "cypher": (
                "MATCH (seed:Entity)\n"
                "WHERE seed.name IN $seeds\n"
                f"MATCH p = (seed)-[*1..{max(1, min(int(hops), 4))}]-(n:Entity)\n"
                "WHERE all(rel IN relationships(p)\n"
                "  WHERE type(rel) <> 'PREDICTED_LINK' OR coalesce(rel.confidence, 0) >= 0.25)\n"
                "RETURN nodes(p) AS nodes, relationships(p) AS rels\n"
                "LIMIT $limit"
            ),
        }

    def resolve_entity_matches(self, question: str, limit: int = 4) -> list[dict[str, Any]]:
        question_norm = f" {normalize(question)} "
        candidates = []
        for entity in self.neo4j.list_entities():
            names = [entity["name"], *(entity.get("aliases") or [])]
            best_match = ""
            best_alias = ""
            for name in names:
                name_norm = normalize(name)
                if len(name_norm) < 3:
                    continue
                if re.search(rf"(?<!\w){re.escape(name_norm)}(?!\w)", question_norm):
                    if len(name_norm) > len(best_match):
                        best_match = name_norm
                        best_alias = name
            if best_match:
                candidates.append(
                    {
                        "name": entity["name"],
                        "matchedAlias": best_alias or entity["name"],
                        "aliases": sorted(set(entity.get("aliases") or []))[:8],
                        "labels": entity.get("labels") or [],
                        "kind": entity.get("kind"),
                        "pagerank": entity.get("pagerank"),
                        "score": len(best_match) + float(entity.get("pagerank") or 0) * 100,
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["name"] not in seen:
                seen.add(candidate["name"])
                result.append(candidate)
            if len(result) >= limit:
                break
        return result

    def resolve_entities(self, question: str, limit: int = 4) -> list[str]:
        return [match["name"] for match in self.resolve_entity_matches(question, limit=limit)]

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
                    "Voce e um sintetizador fiel para um sistema RAG/GraphRAG sobre Senhor dos Aneis. "
                    "Nao exponha raciocinio interno, passos ocultos ou texto de thinking. "
                    "Responda em portugues brasileiro e use somente o contexto recuperado. "
                    "Nao use conhecimento externo e nao invente fatos, arestas, falas ou citacoes. "
                    "Se o contexto sustentar uma resposta, responda diretamente e cite as evidencias usadas. "
                    "Se o contexto for insuficiente, diga que os trechos ou arestas recuperados nesta execucao "
                    "nao sustentam a resposta; nao afirme que a relacao ou fato nao existe no corpus inteiro. "
                    "Quando a recuperacao textual trouxer trechos irrelevantes, diga isso explicitamente como "
                    "falha de retrieval para aquela pergunta. "
                    "Nao transforme ausencia de aresta direta em ausencia de relacao; diferencie relacao direta, "
                    "conexao por caminho, coocorrencia e evidencia textual quando isso for relevante. "
                    "So diga que falta evidencia quando o contexto recuperado for vazio ou insuficiente. "
                    "Evite respostas defensivas quando houver evidencias indiretas suficientes. "
                    "Nao escreva frases meta sobre a aula, a apresentacao ou a turma."
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

        edges = graph.get("edges") or []
        edge_text = self.direct_edge_summary(entities, edges)
        connector_text = self.connector_summary(entities, edges)
        evidence_text = self.text_evidence_summary(documents)
        path_text = self.shortest_path_summary(entities) if mode in {"graph", "hybrid"} else ""

        parts = [
            f"{prefix}Retrieval-only (`{mode}`): sem sintese com LLM. "
            "Abaixo estao somente evidencias recuperadas pelo sistema.",
        ]
        if entities:
            parts.append("Entidades detectadas: " + ", ".join(entities) + ".")
        else:
            parts.append("Entidades detectadas: nenhuma.")

        if mode in {"graph", "hybrid"}:
            nodes = graph.get("nodes") or []
            parts.append(f"Subgrafo recuperado: {len(nodes)} nos e {len(edges)} arestas.")
        if path_text:
            parts.append(path_text)
        if edge_text:
            parts.append(edge_text)
        if connector_text:
            parts.append(connector_text)
        if evidence_text:
            parts.append(evidence_text)
        return "\n\n".join(parts)

    @staticmethod
    def text_evidence_summary(documents: list[dict[str, Any]], limit: int = 4) -> str:
        if not documents:
            return ""
        lines = ["Evidencias textuais mais bem ranqueadas:"]
        for idx, doc in enumerate(documents[:limit], start=1):
            label = "Livro" if doc.get("sourceType") == "book" else ("Script" if doc.get("sourceType") == "dialogue" else "Texto")
            source = doc.get("sourceTitle") or doc.get("sourceType") or "texto"
            chapter = f" / {doc['chapterTitle']}" if doc.get("chapterTitle") else ""
            speaker = f" / fala de {doc['speaker']}" if doc.get("speaker") else ""
            method = doc.get("retrievalMethod") or "retrieval"
            score = float(doc.get("score") or 0.0)
            text = " ".join(str(doc.get("snippet") or doc.get("text") or "").split())
            if len(text) > 280:
                text = text[:277].rstrip() + "..."
            lines.append(f"{idx}. {label} | {source}{chapter}{speaker} | {method} | score={score:.3f}: {text}")
        return "\n".join(lines)

    def shortest_path_summary(self, entities: list[str]) -> str:
        summaries = [
            f"{item['source']} -> {item['target']}: " + " -> ".join(item["path"])
            for item in self.shortest_paths(entities)
        ]
        if not summaries:
            return ""
        return "Caminhos mais curtos recuperados: " + "; ".join(summaries) + "."

    def shortest_paths(self, entities: list[str], limit: int = 3) -> list[dict[str, Any]]:
        if len(entities) < 2:
            return []
        paths = []
        for source, target in combinations(entities[:3], 2):
            path = self.neo4j.shortest_path(source, target, max_depth=5)
            if path:
                paths.append(
                    {
                        "source": source,
                        "target": target,
                        "path": path,
                        "length": max(0, len(path) - 1),
                    }
                )
            if len(paths) >= limit:
                break
        return paths

    @staticmethod
    def direct_edge_summary(entities: list[str], edges: list[dict[str, Any]]) -> str:
        direct = GraphRAG.direct_edges(entities, edges)
        if not direct:
            return ""
        fragments = []
        for edge in direct[:6]:
            fragments.append(
                f"{edge['sourceName']} -[{edge['type']}, peso={edge.get('weight') or edge.get('confidence') or 1}]-> {edge['targetName']}"
            )
        return "Relacoes diretas no subgrafo recuperado: " + "; ".join(fragments) + "."

    @staticmethod
    def direct_edges(entities: list[str], edges: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        if len(entities) < 2:
            return []
        wanted = set(entities)
        direct = [
            edge
            for edge in edges
            if {edge.get("sourceName"), edge.get("targetName")} <= wanted
        ]
        return sorted(direct, key=lambda item: (item.get("type") or "", -(item.get("weight") or item.get("confidence") or 1)))[:limit]

    @staticmethod
    def connector_summary(entities: list[str], edges: list[dict[str, Any]]) -> str:
        connectors = GraphRAG.connector_rows(entities, edges)
        if not connectors:
            return ""
        left, right = entities[0], entities[1]
        formatted = ", ".join(
            f"{row['name']} (peso combinado={row['combinedWeight']:.0f})"
            for row in connectors
        )
        return f"Conectores 2-hop no subgrafo recuperado entre {left} e {right}: {formatted}."

    @staticmethod
    def connector_rows(entities: list[str], edges: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
        if len(entities) < 2:
            return []
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
            return []
        ranked = sorted(
            common,
            key=lambda name: -(neighbors[left].get(name, 0) + neighbors[right].get(name, 0)),
        )[:limit]
        return [
            {
                "name": name,
                "leftWeight": neighbors[left].get(name, 0),
                "rightWeight": neighbors[right].get(name, 0),
                "combinedWeight": neighbors[left].get(name, 0) + neighbors[right].get(name, 0),
            }
            for name in ranked
        ]
