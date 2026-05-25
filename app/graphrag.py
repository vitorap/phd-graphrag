from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from math import log
from typing import Any

from app.config import settings
from app.llm_service import LLMService
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.text_utils import normalize, snippet, tokenize
from app.vector_store import VectorStore


GRAPH_RAG_STRATEGY_ORDER = [
    "kg_index",
    "vector_first",
    "graph_filter",
    "path",
    "community",
    "cypher",
]

GRAPH_RAG_STRATEGIES: dict[str, dict[str, Any]] = {
    "kg_index": {
        "id": "kg_index",
        "name": "KG-as-Index / Graph Boost",
        "shortName": "KG-as-Index",
        "subtitle": "entity grounding -> k-hop subgraph -> vector retrieval with graph boost",
        "description": "O grafo funciona como indice estrutural: entidades da pergunta abrem um subgrafo e os chunks vetoriais que mencionam esses nos ganham reforco no ranking.",
        "graph": "deterministic_k_hop",
        "text": "vector_similarity_plus_graph_boost",
        "synthesis": "structural_and_textual_evidence",
        "scoreFormula": "final = cosine + 0.08*seed_mentions + 0.015*subgraph_mentions_capped_8 + 0.08_if_two_seed_mentions",
        "flow": ["pergunta", "entidades", "subgrafo k-hop", "boost vetorial", "sintese"],
        "stageDetails": [
            {"label": "Grounding", "detail": "Aliases resolvem entidades sementes na pergunta."},
            {"label": "Subgrafo", "detail": "Neo4j expande vizinhanca k-hop dessas entidades."},
            {"label": "Boost", "detail": "Chunks que mencionam sementes/vizinhos sobem no ranking."},
            {"label": "Sintese", "detail": "LLM recebe evidencia estrutural e textual."},
        ],
        "bestFor": "Perguntas relacionais que precisam de texto narrativo e de uma trilha estrutural auditavel.",
        "graphRole": "Indice e reranker: o grafo nao substitui o embedding, ele altera a ordem das evidencias.",
        "textRole": "Embeddings continuam trazendo os chunks; o boost so favorece chunks conectados ao subgrafo.",
        "lectureCue": "Baseline GraphRAG solido para perguntas relacionais como Frodo-Sauron.",
        "visualHint": "Procure docs com cosine alto e boost positivo; eles citam sementes ou vizinhos do subgrafo.",
        "risk": "Se o entity grounding errar, o boost amplifica o erro.",
        "references": [
            {"label": "Microsoft GraphRAG Local Search", "url": "https://microsoft.github.io/graphrag/query/local_search/"},
            {"label": "KG2RAG", "url": "https://arxiv.org/abs/2502.06864"},
        ],
    },
    "vector_first": {
        "id": "vector_first",
        "name": "Vector-first / Graph Expansion",
        "shortName": "Vector-first",
        "subtitle": "vector retrieval -> mentions from hits -> graph expansion -> synthesis",
        "description": "Comeca como RAG comum. Depois usa as entidades mencionadas nos chunks recuperados para abrir o grafo e explicar conexoes que o texto sozinho nao mostra.",
        "graph": "expanded_from_vector_hits",
        "text": "pure_vector_similarity",
        "synthesis": "text_then_graph_context",
        "scoreFormula": "final = cosine; graph is added after retrieval from retrieved mentions",
        "flow": ["embedding", "top-k textual", "entidades dos chunks", "subgrafo", "sintese"],
        "stageDetails": [
            {"label": "Busca inicial", "detail": "A pergunta vai primeiro para o indice vetorial puro."},
            {"label": "Mencoes", "detail": "Entidades citadas nos chunks viram novas sementes."},
            {"label": "Expansao", "detail": "O grafo entra depois para explicar/expandir o que o texto achou."},
            {"label": "Sintese", "detail": "O contexto textual lidera; o grafo ajuda na explicabilidade."},
        ],
        "bestFor": "Perguntas em que o texto deve liderar e o grafo serve para contextualizar hits encontrados.",
        "graphRole": "Explicador posterior: o grafo e derivado dos chunks recuperados, nao filtro inicial.",
        "textRole": "Cosine puro decide o top-k; a estrutura aparece depois como leitura complementar.",
        "lectureCue": "RAG textual primeiro; o grafo entra depois para explicar conexoes encontradas.",
        "visualHint": "O score fica cosine puro; o subgrafo deve refletir entidades mencionadas pelos chunks.",
        "risk": "Se o top-k inicial nao trouxer boas entidades, o grafo expande contexto errado.",
        "references": [
            {"label": "RAG original", "url": "https://arxiv.org/abs/2005.11401"},
            {"label": "LightRAG", "url": "https://arxiv.org/abs/2410.05779"},
        ],
    },
    "graph_filter": {
        "id": "graph_filter",
        "name": "Graph-Constrained Retrieval",
        "shortName": "Graph filter",
        "subtitle": "entity grounding -> k-hop subgraph -> strict document filter -> rerank",
        "description": "O grafo atua como filtro duro: so entram evidencias textuais que mencionam entidades do subgrafo ativado pela pergunta.",
        "graph": "deterministic_k_hop",
        "text": "vector_similarity_filtered_by_graph_mentions",
        "synthesis": "filtered_text_with_graph_context",
        "scoreFormula": "candidate docs must mention a seed or subgraph node; final score keeps cosine+boost",
        "flow": ["entidades", "subgrafo", "docs ligados", "rerank", "sintese"],
        "stageDetails": [
            {"label": "Grounding", "detail": "Entidades da pergunta ativam o subgrafo."},
            {"label": "Filtro", "detail": "So entram documentos com MENTIONS para sementes/vizinhos."},
            {"label": "Rerank", "detail": "A busca vetorial reordena somente candidatos estruturalmente ligados."},
            {"label": "Sintese", "detail": "A resposta fica mais auditavel, mas menos aberta."},
        ],
        "bestFor": "Perguntas em que precisao e auditabilidade importam mais do que cobertura ampla.",
        "graphRole": "Filtro duro: define o conjunto candidato antes da resposta.",
        "textRole": "Texto ainda explica, mas precisa estar ligado por MENTIONS ao subgrafo ativado.",
        "lectureCue": "Precision vs recall ficam visiveis: filtro forte reduz ruido e pode perder evidencias.",
        "visualHint": "Todos os docs devem ter seed hit, graph hit ou foco em entidades do subgrafo.",
        "risk": "Mais preciso, mas pode perder trechos bons que nao foram ligados por MENTIONS.",
        "references": [
            {"label": "GRAG", "url": "https://arxiv.org/abs/2405.16506"},
            {"label": "KG2RAG", "url": "https://arxiv.org/abs/2502.06864"},
        ],
    },
    "path": {
        "id": "path",
        "name": "Path / Connector Retrieval",
        "shortName": "Paths",
        "subtitle": "entity pairs -> shortest paths/connectors -> path-aware evidence",
        "description": "Foca em perguntas relacionais: caminhos curtos e conectores 2-hop definem quais entidades devem aparecer nas evidencias textuais.",
        "graph": "shortest_paths_and_connectors",
        "text": "vector_similarity_plus_path_entity_boost",
        "synthesis": "path_explanation_with_text",
        "scoreFormula": "requires >=2 grounded entities; final = cosine+seed_boost+path_focus_boost+0.06_if_seed_pair_seen",
        "flow": ["pares de entidades", "shortest path", "conectores", "chunks com ponte", "sintese"],
        "stageDetails": [
            {"label": "Pares", "detail": "Entidades detectadas formam pares relacionais."},
            {"label": "Caminhos", "detail": "Shortest paths e conectores 2-hop viram foco da busca."},
            {"label": "Rerank", "detail": "Chunks que mencionam entidades do caminho ganham peso."},
            {"label": "Explicacao", "detail": "A resposta pode narrar a ponte estrutural."},
        ],
        "bestFor": "Perguntas do tipo 'como A se conecta a B?' ou 'quem faz a ponte?'.",
        "graphRole": "Raciocinio por caminho: o grafo escolhe intermediarios e conectores.",
        "textRole": "Texto sustenta por que a ponte importa narrativamente.",
        "lectureCue": "A variante mais proxima da analogia com message passing k-hop e caminhos explicaveis.",
        "visualHint": "Veja a linha Caminhos/Conectores no trace e docs com foco em entidades intermediarias.",
        "risk": "Caminho curto pode ser topologicamente valido e narrativamente fraco.",
        "references": [
            {"label": "HippoRAG", "url": "https://arxiv.org/abs/2405.14831"},
            {"label": "GNN-RAG", "url": "https://arxiv.org/abs/2405.20139"},
        ],
    },
    "community": {
        "id": "community",
        "name": "Community / Local-to-Global",
        "shortName": "Community",
        "subtitle": "entity grounding -> graph community -> representative evidence",
        "description": "Usa a comunidade estrutural dos personagens como contexto agregado, aproximando a ideia local-to-global sem precomputar summaries com LLM.",
        "graph": "seed_community_subgraph",
        "text": "vector_similarity_plus_community_entity_boost",
        "synthesis": "community_context_plus_text",
        "scoreFormula": "final = cosine + boost/rerank for mentions of seed/community entities",
        "flow": ["entidades", "comunidade", "nos centrais", "evidencias representativas", "sintese"],
        "stageDetails": [
            {"label": "Comunidade", "detail": "Sementes localizam uma comunidade estrutural no grafo."},
            {"label": "Centrais", "detail": "Nos centrais da comunidade viram contexto agregado."},
            {"label": "Evidencias", "detail": "Chunks que mencionam a comunidade recebem reforco."},
            {"label": "Resumo", "detail": "A resposta combina contexto local e visao mais global."},
        ],
        "bestFor": "Perguntas mais amplas sobre grupos, alianças, nucleos narrativos ou contexto ao redor de personagens.",
        "graphRole": "Agregador local-to-global: troca uma vizinhanca curta por uma comunidade estrutural.",
        "textRole": "Texto selecionado deve representar a comunidade, nao apenas a entidade exata.",
        "lectureCue": "Aproxima a ideia Microsoft GraphRAG global/local sem depender de summaries precomputados.",
        "visualHint": "O subgrafo costuma ter menos ruido local e mais personagens da mesma comunidade.",
        "risk": "Comunidades resumem contexto, mas podem diluir relacoes especificas.",
        "references": [
            {"label": "From Local to Global GraphRAG", "url": "https://arxiv.org/abs/2404.16130"},
            {"label": "Microsoft Global Search", "url": "https://microsoft.github.io/graphrag/examples_notebooks/global_search/"},
        ],
    },
    "cypher": {
        "id": "cypher",
        "name": "Symbolic Cypher / MENTIONS Query",
        "shortName": "Cypher",
        "subtitle": "entities -> auditable Cypher template -> graph rows/docs -> synthesis",
        "description": "Mostra a familia query-driven: entidades da pergunta alimentam uma consulta Cypher deterministica e auditavel. A aba Graph separada demonstra a geracao de Cypher por LLM.",
        "graph": "symbolic_query_template",
        "text": "documents_queried_by_mentions",
        "synthesis": "query_result_with_text",
        "scoreFormula": "score = entityHits from Cypher + small rank prior; no vector cosine",
        "flow": ["pergunta", "entidades", "Cypher read-only", "linhas/docs", "sintese"],
        "stageDetails": [
            {"label": "Entidades", "detail": "A pergunta resolve nomes usados como parametros da query."},
            {"label": "Cypher", "detail": "Uma consulta simbolica busca documentos via MENTIONS."},
            {"label": "Linhas", "detail": "Score vem de hits de entidades, nao de cosine."},
            {"label": "Sintese", "detail": "A query vira objeto auditavel da explicacao."},
        ],
        "bestFor": "Perguntas que podem virar consulta clara: entidades, relacoes, contagens, vizinhos ou documentos ligados.",
        "graphRole": "Plano simbolico: a query explicita exatamente o que foi buscado.",
        "textRole": "Docs entram por MENTIONS e entityHits; embedding nao decide o ranking principal.",
        "lectureCue": "Conecta a aba Graph com a familia query-driven de GraphRAG.",
        "visualHint": "O trace deve marcar score como symbolic entity hits e mostrar a Cypher equivalente.",
        "risk": "A qualidade depende do entity grounding e do template escolhido; nao e geracao livre de Cypher.",
        "references": [
            {"label": "Microsoft GraphRAG Local Search", "url": "https://microsoft.github.io/graphrag/query/local_search/"},
            {"label": "Neo4j Cypher Manual", "url": "https://neo4j.com/docs/cypher-manual/current/"},
        ],
    },
}


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
        self.llm_service = LLMService(base_url=ollama.base_url, model=ollama.model)

    def answer(
        self,
        question: str,
        hops: int = 2,
        top_k: int = 8,
        mode: str = "graph",
        model: str | None = None,
        use_llm: bool = True,
        graph_rag_strategy: str = "kg_index",
    ) -> dict[str, Any]:
        mode = self.normalize_mode(mode)
        graph_rag_strategy = self.normalize_graph_rag_strategy(graph_rag_strategy)
        entity_matches = self.resolve_entity_matches(question)
        entities = [match["name"] for match in entity_matches]
        strategy_runtime: dict[str, Any] = {
            "strategy": graph_rag_strategy if mode == "hybrid" else mode,
            "graphSeeds": entities,
            "derivedEntities": [],
            "pathEntities": [],
            "communityEntities": [],
            "queryEntities": [],
            "degraded": False,
            "fallbackStrategy": None,
            "notes": [],
        }
        if mode == "rag":
            graph = {}
            documents = self.retrieve_text(question, entities, graph, mode, limit=top_k)
        elif mode == "graph":
            graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180)
            documents = []
        else:
            graph, documents, strategy_runtime = self.retrieve_graphrag(
                question=question,
                entities=entities,
                hops=hops,
                top_k=top_k,
                strategy=graph_rag_strategy,
            )
        context_sections = self.build_context_sections(
            question,
            entities,
            graph,
            documents,
            hops=hops,
            mode=mode,
            strategy_runtime=strategy_runtime,
        )
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
            strategy_runtime=strategy_runtime,
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
                "graphRagStrategy": graph_rag_strategy if mode == "hybrid" else None,
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
                "structuredAnswer": None,
                "llmTrace": None,
                "model": model or self.ollama.model,
                "llmStatus": "retrieval-only",
            }

        structured_answer: dict[str, Any] | None = None
        llm_trace: dict[str, Any] | None = None
        try:
            llm_result = self.llm_service.synthesize_answer(
                question=question,
                mode=mode,
                context_sections=context_sections,
                strategy=GRAPH_RAG_STRATEGIES.get(graph_rag_strategy) if mode == "hybrid" else None,
                strategy_runtime=strategy_runtime,
                model=model,
            )
            answer = llm_result.answer
            structured_answer = llm_result.structured_answer.model_dump() if llm_result.structured_answer else None
            llm_trace = llm_result.trace.model_dump()
            trace = self.apply_llm_trace(trace, llm_trace)
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
                status = llm_result.status
        except Exception as exc:
            answer = self.extractive_answer(question, entities, graph, documents, mode, llm_error=str(exc))
            status = f"fallback: {exc}"
            llm_trace = {
                "provider": "ollama",
                "adapter": "langchain-ollama",
                "model": model or self.ollama.model,
                "template": f"answer:{mode}",
                "schemaName": "GroundedAnswer",
                "promptChars": len(context),
                "estimatedTokens": max(1, len(context) // 4) if context else 0,
                "attempts": 0,
                "status": "fallback",
                "error": str(exc),
                "preview": context[:1800],
            }
            trace = self.apply_llm_trace(trace, llm_trace)

        return {
            "question": question,
            "entities": entities,
            "hops": hops,
            "topK": top_k,
            "topKPerSource": top_k if mode == "rag" else None,
            "mode": mode,
            "graphRagStrategy": graph_rag_strategy if mode == "hybrid" else None,
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
            "structuredAnswer": structured_answer,
            "llmTrace": llm_trace,
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
        graph_rag_strategy: str = "kg_index",
    ) -> dict[str, Any]:
        modes = ["rag", "graph", "hybrid"]
        return {
            "question": question,
            "hops": hops,
            "topK": top_k,
            "graphRagStrategy": self.normalize_graph_rag_strategy(graph_rag_strategy),
            "results": {
                mode: self.answer(
                    question,
                    hops=hops,
                    top_k=top_k,
                    mode=mode,
                    model=model,
                    use_llm=use_llm,
                    graph_rag_strategy=graph_rag_strategy,
                )
                for mode in modes
            },
        }

    def compare_graphrag_strategies(
        self,
        question: str,
        hops: int = 2,
        top_k: int = 8,
        model: str | None = None,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "hops": hops,
            "topK": top_k,
            "llmStatus": "retrieval-only",
            "strategies": self.strategy_catalog(),
            "results": {
                strategy: self.answer(
                    question,
                    hops=hops,
                    top_k=top_k,
                    mode="hybrid",
                    model=model,
                    use_llm=False,
                    graph_rag_strategy=strategy,
                )
                for strategy in GRAPH_RAG_STRATEGY_ORDER
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
    def normalize_graph_rag_strategy(strategy: str | None) -> str:
        if not strategy:
            return "kg_index"
        clean = str(strategy).strip().lower().replace("-", "_")
        aliases = {
            "hybrid": "kg_index",
            "boost": "kg_index",
            "kg": "kg_index",
            "filter": "graph_filter",
            "paths": "path",
            "path_connector": "path",
            "local_global": "community",
            "global": "community",
            "text2cypher": "cypher",
            "text_to_cypher": "cypher",
            "query": "cypher",
        }
        clean = aliases.get(clean, clean)
        return clean if clean in GRAPH_RAG_STRATEGIES else "kg_index"

    @staticmethod
    def strategy_catalog() -> list[dict[str, Any]]:
        return [dict(GRAPH_RAG_STRATEGIES[key]) for key in GRAPH_RAG_STRATEGY_ORDER]

    def retrieve_graphrag(
        self,
        question: str,
        entities: list[str],
        hops: int,
        top_k: int,
        strategy: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        strategy = self.normalize_graph_rag_strategy(strategy)
        runtime: dict[str, Any] = {
            "strategy": strategy,
            "graphSeeds": entities,
            "questionEntities": entities,
            "derivedEntities": [],
            "pathEntities": [],
            "communityEntities": [],
            "queryEntities": [],
            "degraded": False,
            "fallbackStrategy": None,
            "notes": [],
        }

        if strategy == "vector_first":
            seed_docs = self.retrieve_text_ranked(
                question,
                entities=[],
                graph={},
                mode="rag",
                limit=max(top_k, 12),
                apply_boost=False,
            )
            derived_entities = self.ranked_mentions(seed_docs, limit=8)
            graph_seeds = self.unique_names(derived_entities)
            degraded = False
            notes = ["O grafo foi expandido depois dos primeiros hits vetoriais; o ranking textual fica cosine puro."]
            if not graph_seeds:
                graph_seeds = entities
                degraded = True
                notes.append("Nenhuma entidade apareceu nos hits vetoriais; fallback usa entidades da pergunta so para desenhar o grafo.")
            graph = self.neo4j.subgraph_for_seeds(graph_seeds, hops=hops, limit=180)
            runtime.update(
                {
                    "graphSeeds": graph_seeds,
                    "derivedEntities": derived_entities,
                    "degraded": degraded,
                    "fallbackStrategy": "question_entity_graph" if degraded else None,
                    "notes": notes,
                }
            )
            return graph, self.tag_documents(seed_docs[:top_k], "vector_first"), runtime

        if strategy == "community":
            graph = self.neo4j.community_subgraph_for_seeds(entities, limit=180)
            degraded = False
            notes = ["A comunidade estrutural substitui a expansao k-hop comum."]
            if not graph.get("nodes"):
                graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180)
                degraded = True
                notes = ["Nenhuma comunidade estrutural foi encontrada; fallback usa subgrafo k-hop."]
            community_entities = self.top_graph_node_names(graph, limit=28)
            runtime.update(
                {
                    "graphSeeds": entities,
                    "communityEntities": community_entities,
                    "degraded": degraded,
                    "fallbackStrategy": "kg_index" if degraded else None,
                    "notes": notes,
                }
            )
            docs = self.retrieve_text_ranked(
                question,
                entities=entities,
                graph=graph,
                mode="hybrid",
                limit=max(top_k * 6, 36),
                apply_boost=True,
                boost_entities=community_entities,
            )
            docs = self.rerank_by_mentions(docs, entities, community_entities, "community", seed_pair_bonus=False)
            return graph, self.tag_documents(docs[:top_k], "community"), runtime

        if strategy == "cypher":
            query_entities = self.unique_names(entities)
            if not query_entities:
                graph = self.neo4j.global_graph(limit=120)
                docs = self.retrieve_text_ranked(
                    question,
                    entities=[],
                    graph={},
                    mode="rag",
                    limit=top_k,
                    apply_boost=False,
                )
                runtime.update(
                    {
                        "graphSeeds": [],
                        "queryEntities": [],
                        "degraded": True,
                        "fallbackStrategy": "vector_first",
                        "notes": ["Sem entidade resolvida, nao ha parametros seguros para a query simbolica; fallback usa busca vetorial pura."],
                    }
                )
                return graph, self.tag_documents(docs[:top_k], "cypher_fallback_vector"), runtime

            graph = self.neo4j.mentions_graph_for_entities(query_entities, limit=max(top_k * 3, 18))
            runtime.update(
                {
                    "graphSeeds": query_entities,
                    "queryEntities": query_entities,
                    "notes": ["A consulta simbolica usa MENTIONS como ponte auditavel entre entidades e documentos."],
                }
            )
            docs = self.entity_documents(query_entities, question, method="cypher_mentions", limit=top_k)
            if not docs:
                docs = self.retrieve_text_ranked(
                    question,
                    entities=query_entities,
                    graph=graph,
                    mode="hybrid",
                    limit=top_k,
                    apply_boost=True,
                    required_entities=query_entities,
                )
                runtime.update(
                    {
                        "degraded": True,
                        "fallbackStrategy": "graph_filter",
                        "notes": [
                            "A query simbolica nao retornou documentos suficientes; fallback usa busca vetorial filtrada por entidades."
                        ],
                    }
                )
            return graph, self.tag_documents(docs[:top_k], "cypher"), runtime

        graph = self.neo4j.subgraph_for_seeds(entities, hops=hops, limit=180)

        if strategy == "graph_filter":
            linked_entities = self.top_graph_node_names(graph, limit=64)
            required_entities = self.unique_names([*entities, *linked_entities])
            runtime.update(
                {
                    "graphSeeds": entities,
                    "derivedEntities": linked_entities[:12],
                    "notes": ["O indice vetorial e varrido com filtro duro: documentos precisam mencionar sementes ou nos do subgrafo."],
                }
            )
            docs = self.retrieve_text_ranked(
                question,
                entities=entities,
                graph=graph,
                mode="hybrid",
                limit=top_k,
                apply_boost=True,
                required_entities=required_entities,
                boost_entities=linked_entities,
            )
            if len(docs) < top_k:
                docs = self.merge_documents(
                    docs,
                    self.entity_documents(required_entities[:40], question, method="graph_filter_mentions", limit=top_k),
                )
            return graph, self.tag_documents(docs[:top_k], "graph_filter"), runtime

        if strategy == "path":
            path_focus = self.path_focus(entities, graph)
            path_entities = path_focus["entities"]
            if not path_focus["hasRelationalFocus"]:
                runtime.update(
                    {
                        "graphSeeds": entities,
                        "pathEntities": path_entities,
                        "degraded": True,
                        "fallbackStrategy": "kg_index",
                        "notes": [
                            "Path retrieval exige pelo menos duas entidades e um caminho/conector; fallback usa KG-as-Index."
                        ],
                    }
                )
                docs = self.retrieve_text_ranked(
                    question,
                    entities=entities,
                    graph=graph,
                    mode="hybrid",
                    limit=top_k,
                    apply_boost=True,
                )
                return graph, self.tag_documents(docs[:top_k], "path_fallback_kg_index"), runtime
            runtime.update(
                {
                    "graphSeeds": entities,
                    "pathEntities": path_entities,
                    "pathsFound": path_focus["paths"],
                    "connectorsFound": path_focus["connectors"],
                    "notes": ["Caminhos curtos e conectores 2-hop recebem peso extra no reranking."],
                }
            )
            candidates = self.retrieve_text_ranked(
                question,
                entities=entities,
                graph=graph,
                mode="hybrid",
                limit=max(top_k * 10, 48),
                apply_boost=True,
                boost_entities=path_entities,
            )
            docs = self.rerank_by_mentions(candidates, entities, path_entities, "path", seed_pair_bonus=True)
            focused_docs = self.filter_docs_by_mentions(docs, [*entities, *path_entities])
            if len(focused_docs) >= max(2, top_k // 3):
                docs = focused_docs
            return graph, self.tag_documents(docs[:top_k], "path"), runtime

        docs = self.retrieve_text_ranked(
            question,
            entities=entities,
            graph=graph,
            mode="hybrid",
            limit=top_k,
            apply_boost=True,
        )
        runtime["notes"] = ["Estrategia padrao: subgrafo k-hop reforca o ranking vetorial."]
        return graph, self.tag_documents(docs, "kg_index"), runtime

    @staticmethod
    def unique_names(names: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    @staticmethod
    def ranked_mentions(documents: list[dict[str, Any]], limit: int = 8) -> list[str]:
        counts: Counter[str] = Counter()
        for doc in documents:
            for name in doc.get("mentions") or []:
                counts[name] += 1
        return [name for name, _count in counts.most_common(limit)]

    @staticmethod
    def top_graph_node_names(graph: dict[str, Any], limit: int = 32) -> list[str]:
        nodes = graph.get("nodes") or []
        ranked = sorted(
            [node for node in nodes if node.get("name")],
            key=lambda node: (
                -(float(node.get("pagerank") or 0.0)),
                -(float(node.get("weightedDegree") or 0.0)),
                str(node.get("name") or ""),
            ),
        )
        return [str(node["name"]) for node in ranked[:limit]]

    def path_focus(self, entities: list[str], graph: dict[str, Any]) -> dict[str, Any]:
        paths = self.shortest_paths(entities, limit=4)
        connectors = self.connector_rows(entities, graph.get("edges") or [], limit=8)
        names: list[str] = []
        seed_set = set(entities)
        for path in paths:
            names.extend(name for name in path.get("path") or [] if name not in seed_set)
        names.extend(row["name"] for row in connectors)
        focus_entities = self.unique_names(names)
        return {
            "entities": focus_entities,
            "paths": paths,
            "connectors": connectors,
            "hasRelationalFocus": len(entities) >= 2 and bool(paths or connectors),
        }

    @staticmethod
    def filter_docs_by_mentions(documents: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
        wanted = set(names)
        return [doc for doc in documents if set(doc.get("mentions") or []) & wanted]

    @staticmethod
    def merge_documents(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for doc in group:
                key = str(doc.get("id") or len(seen))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return merged

    def entity_documents(
        self,
        names: list[str],
        question: str,
        method: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        docs = self.neo4j.documents_for_entities(self.unique_names(names), limit=max(limit, 1))
        query_tokens = tokenize(question)
        for idx, doc in enumerate(docs, start=1):
            entity_hits = int(doc.get("entityHits") or 0)
            doc["sourceRank"] = idx
            doc["sourceBucket"] = doc.get("sourceType") or "other"
            doc["score"] = float(entity_hits) + max(0.0, 1.0 - idx / (limit + 2))
            doc["vectorScore"] = None
            doc["graphBoost"] = float(entity_hits)
            doc["retrievalMethod"] = method
            doc["snippet"] = snippet(str(doc.get("text") or ""), query_tokens, max_chars=780)
        return docs

    @staticmethod
    def rerank_by_mentions(
        documents: list[dict[str, Any]],
        seed_entities: list[str],
        focus_entities: list[str],
        strategy: str,
        seed_pair_bonus: bool,
    ) -> list[dict[str, Any]]:
        seed_set = set(seed_entities)
        focus_set = set(focus_entities)
        reranked: list[dict[str, Any]] = []
        for doc in documents:
            mentions = set(doc.get("mentions") or [])
            focus_hits = mentions & focus_set
            seed_hits = mentions & seed_set
            extra = len(focus_hits) * 0.04
            if seed_pair_bonus and len(seed_hits) >= 2:
                extra += 0.06
            enriched = dict(doc)
            enriched["score"] = float(enriched.get("score") or 0.0) + extra
            enriched["graphBoost"] = float(enriched.get("graphBoost") or 0.0) + extra
            enriched["strategyFocusHits"] = sorted(focus_hits)
            enriched["strategyMethod"] = strategy
            reranked.append(enriched)
        reranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return reranked

    @staticmethod
    def tag_documents(documents: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
        tagged: list[dict[str, Any]] = []
        for idx, doc in enumerate(documents, start=1):
            enriched = dict(doc)
            enriched["sourceRank"] = idx
            enriched["strategyMethod"] = strategy
            base_method = str(enriched.get("retrievalMethod") or "retrieval")
            if strategy not in base_method:
                enriched["retrievalMethod"] = f"{base_method}+{strategy}"
            tagged.append(enriched)
        return tagged

    @staticmethod
    def retrieval_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
        methods = sorted({doc.get("retrievalMethod") or "unknown" for doc in documents})
        by_source = Counter(doc.get("sourceType") or "other" for doc in documents)
        has_boost = any(float(doc.get("graphBoost") or 0.0) > 0 for doc in documents)
        has_vector = any(doc.get("vectorScore") is not None for doc in documents)
        has_symbolic = any("cypher" in str(doc.get("retrievalMethod") or "") for doc in documents)
        if has_vector and has_boost:
            score_mode = "cosine+graph-boost"
        elif has_vector:
            score_mode = "cosine"
        elif has_symbolic:
            score_mode = "symbolic entity hits"
        elif documents:
            score_mode = "bm25"
        else:
            score_mode = "none"
        best_doc = max(
            documents,
            key=lambda item: float(item.get("score") or 0.0),
            default=None,
        )
        return {
            "method": methods[0] if len(methods) == 1 else ("+".join(methods) if methods else "none"),
            "documents": len(documents),
            "bySource": dict(sorted(by_source.items())),
            "scoreMode": score_mode,
            "topScore": float(best_doc.get("score") or 0.0) if best_doc else 0.0,
            "topVectorScore": best_doc.get("vectorScore") if best_doc else None,
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
        strategy_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        entities = [match["name"] for match in entity_matches]
        strategy_id = self.normalize_graph_rag_strategy(strategy_runtime.get("strategy") if mode == "hybrid" else None)
        strategy = dict(GRAPH_RAG_STRATEGIES[strategy_id])
        edges = graph.get("edges") or []
        nodes = graph.get("nodes") or []
        relationship_counts = Counter(edge.get("type") or "UNKNOWN" for edge in edges)
        boosted_documents = [doc for doc in documents if float(doc.get("graphBoost") or 0.0) > 0]
        symbolic_query = mode == "hybrid" and strategy_id == "cypher"
        direct_edges = [] if symbolic_query else self.direct_edges(entities, edges)
        connectors = [] if symbolic_query else self.connector_rows(entities, edges)
        paths = [] if symbolic_query else (self.shortest_paths(entities) if mode in {"graph", "hybrid"} else [])
        context = context_sections.get("selected") or ""

        return {
            "variant": {
                "id": strategy_id,
                "name": strategy["name"] if mode == "hybrid" else ("RAG textual" if mode == "rag" else "Graph-only"),
                "shortName": strategy["shortName"] if mode == "hybrid" else mode,
                "subtitle": strategy["subtitle"] if mode == "hybrid" else "modo nao-hibrido",
                "description": strategy["description"] if mode == "hybrid" else "",
                "active": mode == "hybrid",
                "flow": strategy["flow"] if mode == "hybrid" else [],
                "risk": strategy["risk"] if mode == "hybrid" else "",
                "references": strategy["references"] if mode == "hybrid" else [],
                "implemented": [strategy["graph"], strategy["text"], strategy["synthesis"]] if mode == "hybrid" else [],
                "notImplemented": [],
                "degraded": bool(strategy_runtime.get("degraded")) if mode == "hybrid" else False,
                "fallbackStrategy": strategy_runtime.get("fallbackStrategy") if mode == "hybrid" else None,
            },
            "strategy": {
                "mode": mode,
                "graphRagStrategy": strategy_id if mode == "hybrid" else None,
                "graph": strategy["graph"] if mode == "hybrid" else ("deterministic_k_hop" if mode == "graph" else "disabled"),
                "text": strategy["text"] if mode == "hybrid" else ("pure_vector_similarity" if mode == "rag" else "disabled"),
                "synthesis": strategy["synthesis"] if mode == "hybrid" else "ollama_or_retrieval_only",
                "scoreFormula": strategy["scoreFormula"] if mode == "hybrid" else "n/a",
                "runtime": strategy_runtime,
                "notes": strategy_runtime.get("notes") or [],
                "degraded": bool(strategy_runtime.get("degraded")),
                "fallbackStrategy": strategy_runtime.get("fallbackStrategy"),
            },
            "grounding": {
                "question": question,
                "entities": entity_matches,
                "entityCount": len(entity_matches),
                "graphSeeds": strategy_runtime.get("graphSeeds") or entities,
                "derivedEntities": strategy_runtime.get("derivedEntities") or [],
                "pathEntities": strategy_runtime.get("pathEntities") or [],
                "communityEntities": strategy_runtime.get("communityEntities") or [],
                "queryEntities": strategy_runtime.get("queryEntities") or [],
            },
            "graph": {
                "hops": hops,
                "nodeCount": len(nodes),
                "edgeCount": len(edges),
                "relationshipTypes": dict(sorted(relationship_counts.items())),
                "paths": paths,
                "directEdges": direct_edges,
                "connectors": connectors,
                "query": self.graph_query_trace(entities, hops, strategy_id, strategy_runtime),
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
                {
                    "id": "graph",
                    "label": strategy["shortName"] if mode == "hybrid" else "Subgrafo",
                    "value": f"{len(nodes)} nos / {len(edges)} arestas",
                },
                {
                    "id": "retrieval",
                    "label": "Evidencias",
                    "value": f"{len(documents)} docs / {len(boosted_documents)} boosted",
                },
                {"id": "prompt", "label": "Prompt", "value": f"~{max(1, len(context) // 4) if context else 0} tokens"},
            ],
        }

    @staticmethod
    def apply_llm_trace(trace: dict[str, Any], llm_trace: dict[str, Any] | None) -> dict[str, Any]:
        if not llm_trace:
            return trace
        prompt = trace.setdefault("prompt", {})
        prompt.update(
            {
                "template": llm_trace.get("template"),
                "schema": llm_trace.get("schemaName") or llm_trace.get("schema"),
                "selectedChars": llm_trace.get("promptChars", prompt.get("selectedChars", 0)),
                "estimatedTokens": llm_trace.get("estimatedTokens", prompt.get("estimatedTokens", 0)),
                "preview": llm_trace.get("preview") or prompt.get("preview", ""),
                "attempts": llm_trace.get("attempts"),
                "adapter": llm_trace.get("adapter"),
                "status": llm_trace.get("status"),
            }
        )
        return trace

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
            "strategyFocusHits": sorted(doc.get("strategyFocusHits") or []),
            "strategyMethod": doc.get("strategyMethod"),
            "boostReason": reason,
            "snippet": doc.get("snippet") or doc.get("text") or "",
        }

    @staticmethod
    def graph_query_trace(
        entities: list[str],
        hops: int,
        strategy: str = "kg_index",
        strategy_runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        strategy_runtime = strategy_runtime or {}
        if strategy == "community":
            if strategy_runtime.get("degraded"):
                return {
                    "label": "community fallback: deterministic k-hop expansion",
                    "parameters": {"seeds": entities, "hops": hops, "limit": 180},
                    "cypher": GraphRAG.graph_query_trace(entities, hops, "kg_index", strategy_runtime)["cypher"],
                }
            return {
                "label": "community subgraph selection",
                "parameters": {"seeds": entities, "limit": 180},
                "cypher": (
                    "MATCH (seed:Character)\n"
                    "WHERE seed.name IN $seeds AND seed.community IS NOT NULL\n"
                    "WITH collect(DISTINCT seed.community) AS communities\n"
                    "MATCH p = (a:Character)-[r]-(b:Character)\n"
                    "WHERE a.community IN communities AND b.community = a.community\n"
                    "RETURN p ORDER BY coalesce(r.weight, r.confidence, 1) DESC LIMIT $limit"
                ),
            }
        if strategy == "cypher":
            if strategy_runtime.get("degraded"):
                return {
                    "label": "symbolic query fallback",
                    "parameters": {
                        "entities": strategy_runtime.get("queryEntities") or entities,
                        "fallbackStrategy": strategy_runtime.get("fallbackStrategy"),
                    },
                    "cypher": (
                        "/* symbolic query requires grounded entities; this execution used the fallback reported above */\n"
                        "MATCH (d:RetrievalDocument) RETURN d ORDER BY d.sequence LIMIT $limit"
                    ),
                }
            return {
                "label": "deterministic symbolic MENTIONS query",
                "parameters": {"entities": strategy_runtime.get("queryEntities") or entities, "limit": 180},
                "cypher": (
                    "MATCH (d:RetrievalDocument)-[:MENTIONS]->(e:Entity)\n"
                    "WHERE e.name IN $entities\n"
                    "WITH d, count(DISTINCT e) AS entityHits\n"
                    "ORDER BY entityHits DESC, d.sequence LIMIT $limit\n"
                    "MATCH (d)-[m:MENTIONS]->(entity:Entity)\n"
                    "WHERE entity.name IN $entities\n"
                    "RETURN d, m, entity"
                ),
            }
        if strategy == "vector_first":
            return {
                "label": "vector-derived k-hop expansion",
                "parameters": {
                    "questionEntities": strategy_runtime.get("questionEntities") or entities,
                    "derivedEntities": strategy_runtime.get("derivedEntities") or [],
                    "seeds": strategy_runtime.get("graphSeeds") or entities,
                    "hops": hops,
                    "limit": 180,
                },
                "cypher": (
                    "/* seeds come from entities mentioned by initial vector hits */\n"
                    "MATCH (seed:Entity)\n"
                    "WHERE seed.name IN $seeds\n"
                    f"MATCH p = (seed)-[*1..{max(1, min(int(hops), 4))}]-(n:Entity)\n"
                    "RETURN p LIMIT $limit"
                ),
            }
        if strategy == "path":
            if strategy_runtime.get("degraded"):
                return {
                    "label": "path fallback: deterministic k-hop expansion",
                    "parameters": {
                        "seeds": entities,
                        "hops": hops,
                        "fallbackStrategy": strategy_runtime.get("fallbackStrategy"),
                    },
                    "cypher": GraphRAG.graph_query_trace(entities, hops, "kg_index", strategy_runtime)["cypher"],
                }
            return {
                "label": "shortest paths + 2-hop connectors",
                "parameters": {
                    "seeds": entities,
                    "source": entities[0] if entities else None,
                    "target": entities[1] if len(entities) > 1 else None,
                    "pathEntities": strategy_runtime.get("pathEntities") or [],
                    "limit": 180,
                },
                "cypher": (
                    "MATCH (a:Entity {name: $source})\n"
                    "MATCH (b:Entity {name: $target})\n"
                    "MATCH p = shortestPath((a)-[*..5]-(b))\n"
                    "RETURN p"
                ),
            }
        if strategy == "graph_filter":
            return {
                "label": "k-hop expansion + vector search constrained by MENTIONS",
                "parameters": {"seeds": entities, "hops": hops, "limit": 180},
                "cypher": (
                    "MATCH (seed:Entity)\n"
                    "WHERE seed.name IN $seeds\n"
                    f"MATCH p = (seed)-[*1..{max(1, min(int(hops), 4))}]-(n:Entity)\n"
                    "WITH collect(DISTINCT n.name) + $seeds AS graphNames\n"
                    "MATCH (d:RetrievalDocument)-[:MENTIONS]->(e:Entity)\n"
                    "WHERE e.name IN graphNames\n"
                    "WITH d, graphNames, count(DISTINCT e) AS graphHits\n"
                    "ORDER BY graphHits DESC LIMIT $limit\n"
                    "MATCH (d)-[m:MENTIONS]->(entity:Entity)\n"
                    "WHERE entity.name IN graphNames\n"
                    "RETURN d, m, entity"
                ),
            }
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
                "RETURN p\n"
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
        return self.build_context_sections(question, entities, graph, documents, hops, mode, {})["selected"]

    def build_context_sections(
        self,
        question: str,
        entities: list[str],
        graph: dict[str, Any],
        documents: list[dict[str, Any]],
        hops: int,
        mode: str,
        strategy_runtime: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        strategy_runtime = strategy_runtime or {}
        text_context = (
            self.build_text_context(question, documents, mode=mode, strategy_runtime=strategy_runtime)
            if mode in {"rag", "hybrid"}
            else ""
        )
        graph_context = (
            self.build_graph_context(question, entities, graph, hops=hops, strategy_runtime=strategy_runtime)
            if mode in {"graph", "hybrid"}
            else ""
        )
        hybrid_context = (
            self.build_hybrid_context(question, text_context, graph_context, strategy_runtime)
            if mode == "hybrid"
            else ""
        )
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

    def build_text_context(
        self,
        question: str,
        documents: list[dict[str, Any]],
        mode: str,
        strategy_runtime: dict[str, Any] | None = None,
    ) -> str:
        strategy_runtime = strategy_runtime or {}
        strategy_id = self.normalize_graph_rag_strategy(strategy_runtime.get("strategy") if mode == "hybrid" else None)
        strategy = GRAPH_RAG_STRATEGIES[strategy_id]
        lines: list[str] = []
        lines.append(f"Pergunta: {question}")
        if mode == "rag":
            lines.append("Modo de retrieval: rag textual puro")
            lines.append("Score: similaridade textual apenas.")
        elif mode == "hybrid":
            lines.append(f"Modo de retrieval: GraphRAG / {strategy['name']}")
            lines.append(f"Score: {strategy['scoreFormula']}")
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
        strategy_runtime: dict[str, Any] | None = None,
    ) -> str:
        strategy_runtime = strategy_runtime or {}
        strategy_id = self.normalize_graph_rag_strategy(strategy_runtime.get("strategy"))
        symbolic_query = strategy_id == "cypher"
        lines: list[str] = []
        lines.append(f"Pergunta: {question}")
        if symbolic_query:
            lines.append("Modo de retrieval: query simbolica sobre MENTIONS")
            lines.append("Padrao: (RetrievalDocument)-[:MENTIONS]->(Entity)")
        else:
            lines.append("Modo de retrieval: graph estrutural")
            lines.append(f"Profundidade k-hop: {hops}")

        if entities:
            lines.append("Entidades detectadas: " + ", ".join(entities))
        else:
            lines.append("Entidades detectadas: nenhuma.")

        if len(entities) >= 2 and not symbolic_query:
            lines.append("")
            lines.append("Caminhos curtos entre entidades detectadas:")
            for idx, source in enumerate(entities):
                for target in entities[idx + 1 :]:
                    path = self.neo4j.shortest_path(source, target, max_depth=5)
                    if path:
                        lines.append(f"- {source} -> {target}: " + " -> ".join(path))

        if graph.get("edges"):
            lines.append("")
            if symbolic_query:
                lines.append("Arestas recuperadas da query MENTIONS:")
            else:
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
            if symbolic_query:
                lines.append("Nenhuma linha MENTIONS foi recuperada. Responda apenas com a incerteza apropriada.")
            else:
                lines.append("Nenhum subgrafo especifico foi recuperado. Responda apenas com a incerteza apropriada.")

        return "\n".join(lines)

    @staticmethod
    def build_hybrid_context(
        question: str,
        text_context: str,
        graph_context: str,
        strategy_runtime: dict[str, Any] | None = None,
    ) -> str:
        strategy_runtime = strategy_runtime or {}
        strategy_id = GraphRAG.normalize_graph_rag_strategy(strategy_runtime.get("strategy"))
        strategy = GRAPH_RAG_STRATEGIES[strategy_id]
        notes = strategy_runtime.get("notes") or []
        return "\n\n".join(
            [
                f"Pergunta: {question}",
                f"Modo de retrieval: GraphRAG / {strategy['name']}",
                f"Definicao da variante: {strategy['description']}",
                f"Fluxo: {' -> '.join(strategy['flow'])}",
                f"Risco interpretativo: {strategy['risk']}",
                *([f"Notas da execucao: {'; '.join(notes)}"] if notes else []),
                "=== Evidencia estrutural ===",
                graph_context,
                "=== Evidencia textual ===",
                text_context,
            ]
        )

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
            documents.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            for idx, doc in enumerate(documents, start=1):
                doc["globalRank"] = idx
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
        required_entities: list[str] | None = None,
        boost_entities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if boost_entities is None:
            graph_names = {
                node["name"]
                for node in graph.get("nodes", [])
                if node.get("name") and node.get("name") not in entities
            }
        else:
            graph_names = {name for name in boost_entities if name and name not in entities}
        if self.vector_store.exists():
            try:
                results = self.vector_store.search(
                    question,
                    self.ollama,
                    model=settings.ollama_embed_model,
                    limit=limit,
                    seed_entities=entities,
                    graph_entities=sorted(graph_names) if apply_boost else [],
                    required_entities=required_entities or [],
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
                    required_entities=required_entities,
                    boost_entities=boost_entities,
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
            required_entities=required_entities,
            boost_entities=boost_entities,
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
        required_entities: list[str] | None = None,
        boost_entities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        docs = self.neo4j.retrieval_documents()
        if not docs:
            return []
        if source_type:
            docs = [doc for doc in docs if doc.get("sourceType") == source_type]
        required_set = set(required_entities or [])
        if required_set:
            docs = [doc for doc in docs if set(doc.get("mentions") or []) & required_set]
        if not docs:
            return []

        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        if boost_entities is None:
            graph_names = {
                node["name"]
                for node in graph.get("nodes", [])
                if node.get("name") and node.get("name") not in entities
            }
        else:
            graph_names = {name for name in boost_entities if name and name not in entities}
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
            prefix = "Sintese com LLM indisponivel; exibindo somente evidencias recuperadas. "

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
