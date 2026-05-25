from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm_contracts import CypherDraft, EvidenceItem, GroundedAnswer
from app.llm_prompts import (
    answer_prompt,
    cypher_generation_prompt,
    cypher_generation_repair_prompt,
    cypher_synthesis_prompt,
)
from app.llm_service import LLMContractError, LLMService


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def render_prompt_smoke() -> None:
    parser = PydanticOutputParser(pydantic_object=GroundedAnswer)
    for mode in ["rag", "graph", "hybrid"]:
        messages = answer_prompt(mode).format_messages(
            question="Como Frodo se conecta a Sauron?",
            mode=mode,
            strategy="{}",
            runtime="{}",
            text_evidence="Chunk #1: Frodo carrega o Anel.",
            graph_evidence="Frodo -[CO_OCCURS_WITH]-> Sauron",
            selected_evidence="Evidencia selecionada.",
            format_instructions=parser.get_format_instructions(),
        )
        rendered = "\n".join(str(message.content) for message in messages)
        require("Como Frodo se conecta a Sauron?" in rendered, f"{mode}: pergunta ausente")
        require("JSON" in rendered, f"{mode}: contrato JSON ausente")

    cypher_parser = PydanticOutputParser(pydantic_object=CypherDraft)
    cypher_generation_prompt().format_messages(
        question="caminho entre Frodo e Sauron",
        schema="Entity(name)",
        examples="MATCH (a) RETURN a LIMIT 1",
        format_instructions=cypher_parser.get_format_instructions(),
    )
    cypher_generation_repair_prompt().format_messages(
        question="caminho entre Frodo e Sauron",
        schema="Entity(name)",
        examples="MATCH (a) RETURN a LIMIT 1",
        error="saida sem JSON",
        bad_output="MATCH (a) RETURN a",
        format_instructions=cypher_parser.get_format_instructions(),
    )
    cypher_synthesis_prompt().format_messages(
        question="Como Frodo se conecta a Sauron?",
        structural_context='{"nodes": [], "edges": []}',
        format_instructions=parser.get_format_instructions(),
    )


def schema_smoke() -> None:
    answer = GroundedAnswer(
        answer="Frodo se conecta a Sauron pela missao envolvendo o Anel.",
        textEvidence=[EvidenceItem(source="text", label="Texto", detail="Trecho menciona a missao.")],
        graphEvidence=[EvidenceItem(source="graph", label="Grafo", detail="Aresta de coocorrencia.")],
        confidence="medium",
    )
    require("Evidencias textuais" in answer.to_markdown(), "markdown estruturado sem evidencias textuais")
    CypherDraft(query="MATCH (a) RETURN a LIMIT 1", explanation="Retorna nos.")
    normalized = LLMService._normalize_structured_payload(
        {
            "answer": "Frodo se conecta a Sauron pelo Anel.",
            "textEvidence": ["Trecho textual recuperado."],
            "graphEvidence": ["Frodo -[CO_OCCURS_WITH]-> Sauron"],
            "confidence": "medium",
        },
        GroundedAnswer,
    )
    parsed = GroundedAnswer.model_validate(normalized)
    require(parsed.textEvidence[0].source == "text", "normalizacao perdeu source textual")
    require(parsed.graphEvidence[0].source == "graph", "normalizacao perdeu source estrutural")


def contract_smoke() -> None:
    rag_bad = GroundedAnswer(
        answer="Frodo tem uma aresta no grafo.",
        graphEvidence=[EvidenceItem(source="graph", label="Aresta", detail="Frodo -> Sauron")],
    )
    try:
        LLMService._validate_answer_contract(rag_bad, "rag")
    except LLMContractError:
        pass
    else:
        raise AssertionError("RAG aceitou evidencia estrutural")

    graph_bad = GroundedAnswer(
        answer="Um trecho do livro explica a relacao.",
        textEvidence=[EvidenceItem(source="text", label="Trecho", detail="Texto narrativo")],
    )
    try:
        LLMService._validate_answer_contract(graph_bad, "graph")
    except LLMContractError:
        pass
    else:
        raise AssertionError("Graph aceitou evidencia textual")


def live_ollama_smoke() -> None:
    service = LLMService()
    contexts = {
        "text": "Pergunta: Como Frodo se conecta a Sauron?\nChunks dos livros:\n- Frodo carrega uma missao secreta ligada ao Anel.",
        "graph": "Pergunta: Como Frodo se conecta a Sauron?\nArestas recuperadas:\n- Frodo -[CO_OCCURS_WITH, peso=29]-> Sauron",
        "hybrid": "Evidencia estrutural: Frodo -[CO_OCCURS_WITH]-> Sauron\nEvidencia textual: Frodo carrega a missao secreta.",
        "selected": "",
    }
    contexts["selected"] = contexts["hybrid"]

    rag = service.synthesize_answer(
        question="Como Frodo se conecta a Sauron?",
        mode="rag",
        context_sections={"text": contexts["text"], "graph": "", "hybrid": "", "selected": contexts["text"]},
    )
    require(not rag.structured_answer.graphEvidence, "RAG live retornou graphEvidence")
    require("grafo" not in rag.answer.lower(), "RAG live vazou termo de grafo")

    graph = service.synthesize_answer(
        question="Como Frodo se conecta a Sauron?",
        mode="graph",
        context_sections={"text": "", "graph": contexts["graph"], "hybrid": "", "selected": contexts["graph"]},
    )
    require(graph.structured_answer.graphEvidence, "Graph live nao retornou graphEvidence")

    hybrid = service.synthesize_answer(
        question="Como Frodo se conecta a Sauron?",
        mode="hybrid",
        context_sections=contexts,
        strategy={"name": "KG-as-Index / Graph Boost"},
    )
    require(hybrid.structured_answer.textEvidence, "Hybrid live sem textEvidence")
    require(hybrid.structured_answer.graphEvidence, "Hybrid live sem graphEvidence")


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida contratos LangChain/Ollama dos prompts.")
    parser.add_argument("--with-ollama", action="store_true", help="tambem chama o Ollama local")
    args = parser.parse_args()

    render_prompt_smoke()
    schema_smoke()
    contract_smoke()
    if args.with_ollama:
        live_ollama_smoke()
    print("LLM prompt contracts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
