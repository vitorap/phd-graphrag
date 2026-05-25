from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.config import settings
from app.llm_contracts import CypherDraft, GroundedAnswer, LLMTrace
from app.llm_prompts import (
    answer_prompt,
    answer_repair_prompt,
    cypher_generation_prompt,
    cypher_generation_repair_prompt,
    cypher_synthesis_prompt,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class LLMRunResult:
    answer: str
    structured_answer: GroundedAnswer | None
    trace: LLMTrace
    status: str
    raw: str = ""


@dataclass
class CypherRunResult:
    draft: CypherDraft
    trace: LLMTrace
    raw: str = ""


class LLMContractError(RuntimeError):
    pass


class LLMService:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.num_ctx = settings.ollama_num_ctx

    def synthesize_answer(
        self,
        *,
        question: str,
        mode: str,
        context_sections: dict[str, str],
        strategy: dict[str, Any] | None = None,
        strategy_runtime: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMRunResult:
        selected = context_sections.get("selected") or ""
        prompt = answer_prompt(mode)
        variables = {
            "question": question,
            "mode": mode,
            "strategy": json.dumps(strategy or {}, ensure_ascii=False),
            "runtime": json.dumps(strategy_runtime or {}, ensure_ascii=False),
            "text_evidence": context_sections.get("text") or "Nenhuma evidencia textual fornecida.",
            "graph_evidence": context_sections.get("graph") or "Nenhuma evidencia estrutural fornecida.",
            "selected_evidence": selected or "Nenhuma evidencia selecionada.",
        }
        parsed, trace, raw = self._invoke_with_retry(
            prompt=prompt,
            schema=GroundedAnswer,
            template=f"answer:{mode}",
            variables=variables,
            model=model,
            mode=mode,
        )
        answer = self._grounded_answer_to_markdown(parsed)
        return LLMRunResult(
            answer=answer,
            structured_answer=parsed,
            trace=trace,
            status="retrieval+langchain+ollama",
            raw=raw,
        )

    def generate_cypher(self, question: str, model: str | None = None) -> CypherRunResult:
        schema = """
Labels principais:
- Entity(name, kind, pagerank, community, weightedDegree)
- Character(name, race, gender, pagerank, community, weightedDegree)
- Race(name), Place(name), Weapon(name), Chapter(title, bookTitle)
- RetrievalDocument(id, sourceType, sourceTitle, chapterTitle, speaker, text)
- TextChunk, DialogueLine

Relacoes principais:
- CO_OCCURS_WITH(weight), INTERACTS_WITH(weight), ENEMY_OF, FRIEND_OF
- HAS_RACE, HAS_WEAPON, IN_CHAPTER, MENTIONS, SPEAKS_LINE
- SIMILAR_CHAPTER(weight), PREDICTED_LINK(confidence, method)
""".strip()
        examples = """
Pergunta: caminho entre Frodo e Sauron
Cypher:
MATCH (a:Entity {name: 'Frodo'})
MATCH (b:Entity {name: 'Sauron'})
MATCH p = shortestPath((a)-[*..5]-(b))
WHERE all(rel IN relationships(p) WHERE type(rel) <> 'PREDICTED_LINK')
RETURN p, [n IN nodes(p) | n.name] AS caminho, length(p) AS saltos
LIMIT 20

Pergunta: relacoes entre elfos e orcs
Cypher:
MATCH p = (elf:Character)-[r]-(orc:Character)
WHERE elf.race = 'Elf' AND orc.race = 'Orc'
RETURN p, elf.name AS elfo, orc.name AS orc, type(r) AS relacao, coalesce(r.weight, r.confidence, 1) AS peso
ORDER BY peso DESC
LIMIT 30
""".strip()
        variables = {"question": question, "schema": schema, "examples": examples}
        try:
            parsed, trace, raw = self._invoke_once(
                prompt=cypher_generation_prompt(),
                schema=CypherDraft,
                template="cypher_generate",
                variables=variables,
                model=model,
            )
        except Exception as first_error:
            repair_variables = {
                **variables,
                "error": str(first_error),
                "bad_output": getattr(first_error, "raw_output", ""),
            }
            parsed, trace, raw = self._invoke_once(
                prompt=cypher_generation_repair_prompt(),
                schema=CypherDraft,
                template="cypher_generate:repair",
                variables=repair_variables,
                model=model,
                attempts=2,
            )
        return CypherRunResult(draft=parsed, trace=trace, raw=raw)

    def synthesize_cypher(
        self,
        *,
        question: str,
        query: str,
        rows: list[dict[str, Any]],
        graph: dict[str, Any],
        model: str | None = None,
    ) -> LLMRunResult:
        structural_context = self.cypher_structural_context(query=query, rows=rows, graph=graph)
        parsed, trace, raw = self._invoke_with_retry(
            prompt=cypher_synthesis_prompt(),
            schema=GroundedAnswer,
            template="cypher_synthesize",
            variables={
                "question": question,
                "structural_context": json.dumps(structural_context, ensure_ascii=False),
                "text_evidence": "Nenhuma evidencia textual fornecida.",
                "graph_evidence": json.dumps(structural_context, ensure_ascii=False),
                "selected_evidence": json.dumps(structural_context, ensure_ascii=False),
                "strategy": "{}",
                "runtime": "{}",
            },
            model=model,
            mode="graph",
        )
        answer = self._grounded_answer_to_markdown(parsed)
        return LLMRunResult(
            answer=answer,
            structured_answer=parsed,
            trace=trace,
            status="retrieval+langchain+ollama",
            raw=raw,
        )

    @staticmethod
    def cypher_structural_context(*, query: str, rows: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
        nodes = (graph.get("nodes") or [])[:80]
        edges = (graph.get("edges") or [])[:120]
        return {
            "query": query,
            "nodes": [
                {
                    "name": node.get("name"),
                    "labels": node.get("labels"),
                    "race": node.get("race"),
                    "pagerank": node.get("pagerank"),
                    "community": node.get("community"),
                }
                for node in nodes
            ],
            "edges": [
                {
                    "source": edge.get("sourceName"),
                    "type": edge.get("type"),
                    "target": edge.get("targetName"),
                    "weight": edge.get("weight"),
                    "confidence": edge.get("confidence"),
                    "method": edge.get("method"),
                }
                for edge in edges
            ],
            "rows": rows[:20],
            "counts": {
                "nodes": len(graph.get("nodes") or []),
                "edges": len(graph.get("edges") or []),
                "rows": len(rows),
            },
        }

    def _chat(self, model: str | None = None) -> ChatOllama:
        return ChatOllama(
            model=model or self.model,
            base_url=self.base_url,
            reasoning=False,
            disable_streaming=True,
            temperature=0.1,
            num_ctx=self.num_ctx,
            num_predict=420,
            format="json",
        )

    def _invoke_with_retry(
        self,
        *,
        prompt: ChatPromptTemplate,
        schema: type[SchemaT],
        template: str,
        variables: dict[str, Any],
        model: str | None,
        mode: str,
    ) -> tuple[SchemaT, LLMTrace, str]:
        try:
            parsed, trace, raw = self._invoke_once(
                prompt=prompt,
                schema=schema,
                template=template,
                variables=variables,
                model=model,
            )
            self._validate_answer_contract(parsed, mode)
            return parsed, trace, raw
        except Exception as first_error:
            repair_variables = {
                **variables,
                "error": str(first_error),
                "bad_output": getattr(first_error, "raw_output", ""),
            }
            parsed, trace, raw = self._invoke_once(
                prompt=answer_repair_prompt(mode),
                schema=schema,
                template=f"{template}:repair",
                variables=repair_variables,
                model=model,
                attempts=2,
            )
            self._validate_answer_contract(parsed, mode)
            return parsed, trace, raw

    def _invoke_once(
        self,
        *,
        prompt: ChatPromptTemplate,
        schema: type[SchemaT],
        template: str,
        variables: dict[str, Any],
        model: str | None,
        attempts: int = 1,
    ) -> tuple[SchemaT, LLMTrace, str]:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt_vars = {**variables, "format_instructions": parser.get_format_instructions()}
        messages = prompt.format_messages(**prompt_vars)
        rendered_prompt = "\n\n".join(str(message.content) for message in messages)
        preview = rendered_prompt[:1800]
        raw = ""
        try:
            response = self._chat(model).invoke(messages)
            raw = str(getattr(response, "content", "") or "")
            parsed = self._parse_structured(raw, schema, parser)
            trace = self._trace(
                model=model or self.model,
                template=template,
                schema_name=schema.__name__,
                prompt_chars=len(rendered_prompt),
                preview=preview,
                attempts=attempts,
                status="ok",
            )
            return parsed, trace, raw
        except Exception as exc:
            setattr(exc, "raw_output", raw)
            raise

    @staticmethod
    def _parse_structured(raw: str, schema: type[SchemaT], parser: PydanticOutputParser) -> SchemaT:
        try:
            return parser.parse(raw)
        except Exception as parse_error:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise parse_error
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise parse_error
            return schema.model_validate(LLMService._normalize_structured_payload(data, schema))

    @staticmethod
    def _normalize_structured_payload(data: Any, schema: type[SchemaT]) -> Any:
        if schema is not GroundedAnswer or not isinstance(data, dict):
            return data

        normalized = dict(data)
        if "answer" not in normalized or normalized["answer"] is None:
            normalized["answer"] = ""
        normalized["answer"] = str(normalized["answer"]).strip()
        normalized["limits"] = str(normalized.get("limits") or "").strip()
        confidence = str(normalized.get("confidence") or "medium").lower()
        normalized["confidence"] = confidence if confidence in {"low", "medium", "high"} else "medium"

        for field_name, source, label in [
            ("textEvidence", "text", "Evidencia textual"),
            ("graphEvidence", "graph", "Evidencia estrutural"),
        ]:
            raw_items = normalized.get(field_name) or []
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = []

            evidence_items: list[dict[str, str]] = []
            for idx, item in enumerate(raw_items[:6], start=1):
                if isinstance(item, str):
                    detail = item.strip()
                    if detail:
                        evidence_items.append({"source": source, "label": f"{label} {idx}", "detail": detail})
                    continue
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                entry["source"] = source
                entry["label"] = str(
                    entry.get("label")
                    or entry.get("title")
                    or entry.get("sourceTitle")
                    or entry.get("name")
                    or f"{label} {idx}"
                ).strip()
                detail = (
                    entry.get("detail")
                    or entry.get("snippet")
                    or entry.get("text")
                    or entry.get("description")
                    or entry.get("value")
                )
                entry["detail"] = str(detail or "").strip()
                if entry["detail"]:
                    evidence_items.append(
                        {
                            "source": entry["source"],
                            "label": entry["label"],
                            "detail": entry["detail"],
                        }
                    )
            normalized[field_name] = evidence_items
        return normalized

    def _trace(
        self,
        *,
        model: str,
        template: str,
        schema_name: str,
        preview: str,
        prompt_chars: int,
        attempts: int,
        status: str,
        error: str | None = None,
    ) -> LLMTrace:
        return LLMTrace(
            model=model,
            template=template,
            schemaName=schema_name,
            promptChars=prompt_chars,
            estimatedTokens=max(1, prompt_chars // 4) if prompt_chars else 0,
            attempts=attempts,
            status=status,
            error=error,
            preview=preview,
        )

    @staticmethod
    def _validate_answer_contract(parsed: BaseModel, mode: str) -> None:
        if not isinstance(parsed, GroundedAnswer):
            return
        answer_text = json.dumps(parsed.model_dump(), ensure_ascii=False).lower()
        if mode == "rag":
            forbidden = ["grafo", "aresta", "subgrafo", "caminho", "centralidade", "coocorr"]
            if parsed.graphEvidence or any(term in answer_text for term in forbidden):
                raise LLMContractError("RAG textual vazou evidencia estrutural.")
        if mode == "graph":
            forbidden = ["chunk", "chunks", "fala", "falas", "trecho", "trechos", "script", "texto narrativo"]
            if parsed.textEvidence or any(term in answer_text for term in forbidden):
                raise LLMContractError("Graph-only vazou evidencia textual.")

    @staticmethod
    def _grounded_answer_to_markdown(answer: GroundedAnswer) -> str:
        return answer.to_markdown()
