from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EvidenceSource = Literal["text", "graph"]
Confidence = Literal["low", "medium", "high"]


class EvidenceItem(BaseModel):
    source: EvidenceSource = Field(..., description="Origem da evidencia: text ou graph.")
    label: str = Field(..., description="Titulo curto da evidencia.")
    detail: str = Field(..., description="Resumo curto do suporte recuperado.")


class GroundedAnswer(BaseModel):
    answer: str = Field(..., description="Resposta final curta, direta e em portugues brasileiro.")
    textEvidence: list[EvidenceItem] = Field(default_factory=list, description="Evidencias vindas de chunks/falas.")
    graphEvidence: list[EvidenceItem] = Field(default_factory=list, description="Evidencias vindas de nos/arestas/caminhos.")
    limits: str = Field("", description="Limite da resposta quando a evidencia for insuficiente.")
    confidence: Confidence = Field("medium", description="Confianca calibrada pela evidencia recuperada.")

    def to_markdown(self) -> str:
        parts = [self.answer.strip()]
        if self.textEvidence:
            parts.append("Evidencias textuais")
            parts.extend(f"- {item.label}: {item.detail}" for item in self.textEvidence[:3])
        if self.graphEvidence:
            parts.append("Evidencias estruturais")
            parts.extend(f"- {item.label}: {item.detail}" for item in self.graphEvidence[:3])
        if self.limits.strip():
            parts.append("Limite")
            parts.append(self.limits.strip())
        parts.append(f"Confianca: {self.confidence}")
        return "\n\n".join(part for part in parts if part)


class CypherDraft(BaseModel):
    query: str = Field(..., description="Consulta Cypher unica e read-only.")
    explanation: str = Field("", description="Uma frase curta explicando o que a query recupera.")
    warnings: list[str] = Field(default_factory=list, description="Alertas sobre ambiguidade ou fallback.")


class LLMTrace(BaseModel):
    provider: str = "ollama"
    adapter: str = "langchain-ollama"
    model: str
    template: str
    schemaName: str
    promptChars: int
    estimatedTokens: int
    attempts: int
    status: str
    error: str | None = None
    preview: str = ""
