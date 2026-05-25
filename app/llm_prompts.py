from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


ANSWER_CONTRACTS: dict[str, str] = {
    "rag": (
        "Modo: RAG textual puro. Use somente chunks/falas recuperados. "
        "Nao mencione grafo, arestas, subgrafo, caminhos, centralidade, coocorrencia ou evidencia estrutural. "
        "Se os textos nao sustentarem a resposta, use limits."
    ),
    "graph": (
        "Modo: Graph-only. Use somente estrutura do grafo: nos, labels, propriedades, arestas, pesos, caminhos, "
        "comunidades e linhas tabulares. Nao use chunks, falas, livros ou texto narrativo como evidencia. "
        "Qualifique coocorrencia como sinal estrutural, nao como causalidade."
    ),
    "hybrid": (
        "Modo: GraphRAG hibrido. Use evidencia estrutural para explicar a conexao e evidencia textual para sustentar "
        "a interpretacao narrativa. Separe claramente textEvidence e graphEvidence."
    ),
}


def answer_prompt(mode: str) -> ChatPromptTemplate:
    contract = ANSWER_CONTRACTS.get(mode, ANSWER_CONTRACTS["hybrid"])
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Voce escreve a resposta final de um sistema RAG/GraphRAG de producao sobre Senhor dos Aneis. "
                    "Responda em portugues brasileiro, com tom direto, natural e conclusivo. "
                    "Use somente as evidencias fornecidas; nao use conhecimento externo e nao invente fatos, arestas, "
                    "falas, capitulos ou citacoes. Nao exponha raciocinio interno, passos ocultos ou thinking. "
                    "Nao escreva frases meta sobre aula, apresentacao, turma, demo ou sistema. "
                    "Comece pela resposta em 1 ou 2 frases, sem 'com base no contexto' ou equivalentes. "
                    "Use no maximo 3 evidencias textuais e 3 evidencias estruturais. "
                    "Use limits somente se a evidencia for vazia, contraditoria ou insuficiente. "
                    f"{contract}\n\n"
                    "Responda exclusivamente em JSON valido no schema abaixo.\n{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "/no_think\n"
                    "Pergunta: {question}\n"
                    "Modo: {mode}\n"
                    "Estrategia GraphRAG: {strategy}\n"
                    "Runtime: {runtime}\n\n"
                    "Evidencia textual:\n{text_evidence}\n\n"
                    "Evidencia estrutural:\n{graph_evidence}\n\n"
                    "Contexto selecionado para a estrategia:\n{selected_evidence}\n\n"
                    "Gere apenas o JSON final."
                ),
            ),
        ]
    )


def answer_repair_prompt(mode: str) -> ChatPromptTemplate:
    contract = ANSWER_CONTRACTS.get(mode, ANSWER_CONTRACTS["hybrid"])
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Corrija a resposta para obedecer estritamente ao contrato e ao schema JSON. "
                    f"{contract}\n\n"
                    "Nao adicione conhecimento externo. Responda exclusivamente em JSON valido.\n{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "/no_think\n"
                    "Pergunta: {question}\n"
                    "Problema encontrado: {error}\n\n"
                    "Resposta anterior:\n{bad_output}\n\n"
                    "Evidencia textual:\n{text_evidence}\n\n"
                    "Evidencia estrutural:\n{graph_evidence}\n\n"
                    "Gere o JSON corrigido."
                ),
            ),
        ]
    )


def cypher_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Voce gera Cypher read-only para uma demo Neo4j de Senhor dos Aneis. "
                    "A query deve ser uma unica consulta, sem ponto e virgula, sem CREATE/MERGE/DELETE/SET/REMOVE/DROP, "
                    "sem APOC e sem GDS. Sempre inclua LIMIT. Para perguntas visuais, prefira RETURN p ou RETURN a, r, b. "
                    "Use apenas labels, propriedades e relacoes do schema fornecido. "
                    "Responda exclusivamente em JSON valido no schema abaixo.\n{format_instructions}"
                ),
            ),
            (
                "human",
                "/no_think\nSchema:\n{schema}\n\nExemplos:\n{examples}\n\nPergunta do usuario: {question}",
            ),
        ]
    )


def cypher_generation_repair_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Corrija a saida para obedecer estritamente ao schema JSON CypherDraft. "
                    "A query deve ser uma unica consulta Cypher read-only, sem ponto e virgula, sem CREATE/MERGE/DELETE/SET/REMOVE/DROP, "
                    "sem APOC, sem GDS e sempre com LIMIT. "
                    "Nao explique fora do JSON.\n{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "/no_think\nSchema Neo4j disponivel:\n{schema}\n\n"
                    "Exemplos:\n{examples}\n\n"
                    "Pergunta do usuario: {question}\n\n"
                    "Problema encontrado: {error}\n\n"
                    "Saida anterior:\n{bad_output}\n\n"
                    "Gere apenas o JSON corrigido."
                ),
            ),
        ]
    )


def cypher_synthesis_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Voce escreve a resposta final de um sistema Graph-only de producao sobre Senhor dos Aneis. "
                    "Use apenas nos, labels, propriedades, arestas, pesos, comunidades, paths e linhas tabulares fornecidas. "
                    "Nao use conhecimento externo, chunks, falas, livros ou texto narrativo como evidencia. "
                    "Comece pela resposta em 1 ou 2 frases. Depois, se ajudar, use graphEvidence. "
                    "Diferencie relacao direta, caminho, coocorrencia, comunidade e link predito. "
                    "Nao trate coocorrencia, caminho ou comunidade como causalidade. "
                    "Se a query nao trouxe estrutura suficiente, use limits e nao conclua que algo nao existe no grafo inteiro. "
                    "Responda exclusivamente em JSON valido no schema abaixo.\n{format_instructions}"
                ),
            ),
            (
                "human",
                "/no_think\nPergunta: {question}\n\nEvidencia estrutural disponivel:\n{structural_context}",
            ),
        ]
    )
