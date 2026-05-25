from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field

from app.config import settings
from app.graphrag import GraphRAG
from app.neo4j_client import Neo4jClient
from app.ollama_client import OllamaClient
from app.vector_store import VectorStore


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="GraphRAG em Middle-earth")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    hops: int = Field(2, ge=1, le=4)
    top_k: int = Field(8, ge=1, le=24)
    mode: str = Field("hybrid", pattern="^(rag|graph|hybrid|baseline)$")
    model: str | None = None
    use_llm: bool = False


class VectorSearchRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(8, ge=1, le=24)
    mode: str = Field("rag", pattern="^(rag|hybrid)$")
    hops: int = Field(2, ge=1, le=4)
    source_type: str | None = Field(None, pattern="^(book|dialogue)$")


class CypherRequest(BaseModel):
    query: str = Field(..., min_length=3)
    limit: int = Field(100, ge=1, le=300)


class CypherGenerateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    model: str | None = None


class CypherSynthesizeRequest(BaseModel):
    question: str = Field(..., min_length=3)
    query: str = Field(..., min_length=3)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    graph: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


CYPHER_EXAMPLES = [
    {
        "id": "frodo-neighbors",
        "title": "Frodo: vizinhanca renderizavel",
        "explain": "Retorna paths reais para a UI e para o Neo4j Browser desenharem o subgrafo.",
        "lesson": "1-hop e o campo receptivo imediato: cada vizinho pode contribuir uma mensagem para atualizar a representacao de Frodo.",
        "gnn": "Message passing camada 1: agrega atributos e relacoes dos vizinhos diretos.",
        "visual": {
            "center": "Frodo",
            "hops": 1,
            "caption": "A query retorna p, entao tabela e visualizacao usam o mesmo resultado estrutural.",
        },
        "query": """
MATCH p = (:Entity {name: 'Frodo'})-[r]-(n:Entity)
WHERE type(r) <> 'MENTIONS' AND type(r) <> 'SPEAKS_LINE'
WITH p, r, n
ORDER BY coalesce(r.weight, r.confidence, 1) DESC, n.name
LIMIT 25
RETURN p,
       n.name AS entidade,
       labels(n) AS labels,
       type(r) AS relacao,
       coalesce(r.weight, r.confidence, 1) AS peso
ORDER BY peso DESC, entidade
""".strip(),
    },
    {
        "id": "frodo-sauron-path",
        "title": "Frodo -> Sauron",
        "explain": "Mostra a ponte estrutural sem deixar link predito encurtar a historia.",
        "lesson": "Esta consulta transforma uma pergunta narrativa em uma pergunta de caminho: qual sequencia de entidades liga heroi e antagonista?",
        "gnn": "Caminhos curtos indicam como informacao pode fluir em poucas camadas; tambem mostram onde um GraphRAG deve buscar contexto.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "A visualizacao abre 2-hop em Frodo para mostrar pontes provaveis ate Sauron.",
        },
        "query": """
MATCH (a:Entity {name: 'Frodo'})
MATCH (b:Entity {name: 'Sauron'})
MATCH p = shortestPath((a)-[*..5]-(b))
WHERE all(rel IN relationships(p) WHERE type(rel) <> 'PREDICTED_LINK')
RETURN p,
       [n IN nodes(p) | n.name] AS caminho,
       [r IN relationships(p) | type(r)] AS relacoes,
       length(p) AS saltos
""".strip(),
    },
    {
        "id": "frodo-sauron-bridges",
        "title": "Frodo/Sauron: pontes 2-hop",
        "explain": "Mostra os intermediarios que aparecem entre heroi e antagonista.",
        "lesson": "O grafo encontra candidatos de explicacao antes do texto: Anel, Mordor, inimigos, coocorrencias ou outros conectores.",
        "gnn": "2-hop corresponde a duas rodadas de propagacao; aumenta cobertura, mas tambem pode trazer ruido.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "O grafo mostra o campo receptivo 2-hop que gera os conectores.",
        },
        "query": """
MATCH p = (f:Entity {name: 'Frodo'})-[r1]-(m:Entity)-[r2]-(s:Entity {name: 'Sauron'})
WHERE m.name <> 'Frodo' AND m.name <> 'Sauron'
WITH p, m, r1, r2
ORDER BY coalesce(r1.weight, r1.confidence, 1) + coalesce(r2.weight, r2.confidence, 1) DESC, m.name
LIMIT 25
RETURN p,
       m.name AS ponte,
       labels(m) AS labels,
       type(r1) AS relacao_com_frodo,
       type(r2) AS relacao_com_sauron,
       coalesce(r1.weight, r1.confidence, 1) + coalesce(r2.weight, r2.confidence, 1) AS forca
ORDER BY forca DESC, ponte
""".strip(),
    },
    {
        "id": "elf-orc-relations",
        "title": "Elfos e Orcs",
        "explain": "Compara relacoes entre racas em um subgrafo de conflito/coocorrencia.",
        "lesson": "Perguntas por grupos pedem filtragem por atributos dos nos antes de analisar as arestas.",
        "gnn": "Atributos como race podem virar features iniciais de nos em uma GNN relacional.",
        "visual": {
            "center": "Legolas",
            "hops": 1,
            "caption": "A query retorna personagens Elf-Orc e as relacoes entre eles.",
        },
        "query": """
MATCH p = (elf:Character)-[r]-(orc:Character)
WHERE elf.race = 'Elf'
  AND orc.race = 'Orc'
  AND type(r) IN ['CO_OCCURS_WITH', 'INTERACTS_WITH', 'ENEMY_OF', 'PREDICTED_LINK']
WITH p, elf, orc, r
ORDER BY coalesce(r.weight, r.confidence, 1) DESC, elf.name, orc.name
LIMIT 30
RETURN p,
       elf.name AS elfo,
       orc.name AS orc,
       type(r) AS relacao,
       coalesce(r.weight, r.confidence, 1) AS peso
""".strip(),
    },
    {
        "id": "race-conflict",
        "title": "Racas e conflito",
        "explain": "Mostra conflitos e coocorrencias fortes entre personagens de racas diferentes.",
        "lesson": "A interpretacao muda quando uma aresta atravessa comunidades ou atributos de grupo.",
        "gnn": "Grafos heterogeneos combinam atributos de nos e tipos de arestas na mensagem.",
        "visual": {
            "center": "Sauron",
            "hops": 2,
            "caption": "O subgrafo destaca relacoes entre grupos narrativos diferentes.",
        },
        "query": """
MATCH p = (a:Character)-[r]-(b:Character)
WHERE a.race IS NOT NULL
  AND b.race IS NOT NULL
  AND a.race <> b.race
  AND type(r) IN ['ENEMY_OF', 'CO_OCCURS_WITH', 'INTERACTS_WITH']
WITH p, a, b, r
ORDER BY CASE type(r) WHEN 'ENEMY_OF' THEN 1000 ELSE coalesce(r.weight, 1) END DESC, a.name, b.name
LIMIT 35
RETURN p,
       a.name AS personagem_a,
       a.race AS raca_a,
       b.name AS personagem_b,
       b.race AS raca_b,
       type(r) AS relacao,
       coalesce(r.weight, r.confidence, 1) AS peso
""".strip(),
    },
    {
        "id": "frodo-community",
        "title": "Comunidade de Frodo",
        "explain": "Renderiza o subgrafo local de personagens na mesma comunidade estrutural.",
        "lesson": "Comunidades agrupam personagens que compartilham vizinhancas parecidas e resumem representacao de grupo.",
        "gnn": "Embeddings de nos tendem a aproximar entidades com contexto estrutural parecido.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "A query mostra a vizinhanca filtrada pela comunidade de Frodo.",
        },
        "query": """
MATCH (f:Character {name: 'Frodo'})
MATCH p = (f)-[*1..2]-(c:Character)
WHERE c.community = f.community
  AND all(rel IN relationships(p) WHERE type(rel) <> 'PREDICTED_LINK')
WITH p, c
ORDER BY length(p), coalesce(c.pagerank, 0) DESC
LIMIT 35
RETURN p,
       c.name AS personagem,
       c.race AS raca,
       round(coalesce(c.pagerank, 0), 5) AS pagerank,
       c.community AS comunidade
""".strip(),
    },
    {
        "id": "sauron-connectors",
        "title": "Sauron e conectores",
        "explain": "Expande Sauron em 2-hop para revelar intermediarios fortes.",
        "lesson": "Um centro antagonista ativa muitos caminhos; o desafio e separar sinal estrutural de ruido.",
        "gnn": "Mais hops aumentam campo receptivo, mas tambem diluem a mensagem.",
        "visual": {
            "center": "Sauron",
            "hops": 2,
            "caption": "A query retorna paths ate conectores de Sauron.",
        },
        "query": """
MATCH p = (:Entity {name: 'Sauron'})-[*1..2]-(n:Entity)
WHERE all(rel IN relationships(p)
          WHERE type(rel) <> 'MENTIONS' AND (type(rel) <> 'PREDICTED_LINK' OR coalesce(rel.confidence, 0) >= 0.25))
WITH p, n
ORDER BY length(p), coalesce(n.pagerank, 0) DESC
LIMIT 40
RETURN p,
       n.name AS entidade,
       labels(n) AS labels,
       length(p) AS saltos
""".strip(),
    },
    {
        "id": "predicted-links",
        "title": "Links preditos como hipoteses",
        "explain": "Mostra arestas inferidas como camada separada de hipoteses.",
        "lesson": "Arestas observadas e hipoteses preditas ficam separadas, preservando auditoria para link prediction.",
        "gnn": "Link prediction e uma tarefa classica de GNN: aprender embeddings para estimar arestas ausentes.",
        "visual": {
            "center": "Sauron",
            "hops": 2,
            "caption": "A visualizacao em Sauron destaca onde links preditos podem aparecer como hipoteses.",
        },
        "query": """
MATCH p = (a:Character)-[r:PREDICTED_LINK]->(b:Character)
WITH p, a, b, r
ORDER BY coalesce(r.confidence, 0) DESC, a.name, b.name
LIMIT 30
RETURN p,
       a.name AS origem,
       b.name AS destino,
       round(coalesce(r.confidence, 0), 3) AS confianca,
       r.method AS metodo
""".strip(),
    },
    {
        "id": "weapon-bearers",
        "title": "Armas e portadores",
        "explain": "Mostra personagens conectados a armas e artefatos.",
        "lesson": "Relacoes semanticas da ontologia criam explicacoes diferentes de coocorrencia textual.",
        "gnn": "Tipos de nos e arestas indicam mensagens diferentes em um grafo heterogeneo.",
        "visual": {
            "center": "Sting",
            "hops": 1,
            "caption": "A query retorna personagens, armas e relacoes semanticas.",
        },
        "query": """
MATCH p = (c:Entity)-[r:HAS_WEAPON]-(w:Weapon)
WITH p, c, w, r
ORDER BY c.name, w.name
LIMIT 30
RETURN p,
       c.name AS personagem,
       w.name AS arma,
       type(r) AS relacao
""".strip(),
    },
    {
        "id": "shared-documents",
        "title": "Documentos que mencionam Frodo e Sauron",
        "explain": "Exemplo propositalmente tabular: mostra documentos, nao subgrafo.",
        "lesson": "Nem toda consulta Cypher deve virar grafo. Aqui o foco e auditoria documental para conectar Graph com GraphRAG.",
        "gnn": "Documentos ligados por MENTIONS podem virar nos de outro grafo, mas esta consulta retorna valores escalares.",
        "visual": {
            "center": "Sauron",
            "hops": 1,
            "caption": "Este exemplo retorna tabela; o aviso da UI deve explicar que nao ha subgrafo renderizavel.",
        },
        "query": """
MATCH (d:RetrievalDocument)-[:MENTIONS]->(:Entity {name: 'Frodo'})
MATCH (d)-[:MENTIONS]->(:Entity {name: 'Sauron'})
RETURN d.sourceTitle AS fonte,
       coalesce(d.chapterTitle, d.speaker, d.sourceType) AS secao,
       labels(d) AS tipo,
       substring(d.text, 0, 220) AS trecho,
       d.mentionCount AS mencoes
ORDER BY mencoes DESC, fonte
LIMIT 12
""".strip(),
    },
]


LECTURE_STEPS = [
    {
        "id": "setup",
        "title": "Uma pergunta, tres recuperacoes",
        "duration": "4 min",
        "mode": "compare",
        "question": "Como Frodo se conecta a Sauron?",
        "center": "Frodo",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre a comparacao. A mesma pergunta vira tres objetos de evidencia: chunks, subgrafo e hibrido.",
        "talkingPoints": [
            "RAG, Graph e GraphRAG recuperam evidencias de natureza diferente.",
            "A pergunta Frodo-Sauron exige relacao estrutural e explicacao narrativa.",
            "A resposta so faz sentido depois de auditar o contexto recuperado.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Pode recuperar chunks parecidos, mas nao ve caminhos nem arestas."},
            {"label": "Graph", "text": "Mostra ENEMY_OF, coocorrencia, conectores e caminhos, mas nao explica a cena."},
            {"label": "GraphRAG", "text": "Usa o subgrafo como filtro/boost e sintetiza evidencias textuais."},
        ],
        "architecture": {
            "name": "Mapa da aula",
            "flow": ["pergunta", "retrieval textual", "retrieval estrutural", "fusao", "sintese"],
            "takeaway": "GraphRAG e uma familia de formas de acoplar texto, grafo e LLM.",
            "risk": "Comparar respostas sem mostrar contexto recuperado vira demo enganosa.",
        },
        "speakerNotes": [
            "Comece em retrieval-only para provar que nada esta roteirizado.",
            "Depois ligue a sintese com LLM e mostre que ela resume evidencias recuperadas.",
        ],
        "quiz": {
            "question": "Se o RAG puro recuperar chunks ruins para Frodo-Sauron, qual conclusao e correta?",
            "options": [
                {"label": "A relacao nao existe no corpus.", "correct": False, "explanation": "Errado. Falha de retrieval nao prova ausencia global no corpus."},
                {"label": "A execucao textual nao trouxe evidencia suficiente.", "correct": True, "explanation": "Correto. A conclusao honesta fica limitada aos chunks recuperados."},
                {"label": "O LLM deve completar com memoria externa.", "correct": False, "explanation": "Errado. Isso quebra a premissa de resposta ancorada."},
            ],
            "answer": "A resposta correta e a segunda: fale sobre a evidencia recuperada, nao sobre o corpus inteiro.",
        },
    },
    {
        "id": "rag-good",
        "title": "RAG textual para pergunta narrativa",
        "duration": "5 min",
        "mode": "rag",
        "question": "Por que o Anel importa para Frodo?",
        "center": "Frodo",
        "topK": 8,
        "demoAction": "Aplicar abre RAG puro. Observe top-k separado para livros e scripts, score cosine e ausencia de boost.",
        "talkingPoints": [
            "A pergunta vira embedding; cada chunk/fala tambem.",
            "O ranking e textual: cosine puro, sem k-hop ou PageRank.",
            "Esse modo e forte quando a resposta esta em passagens narrativas proximas.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Melhor para citar trechos e falas recuperadas."},
            {"label": "Graph", "text": "Pode mostrar relacoes, mas nao substitui a narrativa."},
            {"label": "GraphRAG", "text": "Ajuda quando a pergunta tambem exige entidades e conexoes."},
        ],
        "architecture": {
            "name": "RAG classico",
            "flow": ["query embedding", "vector search", "top-k textual", "prompt com chunks", "resposta"],
            "takeaway": "RAG textual e um indice semantico, nao um motor estrutural.",
            "risk": "Chunk pequeno perde contexto; chunk grande dilui a evidencia.",
        },
        "speakerNotes": [
            "Mostre que livros e scripts sao fontes textuais diferentes.",
            "Use sem LLM primeiro; depois ligue a sintese.",
        ],
        "quiz": {
            "question": "O que o score cosine mede no RAG vetorial?",
            "options": [
                {"label": "Similaridade entre pergunta e unidade textual.", "correct": True, "explanation": "Correto. Ele mede proximidade no espaco de embeddings."},
                {"label": "Importancia global do personagem no grafo.", "correct": False, "explanation": "Isso seria metrica estrutural, como PageRank."},
                {"label": "Veracidade factual da resposta final.", "correct": False, "explanation": "Score de retrieval nao valida a resposta final."},
            ],
            "answer": "Cosine mede similaridade vetorial entre pergunta e unidade recuperavel.",
        },
    },
    {
        "id": "rag-limit",
        "title": "RAG textual para pergunta relacional",
        "duration": "5 min",
        "mode": "rag",
        "question": "Como Frodo se conecta a Sauron?",
        "center": "Frodo",
        "topK": 8,
        "demoAction": "Aplicar mostra a falha honesta: se os chunks top-k nao sustentam a relacao, o sistema deve dizer isso.",
        "talkingPoints": [
            "Perguntas relacionais pedem conectores, tipos de aresta e caminhos multi-hop.",
            "Um chunk isolado pode citar Frodo sem Sauron, ou Sauron sem explicar Frodo.",
            "O erro didatico seria deixar o LLM preencher a lacuna com conhecimento externo.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Recupera texto parecido, mas pode perder a ponte estrutural."},
            {"label": "Graph", "text": "Recupera a ponte, mas nao necessariamente a narrativa."},
            {"label": "GraphRAG", "text": "Recupera a ponte e usa essa estrutura para buscar texto melhor."},
        ],
        "architecture": {
            "name": "Limite de RAG puro",
            "flow": ["pergunta relacional", "top-k local", "contexto parcial", "sintese cautelosa"],
            "takeaway": "Muitas vezes a evidencia certa nunca entrou no prompt.",
            "risk": "Confundir resposta fluente com resposta ancorada.",
        },
        "speakerNotes": [
            "Se o RAG recuperar coisa ruim, isso e bom para a demo: mostra a fronteira do metodo.",
            "Essa etapa justifica o grafo sem demonizar RAG.",
        ],
        "quiz": {
            "question": "Qual problema aparece quando a resposta depende de uma ponte entre entidades?",
            "options": [
                {"label": "Context drift ou evidencia espalhada.", "correct": True, "explanation": "Correto. A evidencia pode estar distribuida em varios trechos e entidades."},
                {"label": "O embedding deixa de funcionar matematicamente.", "correct": False, "explanation": "Nao. Ele funciona, mas pode ranquear evidencia insuficiente."},
                {"label": "Neo4j deixa de ser necessario.", "correct": False, "explanation": "Pelo contrario: a pergunta motiva o grafo."},
            ],
            "answer": "O problema central e evidencia espalhada: o top-k textual pode nao reconstruir a ponte relacional.",
        },
    },
    {
        "id": "graph-structural",
        "title": "Graph-only: caminhos e conectores",
        "duration": "6 min",
        "mode": "graph",
        "question": "Qual o caminho estrutural entre Frodo e Sauron?",
        "center": "Frodo",
        "hops": 2,
        "cypherExample": "frodo-sauron-path",
        "demoAction": "Aplicar abre o exemplo Cypher de menor caminho e o subgrafo em Frodo.",
        "talkingPoints": [
            "A evidencia primaria e simbolica: nos, arestas, pesos e tipos de relacao.",
            "O grafo responde bem a conectividade, centralidade e vizinhanca.",
            "O limite e que estrutura pode ser seca ou ambigua sem texto narrativo.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Nao garante caminho entre entidades."},
            {"label": "Graph", "text": "Mostra caminho, relacao direta, conectores 2-hop e pesos."},
            {"label": "GraphRAG", "text": "Usa essa estrutura como guia para recuperar texto."},
        ],
        "architecture": {
            "name": "Graph as Structural Context",
            "flow": ["entidades sementes", "subgrafo k-hop", "triplas/caminhos", "linearizacao", "prompt"],
            "takeaway": "Aqui o grafo nao indexa texto: o proprio grafo vira contexto.",
            "risk": "Subgrafo denso explode a janela de contexto; e preciso podar.",
        },
        "speakerNotes": [
            "A query e parte da explicacao procedural.",
            "Compare relacao direta com caminho: sao evidencias diferentes.",
        ],
        "quiz": {
            "question": "Qual afirmacao sobre Graph-only e mais precisa?",
            "options": [
                {"label": "Ele substitui completamente o texto.", "correct": False, "explanation": "Errado. Pode faltar narrativa."},
                {"label": "Ele torna a pergunta verificavel por query e subgrafo.", "correct": True, "explanation": "Correto. A resposta pode ser auditada por arestas e caminhos."},
                {"label": "Ele sempre usa embeddings.", "correct": False, "explanation": "Nao neste modo. Aqui a recuperacao e estrutural."},
            ],
            "answer": "Graph-only e forte porque torna a evidencia verificavel por query, caminhos e arestas.",
        },
    },
    {
        "id": "kg-index",
        "title": "GraphRAG: KG-as-Index",
        "duration": "5 min",
        "mode": "hybrid",
        "question": "Como Frodo se conecta a Sauron?",
        "center": "Frodo",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre GraphRAG. O subgrafo ativa Frodo/Sauron e reforca evidencias textuais relacionadas.",
        "talkingPoints": [
            "O KG funciona como andaime de recuperacao.",
            "Um chunk entra por cosine e tambem pode ganhar reforco por estar conectado ao subgrafo.",
            "A explicabilidade e topologica: o trecho apareceu porque esta ligado ao caminho.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Top-k textual puro."},
            {"label": "Graph", "text": "Subgrafo sem texto."},
            {"label": "GraphRAG", "text": "Top-k textual condicionado por entidades e vizinhos."},
        ],
        "architecture": {
            "name": "KG-as-Index",
            "flow": ["embedding inicial", "entidades sementes", "expansao k-hop", "chunks ligados", "sintese"],
            "takeaway": "O grafo ajuda a saltar entre trechos distantes que compartilham entidades relevantes.",
            "risk": "Se a deteccao de entidades errar, o grafo amplifica o erro.",
        },
        "speakerNotes": [
            "Nossa implementacao atual e mais proxima de KG-as-Index + Hybrid.",
            "O grafo acha a estrutura; o texto explica a narrativa.",
        ],
        "quiz": {
            "question": "No KG-as-Index, qual e o papel principal do grafo?",
            "options": [
                {"label": "Organizar e expandir a recuperacao de chunks.", "correct": True, "explanation": "Correto. O grafo funciona como indice estrutural."},
                {"label": "Gerar embeddings no lugar do modelo.", "correct": False, "explanation": "Errado. Embeddings continuam vindo do modelo de embedding."},
                {"label": "Responder sem consultar texto.", "correct": False, "explanation": "Isso e Graph as Structural Context."},
            ],
            "answer": "O grafo organiza a recuperacao textual e justifica por que certos chunks entram no contexto.",
        },
    },
    {
        "id": "architectures",
        "title": "Cinco arquiteturas GraphRAG",
        "duration": "7 min",
        "mode": "compare",
        "question": "Qual arquitetura voce usaria para auditar a relacao entre Frodo, Gollum e o Anel?",
        "center": "Gollum",
        "hops": 2,
        "topK": 6,
        "demoAction": "Aplicar abre Compare. Antes de rodar, discuta qual arquitetura seria mais adequada.",
        "talkingPoints": [
            "KG-as-Index recupera texto via topologia.",
            "Structural Context injeta triplas/caminhos linearizados.",
            "Text-to-Query gera Cypher/SPARQL; KG Builder cria o grafo; Hybrid funde rankings.",
        ],
        "methodContrasts": [
            {"label": "KG-as-Index", "text": "Bom quando o texto correto existe, mas esta distante."},
            {"label": "Text-to-Query", "text": "Bom quando a pergunta pode virar consulta simbolica precisa."},
            {"label": "Hybrid", "text": "Bom quando texto e estrutura sao ambos necessarios."},
        ],
        "architecture": {
            "name": "Espectro neuro-simbolico",
            "flow": ["KG-as-Index", "Structural Context", "Text-to-Query", "KG Builder", "Hybrid"],
            "takeaway": "GraphRAG nao e uma arquitetura unica; e uma familia de acoplamentos.",
            "risk": "Misturar construcao do KG com recuperacao para QA confunde avaliacao.",
        },
        "speakerNotes": [
            "Nossa demo cobre Graph-only, KG-as-Index e Hybrid; Text-to-Query aparece nos exemplos Cypher.",
            "KG Builder e a etapa de construcao: extrair, definir e canonicalizar triplas.",
        ],
        "quiz": {
            "question": "Qual variante tem a query gerada como principal objeto auditavel?",
            "options": [
                {"label": "Text-to-Query", "correct": True, "explanation": "Correto. A query Cypher/SPARQL vira evidencia procedural."},
                {"label": "KG Builder", "correct": False, "explanation": "KG Builder foca extracao e canonicalizacao de triplas."},
                {"label": "RAG vetorial puro", "correct": False, "explanation": "RAG puro audita ranking textual, nao query simbolica."},
            ],
            "answer": "Text-to-Query, porque a query formal e parte central da explicacao.",
        },
    },
    {
        "id": "gnn-bridge",
        "title": "Ponte com GNN: k-hop como campo receptivo",
        "duration": "5 min",
        "mode": "graph",
        "question": "O que muda quando aumento hops de 1 para 3?",
        "center": "Frodo",
        "hops": 1,
        "cypherExample": "frodo-neighbors",
        "demoAction": "Aplicar abre a vizinhanca de Frodo. Altere hops e observe cobertura versus ruido.",
        "talkingPoints": [
            "Em GNN, cada camada agrega mensagens de vizinhos.",
            "Em GraphRAG, cada hop expande o contexto estrutural que pode ativar texto.",
            "Over-squashing e ruido tem analogia pratica: muitos caminhos competem dentro de pouco contexto.",
        ],
        "methodContrasts": [
            {"label": "1-hop", "text": "Alta precisao local; pouca cobertura."},
            {"label": "2-hop", "text": "Bom compromisso para conectores e entidades intermediarias."},
            {"label": "3-hop", "text": "Mais cobertura, maior risco de ruido e drift."},
        ],
        "architecture": {
            "name": "Message passing sem treinar GNN",
            "flow": ["semente", "vizinhos", "agregacao/poda", "contexto", "resposta"],
            "takeaway": "A demo usa intuicao de GNN para recuperar contexto, nao para aprender embeddings de nos.",
            "risk": "Mais hops nao significa melhor resposta; significa maior campo receptivo.",
        },
        "speakerNotes": [
            "Essa e a ponte conceitual com a disciplina de GNN.",
            "Fale de receptive field, agregacao, ruido e interpretabilidade.",
        ],
        "quiz": {
            "question": "Qual analogia correta entre GNN e GraphRAG k-hop?",
            "options": [
                {"label": "Cada hop amplia o campo de informacao disponivel.", "correct": True, "explanation": "Correto. E a intuicao de receptive field."},
                {"label": "Cada hop treina uma nova rede neural.", "correct": False, "explanation": "Nao ha treino de GNN nessa demo."},
                {"label": "Mais hops sempre reduzem ruido.", "correct": False, "explanation": "Mais hops podem aumentar ruido."},
            ],
            "answer": "Cada hop amplia o campo de informacao, como camadas de message passing ampliam o receptive field.",
        },
    },
    {
        "id": "challenge",
        "title": "Desafio: escolher a recuperacao certa",
        "duration": "5 min",
        "mode": "compare",
        "question": "Quem conecta Frodo a Mordor e por que isso importa para a missao?",
        "center": "Frodo",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre Compare. Antes de rodar, peca para a turma escolher qual coluna deve vencer.",
        "talkingPoints": [
            "Perguntas narrativas favorecem RAG textual.",
            "Perguntas topologicas favorecem Graph-only.",
            "Perguntas relacionais com explicacao favorecem GraphRAG hibrido.",
        ],
        "methodContrasts": [
            {"label": "Narrativa", "text": "Por que o Anel importa para Frodo? -> RAG tende a ser suficiente."},
            {"label": "Estrutura", "text": "Qual o menor caminho Frodo-Sauron? -> Graph e o metodo natural."},
            {"label": "Relacao + explicacao", "text": "Como Frodo se conecta a Sauron? -> GraphRAG tende a ser mais completo."},
        ],
        "architecture": {
            "name": "Hybrid Vector + Graph",
            "flow": ["busca vetorial", "busca em grafo", "boost/fusao", "evidencias contrastadas", "sintese"],
            "takeaway": "O melhor sistema escolhe retrieval pela pergunta, nao por dogma.",
            "risk": "Sem mostrar a fusao, Hybrid parece so dois sistemas lado a lado.",
        },
        "speakerNotes": [
            "Feche com decisao de engenharia: qual falha esperada cada metodo cobre?",
            "Mostre que comparar evidencias e mais importante do que uma resposta bonita.",
        ],
        "quiz": {
            "question": "Qual pergunta e melhor para demonstrar GraphRAG hibrido?",
            "options": [
                {"label": "Uma pergunta factual localizada em um paragrafo.", "correct": False, "explanation": "RAG puro provavelmente resolve bem."},
                {"label": "Uma pergunta so de ranking estrutural.", "correct": False, "explanation": "Graph-only provavelmente basta."},
                {"label": "Uma pergunta relacional que tambem pede explicacao narrativa.", "correct": True, "explanation": "Correto. Ela precisa de grafo e texto ao mesmo tempo."},
            ],
            "answer": "GraphRAG brilha quando a pergunta precisa de relacao estrutural e explicacao textual.",
        },
    },
]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    client = Neo4jClient()
    try:
        ok = client.ping()
    finally:
        client.close()
    return {"ok": ok}


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.stats()
    finally:
        client.close()


@app.get("/api/vector/status")
def vector_status() -> dict[str, Any]:
    stats = VectorStore().stats()
    stats["defaultEmbeddingModel"] = settings.ollama_embed_model
    return stats


@app.get("/api/models")
def models() -> dict[str, Any]:
    ollama = OllamaClient()
    try:
        return {
            "ok": True,
            "defaultModel": ollama.model,
            "models": ollama.list_models(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "defaultModel": ollama.model,
            "models": [{"name": ollama.model}],
            "error": str(exc),
        }


@app.get("/api/graph")
def graph(
    center: str | None = None,
    hops: int = Query(1, ge=1, le=4),
    limit: int = Query(160, ge=20, le=260),
) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.graph(center=center, hops=hops, limit=limit)
    finally:
        client.close()


@app.post("/api/vector-search")
def vector_search(payload: VectorSearchRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient())
        entities = rag.resolve_entities(payload.question)
        graph = (
            client.subgraph_for_seeds(entities, hops=payload.hops, limit=180)
            if payload.mode == "hybrid" and entities
            else {}
        )
        graph_names = [
            node["name"]
            for node in graph.get("nodes", [])
            if node.get("name") and node.get("name") not in entities
        ]
        if payload.source_type:
            docs = VectorStore().search(
                payload.question,
                OllamaClient(),
                model=settings.ollama_embed_model,
                limit=payload.top_k,
                seed_entities=entities if payload.mode == "hybrid" else [],
                graph_entities=graph_names if payload.mode == "hybrid" else [],
                source_type=payload.source_type,
                apply_boost=payload.mode == "hybrid",
            )
            for idx, doc in enumerate(docs, start=1):
                doc["sourceRank"] = idx
                doc["sourceBucket"] = doc.get("sourceType") or "other"
        else:
            docs = rag.retrieve_text(payload.question, entities, graph, payload.mode, limit=payload.top_k)
        return {
            "question": payload.question,
            "entities": entities,
            "graphEntities": graph_names,
            "documents": docs,
            "documentsBySource": rag.documents_by_source(docs),
            "retrieval": rag.retrieval_summary(docs),
            "embeddingModel": settings.ollama_embed_model,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        client.close()


def cypher_generation_messages(question: str) -> list[dict[str, str]]:
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
    return [
        {
            "role": "system",
            "content": (
                "Voce gera Cypher read-only para uma demo Neo4j de Senhor dos Aneis. "
                "Responda apenas JSON valido com chaves query, explanation e warnings. "
                "A query deve ser uma unica consulta, sem ponto e virgula, sem CREATE/MERGE/DELETE/SET/REMOVE/DROP, "
                "sem APOC e sem GDS. Sempre inclua LIMIT. Para perguntas visuais, prefira RETURN p ou RETURN a, r, b. "
                "Use apenas labels, propriedades e relacoes do schema fornecido."
            ),
        },
        {
            "role": "user",
            "content": f"/no_think\nSchema:\n{schema}\n\nExemplos:\n{examples}\n\nPergunta do usuario: {question}",
        },
    ]


def parse_cypher_generation(raw: str) -> dict[str, Any]:
    text = raw.strip()
    warnings: list[str] = []
    data: dict[str, Any] | None = None
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            candidate = json.loads(json_match.group(0))
            if isinstance(candidate, dict):
                data = candidate
        except json.JSONDecodeError:
            warnings.append("O modelo nao retornou JSON valido; extraindo Cypher do texto.")

    if data is None:
        code_match = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        query = code_match.group(1).strip() if code_match else extract_cypher_from_text(text)
        data = {
            "query": query,
            "explanation": "Cypher extraido da resposta do modelo.",
            "warnings": warnings,
        }

    query = str(data.get("query") or "").strip().removesuffix(";").strip()
    explanation = str(data.get("explanation") or "Query gerada a partir da pergunta.").strip()
    model_warnings = data.get("warnings") or []
    if isinstance(model_warnings, str):
        model_warnings = [model_warnings]
    warnings.extend(str(item) for item in model_warnings if item)
    return {"query": query, "explanation": explanation, "warnings": warnings}


def extract_cypher_from_text(text: str) -> str:
    match = re.search(r"\b(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|CALL\s+db\.)\b.*", text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else text.strip()


def cypher_synthesis_messages(payload: CypherSynthesizeRequest) -> list[dict[str, str]]:
    graph = payload.graph or {}
    nodes = (graph.get("nodes") or [])[:80]
    edges = (graph.get("edges") or [])[:120]
    rows = payload.rows[:20]
    structural_context = {
        "query": payload.query,
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
        "rows": rows,
        "counts": {"nodes": len(graph.get("nodes") or []), "edges": len(graph.get("edges") or []), "rows": len(payload.rows)},
    }
    return [
        {
            "role": "system",
            "content": (
                "Voce sintetiza somente evidencia estrutural de um grafo Neo4j de Senhor dos Aneis. "
                "Nao use conhecimento externo, chunks, livros, falas ou texto narrativo como evidencia. "
                "Use apenas nos, labels, propriedades, arestas, pesos, comunidades, paths e linhas tabulares fornecidas. "
                "Diferencie relacao direta, caminho, coocorrencia, comunidade e link predito. "
                "Se a query nao trouxe estrutura suficiente, diga isso diretamente. "
                "Responda em portugues brasileiro, curto e didatico."
            ),
        },
        {
            "role": "user",
            "content": (
                "/no_think\n"
                f"Pergunta: {payload.question}\n\n"
                "Contexto estrutural recuperado:\n"
                f"{json.dumps(structural_context, ensure_ascii=False)}"
            ),
        },
    ]


def graph_only_fallback(payload: CypherSynthesizeRequest, error: str | None = None) -> str:
    graph = payload.graph or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    prefix = f"O Ollama nao respondeu ({error}). " if error else ""
    if not nodes and not edges:
        return prefix + "A query nao trouxe estrutura renderizavel; use RETURN p ou RETURN a, r, b para sintetizar o grafo."
    edge_examples = "; ".join(
        f"{edge.get('sourceName')} -[{edge.get('type')}]-> {edge.get('targetName')}"
        for edge in edges[:6]
    )
    return (
        f"{prefix}Leitura Graph-only: a query recuperou {len(nodes)} nos e {len(edges)} arestas. "
        f"Amostra de relacoes: {edge_examples or 'sem arestas na amostra'}."
    )


@app.get("/api/cypher/examples")
def cypher_examples() -> dict[str, Any]:
    return {"examples": CYPHER_EXAMPLES}


@app.post("/api/cypher/run")
def cypher_run(payload: CypherRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        return client.run_readonly_cypher(payload.query, limit=payload.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Neo4jError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        client.close()


@app.post("/api/cypher/generate")
def cypher_generate(payload: CypherGenerateRequest) -> dict[str, Any]:
    ollama = OllamaClient(model=payload.model)
    raw = ollama.chat(cypher_generation_messages(payload.question), model=payload.model)
    parsed = parse_cypher_generation(raw)
    client = Neo4jClient()
    try:
        query = client.prepare_readonly_cypher(parsed["query"], limit=80)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Ollama gerou Cypher invalido: {exc}") from exc
    finally:
        client.close()
    return {
        "query": query,
        "explanation": parsed["explanation"],
        "warnings": parsed["warnings"],
        "model": payload.model or ollama.model,
    }


@app.post("/api/cypher/synthesize")
def cypher_synthesize(payload: CypherSynthesizeRequest) -> dict[str, Any]:
    ollama = OllamaClient(model=payload.model)
    try:
        answer = ollama.chat(cypher_synthesis_messages(payload), model=payload.model)
        status = "retrieval+ollama"
        if not answer.strip():
            answer = graph_only_fallback(payload)
            status = "fallback: resposta vazia do Ollama"
    except Exception as exc:
        answer = graph_only_fallback(payload, error=str(exc))
        status = f"fallback: {exc}"
    return {"answer": answer, "llmStatus": status, "model": payload.model or ollama.model}


@app.get("/api/lecture")
def lecture() -> dict[str, Any]:
    return {"steps": LECTURE_STEPS}


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.answer(
            payload.question,
            hops=payload.hops,
            top_k=payload.top_k,
            mode=payload.mode,
            model=payload.model,
            use_llm=payload.use_llm,
        )
    finally:
        client.close()


@app.post("/api/compare")
def compare(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.compare(
            payload.question,
            hops=payload.hops,
            top_k=payload.top_k,
            model=payload.model,
            use_llm=payload.use_llm,
        )
    finally:
        client.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
