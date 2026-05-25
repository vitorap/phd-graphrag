from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel, Field

from app.config import settings
from app.graphrag import GRAPH_RAG_STRATEGIES, GRAPH_RAG_STRATEGY_ORDER, GraphRAG
from app.llm_service import LLMService
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
    graph_rag_strategy: str = Field("kg_index", pattern="^(kg_index|vector_first|graph_filter|path|community|cypher)$")
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
        "question": "Por que a queda de Boromir e melhor explicada por texto do que por centralidade?",
        "center": "Boromir",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre a comparacao. A mesma pergunta separa motivacao narrativa, posicao estrutural e sintese hibrida.",
        "talkingPoints": [
            "RAG, Graph e GraphRAG recuperam evidencias de natureza diferente.",
            "A pergunta sobre Boromir exige conflito interno, pressao do Anel e papel na Fellowship.",
            "A resposta so faz sentido depois de auditar o contexto recuperado.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Recupera cenas de fala, tensao e arrependimento."},
            {"label": "Graph", "text": "Mostra Boromir na rede da Fellowship, mas nao explica motivacao."},
            {"label": "GraphRAG", "text": "Combina papel estrutural com trechos que sustentam a virada narrativa."},
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
            "question": "Se o grafo mostra Boromir bem conectado, o que ainda falta para explicar sua queda?",
            "options": [
                {"label": "Nada: centralidade ja explica motivacao.", "correct": False, "explanation": "Errado. Posicao estrutural nao revela conflito interno."},
                {"label": "Evidencia textual sobre intencao, fala e contexto narrativo.", "correct": True, "explanation": "Correto. Motivacao precisa de trechos, nao so arestas."},
                {"label": "Um embedding maior resolve sem contexto.", "correct": False, "explanation": "Errado. Embedding melhor nao substitui evidencia relevante."},
            ],
            "answer": "A resposta correta e a segunda: centralidade ajuda a localizar Boromir, mas motivacao exige texto.",
        },
    },
    {
        "id": "rag-good",
        "title": "RAG textual para pergunta narrativa",
        "duration": "5 min",
        "mode": "rag",
        "question": "Como Boromir justifica a tentacao de usar o Anel?",
        "center": "Boromir",
        "topK": 8,
        "demoAction": "Aplicar abre RAG puro. Observe top-k separado para livros e scripts, score cosine e ausencia de boost.",
        "talkingPoints": [
            "A pergunta vira embedding; cada chunk/fala tambem.",
            "O ranking e textual: cosine puro, sem k-hop ou PageRank.",
            "Esse modo e forte quando a resposta depende de fala, tom e justificativa local.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Melhor para recuperar a justificativa textual da tentacao."},
            {"label": "Graph", "text": "Mostra Boromir ligado a aliados e conflitos, mas nao o argumento dele."},
            {"label": "GraphRAG", "text": "Ajuda se quisermos ligar a fala ao papel dele na rede."},
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
            "question": "Para explicar a tentacao de Boromir, qual evidencia e mais forte?",
            "options": [
                {"label": "Um trecho em que ele racionaliza o uso do Anel.", "correct": True, "explanation": "Correto. A pergunta e narrativa e pede justificativa textual."},
                {"label": "Apenas o PageRank de Boromir.", "correct": False, "explanation": "PageRank indica centralidade, nao motivacao."},
                {"label": "Uma lista de todos os vizinhos dele.", "correct": False, "explanation": "Vizinhos ajudam contexto, mas nao bastam para explicar intencao."},
            ],
            "answer": "A melhor evidencia e textual: falas ou narracao que mostrem como Boromir racionaliza a tentacao.",
        },
    },
    {
        "id": "rag-limit",
        "title": "RAG textual para pergunta relacional",
        "duration": "5 min",
        "mode": "rag",
        "question": "Quem funciona como ponte entre Rohan, Gondor e a guerra contra Mordor?",
        "center": "Theoden",
        "topK": 8,
        "demoAction": "Aplicar mostra o limite do RAG puro quando a pergunta pede pontes entre comunidades politicas.",
        "talkingPoints": [
            "Perguntas sobre pontes pedem entidades intermediarias, nao apenas cenas parecidas.",
            "Um chunk pode falar de Rohan sem recuperar Gondor, ou recuperar Mordor sem explicar a ponte.",
            "O erro didatico seria deixar o LLM preencher a lacuna com conhecimento externo.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Pode recuperar cenas de batalha, mas espalhadas por faccoes."},
            {"label": "Graph", "text": "Pode listar personagens que conectam comunidades."},
            {"label": "GraphRAG", "text": "Usa conectores para buscar trechos que explicam por que a ponte importa."},
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
            "question": "Por que essa pergunta e ruim para RAG textual puro?",
            "options": [
                {"label": "Porque a resposta depende de intermediarios entre grupos.", "correct": True, "explanation": "Correto. Isso e uma propriedade estrutural, nao apenas semantica local."},
                {"label": "Porque RAG nao consegue recuperar nomes proprios.", "correct": False, "explanation": "Consegue, mas pode nao montar a ponte entre eles."},
                {"label": "Porque Mordor nao aparece no corpus.", "correct": False, "explanation": "A questao nao e ausencia de termo; e organizacao relacional."},
            ],
            "answer": "A dificuldade e estrutural: identificar pontes entre comunidades exige caminhos/conectores.",
        },
    },
    {
        "id": "graph-structural",
        "title": "Graph-only: caminhos e conectores",
        "duration": "6 min",
        "mode": "graph",
        "question": "Quais personagens conectam Rohan e Gondor no grafo?",
        "center": "Theoden",
        "hops": 2,
        "demoAction": "Aplicar abre o subgrafo em Theoden para discutir pontes entre reinos e guerra.",
        "talkingPoints": [
            "A evidencia primaria e simbolica: nos, arestas, pesos e tipos de relacao.",
            "O grafo responde bem a conectividade, centralidade e vizinhanca.",
            "O limite e que estrutura pode ser seca ou ambigua sem texto narrativo.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Pode citar Rohan ou Gondor, mas nao mede ponte."},
            {"label": "Graph", "text": "Mostra conectores, pesos, comunidades e saltos."},
            {"label": "GraphRAG", "text": "Usa os conectores para recuperar explicacao narrativa."},
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
            "question": "Quando o grafo responde melhor que texto nessa pergunta?",
            "options": [
                {"label": "Quando queremos medir ponte entre comunidades.", "correct": True, "explanation": "Correto. Conectividade e comunidade sao propriedades estruturais."},
                {"label": "Quando queremos uma citacao literal de Theoden.", "correct": False, "explanation": "Isso e melhor para RAG textual."},
                {"label": "Quando queremos que o LLM invente uma teoria politica.", "correct": False, "explanation": "A demo deve ficar ancorada em evidencias."},
            ],
            "answer": "Graph-only e forte quando a pergunta pede ponte, comunidade, caminho, grau ou centralidade.",
        },
    },
    {
        "id": "kg-index",
        "title": "GraphRAG: KG-as-Index",
        "duration": "5 min",
        "mode": "hybrid",
        "question": "Por que a marcha de Aragorn ao Portao Negro ajuda a missao longe dali?",
        "center": "Aragorn",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre GraphRAG. O subgrafo ativa Aragorn, Sauron, Gondor e aliados; o texto explica a distracao estrategica.",
        "talkingPoints": [
            "O KG funciona como andaime de recuperacao.",
            "Um chunk sobre estrategia pode ganhar reforco por estar conectado ao subgrafo de Aragorn e Sauron.",
            "A explicabilidade e topologica: o trecho apareceu porque esta ligado a uma decisao militar no grafo.",
        ],
        "methodContrasts": [
            {"label": "RAG", "text": "Pode achar a fala sobre distracao se os termos baterem."},
            {"label": "Graph", "text": "Mostra Aragorn, Sauron, Gondor e aliados como estrutura."},
            {"label": "GraphRAG", "text": "Liga a estrutura militar ao trecho que explica a estrategia."},
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
            "question": "Por que esse exemplo e melhor que uma pergunta direta A-B?",
            "options": [
                {"label": "Porque exige uma relacao indireta entre acao, alvo e efeito narrativo.", "correct": True, "explanation": "Correto. A resposta cruza estrategia, entidades e texto."},
                {"label": "Porque nao tem personagens nomeados.", "correct": False, "explanation": "Tem personagens; o ponto e a relacao indireta."},
                {"label": "Porque o grafo sozinho narra a cena.", "correct": False, "explanation": "O grafo mostra estrutura, mas o texto explica a estrategia."},
            ],
            "answer": "E um bom GraphRAG porque a relacao e indireta: uma acao militar cria condicoes para uma missao distante.",
        },
    },
    {
        "id": "architectures",
        "title": "Cinco arquiteturas GraphRAG",
        "duration": "7 min",
        "mode": "compare",
        "question": "Como auditar se Saruman e antagonista por fala, alianca ou posicao estrutural?",
        "center": "Saruman",
        "hops": 2,
        "topK": 6,
        "demoAction": "Aplicar abre Compare. Antes de rodar, discuta se a melhor evidencia vem de texto, query, subgrafo ou construcao do KG.",
        "talkingPoints": [
            "KG-as-Index recupera texto via topologia.",
            "Structural Context injeta triplas/caminhos linearizados.",
            "Text-to-Query gera Cypher/SPARQL; KG Builder cria o grafo; Hybrid funde rankings.",
        ],
        "methodContrasts": [
            {"label": "Structural Context", "text": "Bom para auditar aliancas e relacoes como triplas."},
            {"label": "Text-to-Query", "text": "Bom para perguntar diretamente por relacoes de Saruman."},
            {"label": "KG Builder", "text": "Bom para discutir como as relacoes de antagonismo seriam extraidas."},
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
        "question": "O que muda na vizinhanca de Gandalf quando saio de 1-hop para 3-hop?",
        "center": "Gandalf",
        "hops": 1,
        "demoAction": "Aplicar abre a vizinhanca de Gandalf. Altere hops e observe cobertura versus ruido.",
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
        "question": "A influencia de Gandalf vem de centralidade estrutural, autoridade narrativa ou das duas?",
        "center": "Gandalf",
        "hops": 2,
        "topK": 8,
        "demoAction": "Aplicar abre Compare. Antes de rodar, peca para a turma escolher qual coluna deve vencer.",
        "talkingPoints": [
            "Autoridade narrativa favorece RAG textual.",
            "Influencia estrutural favorece Graph-only.",
            "Perguntas que pedem as duas dimensoes favorecem GraphRAG hibrido.",
        ],
        "methodContrasts": [
            {"label": "Narrativa", "text": "Como Gandalf convence e orienta outros personagens? -> RAG tende a ajudar."},
            {"label": "Estrutura", "text": "Quem Gandalf conecta no grafo? -> Graph e o metodo natural."},
            {"label": "Relacao + explicacao", "text": "Por que ele e influente? -> GraphRAG tende a ser mais completo."},
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
                {"label": "Quem tem maior PageRank?", "correct": False, "explanation": "Graph-only provavelmente basta."},
                {"label": "O que Gandalf disse em uma cena especifica?", "correct": False, "explanation": "RAG textual provavelmente resolve melhor."},
                {"label": "Por que Gandalf e influente na rede e na narrativa?", "correct": True, "explanation": "Correto. Ela precisa de estrutura e texto ao mesmo tempo."},
            ],
            "answer": "GraphRAG brilha quando a pergunta pede influencia estrutural e interpretacao textual ao mesmo tempo.",
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
    llm = LLMService(model=payload.model)
    try:
        result = llm.generate_cypher(payload.question, model=payload.model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama/LangChain nao gerou Cypher valido: {exc}") from exc
    client = Neo4jClient()
    try:
        query = client.prepare_readonly_cypher(result.draft.query, limit=80)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Ollama gerou Cypher invalido: {exc}") from exc
    finally:
        client.close()
    return {
        "query": query,
        "explanation": result.draft.explanation,
        "warnings": result.draft.warnings,
        "model": payload.model or llm.model,
        "llmTrace": result.trace.model_dump(),
    }


@app.post("/api/cypher/synthesize")
def cypher_synthesize(payload: CypherSynthesizeRequest) -> dict[str, Any]:
    llm = LLMService(model=payload.model)
    structured_answer = None
    llm_trace = None
    try:
        result = llm.synthesize_cypher(
            question=payload.question,
            query=payload.query,
            rows=payload.rows,
            graph=payload.graph,
            model=payload.model,
        )
        answer = result.answer
        structured_answer = result.structured_answer.model_dump() if result.structured_answer else None
        llm_trace = result.trace.model_dump()
        status = result.status
        if not answer.strip():
            answer = graph_only_fallback(payload)
            status = "fallback: resposta vazia do Ollama"
    except Exception as exc:
        answer = graph_only_fallback(payload, error=str(exc))
        status = f"fallback: {exc}"
        llm_trace = {
            "provider": "ollama",
            "adapter": "langchain-ollama",
            "model": payload.model or llm.model,
            "template": "cypher_synthesize",
            "schemaName": "GroundedAnswer",
            "promptChars": 0,
            "estimatedTokens": 0,
            "attempts": 0,
            "status": "fallback",
            "error": str(exc),
            "preview": "",
        }
    return {
        "answer": answer,
        "structuredAnswer": structured_answer,
        "llmTrace": llm_trace,
        "llmStatus": status,
        "model": payload.model or llm.model,
    }


@app.get("/api/lecture")
def lecture() -> dict[str, Any]:
    return {"steps": LECTURE_STEPS}


@app.get("/api/graphrag/strategies")
def graphrag_strategies() -> dict[str, Any]:
    return {
        "default": "kg_index",
        "order": GRAPH_RAG_STRATEGY_ORDER,
        "strategies": [GRAPH_RAG_STRATEGIES[key] for key in GRAPH_RAG_STRATEGY_ORDER],
    }


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
            graph_rag_strategy=payload.graph_rag_strategy,
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
            graph_rag_strategy=payload.graph_rag_strategy,
        )
    finally:
        client.close()


@app.post("/api/graphrag/compare")
def compare_graphrag_strategies(payload: AskRequest) -> dict[str, Any]:
    client = Neo4jClient()
    try:
        rag = GraphRAG(client, OllamaClient(model=payload.model))
        return rag.compare_graphrag_strategies(
            payload.question,
            hops=payload.hops,
            top_k=payload.top_k,
            model=payload.model,
        )
    finally:
        client.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
