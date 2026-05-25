from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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


CYPHER_EXAMPLES = [
    {
        "id": "frodo-neighbors",
        "title": "1-hop: vizinhos de Frodo",
        "explain": "Mostra quem envia informacao diretamente para Frodo no grafo.",
        "lesson": "Use este exemplo para explicar que 1-hop e o campo receptivo imediato: cada vizinho pode contribuir uma mensagem para atualizar a representacao de Frodo.",
        "gnn": "Message passing camada 1: agrega atributos e relacoes dos vizinhos diretos.",
        "visual": {
            "center": "Frodo",
            "hops": 1,
            "caption": "O grafo foca Frodo e somente as entidades conectadas diretamente.",
        },
        "query": """
MATCH (:Entity {name: 'Frodo'})-[r]-(n:Entity)
RETURN n.name AS entidade, labels(n) AS labels, type(r) AS relacao,
       coalesce(r.weight, r.confidence, 1) AS peso
ORDER BY peso DESC, entidade
LIMIT 20
""".strip(),
    },
    {
        "id": "frodo-sauron-path",
        "title": "Menor caminho Frodo-Sauron",
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
RETURN [n IN nodes(p) | n.name] AS caminho,
       [r IN relationships(p) | type(r)] AS relacoes,
       length(p) AS saltos
""".strip(),
    },
    {
        "id": "frodo-sauron-bridges",
        "title": "Pontes em 2-hop",
        "explain": "Lista personagens e conceitos que conectam Frodo e Sauron em dois saltos.",
        "lesson": "Bom para mostrar que o grafo encontra candidatos de explicacao antes do texto: Anel, Mordor, inimigos, coocorrencias ou outros conectores.",
        "gnn": "2-hop corresponde a duas rodadas de propagacao; aumenta cobertura, mas tambem pode trazer ruido.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "O grafo mostra o campo receptivo 2-hop que gera os conectores.",
        },
        "query": """
MATCH (f:Entity {name: 'Frodo'})-[r1]-(m:Entity)-[r2]-(s:Entity {name: 'Sauron'})
WHERE m.name <> 'Frodo' AND m.name <> 'Sauron'
RETURN m.name AS ponte, labels(m) AS labels,
       type(r1) AS relacao_com_frodo,
       type(r2) AS relacao_com_sauron,
       coalesce(r1.weight, r1.confidence, 1) + coalesce(r2.weight, r2.confidence, 1) AS forca
ORDER BY forca DESC, ponte
LIMIT 20
""".strip(),
    },
    {
        "id": "shared-documents",
        "title": "Chunks que citam Frodo e Sauron",
        "explain": "Conecta a estrutura do grafo com unidades recuperaveis pelo RAG.",
        "lesson": "Aqui a ponte com GraphRAG fica explicita: o grafo localiza entidades, mas a explicacao final vem dos documentos que mencionam essas entidades.",
        "gnn": "Depois de propagar no grafo, a recuperacao textual usa os nos ativados como sinal de contexto.",
        "visual": {
            "center": "Sauron",
            "hops": 1,
            "caption": "A visualizacao foca Sauron enquanto a tabela mostra evidencias textuais compartilhadas.",
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
    {
        "id": "pagerank",
        "title": "Top PageRank",
        "explain": "Centralidade estrutural calculada no grafo personagem-personagem.",
        "lesson": "PageRank ajuda a discutir quais personagens sao estruturalmente influentes, nao necessariamente os mais importantes narrativamente.",
        "gnn": "Centralidade e um resumo global; GNNs aprendem representacoes que podem incorporar sinais estruturais semelhantes.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "A visualizacao em Frodo ajuda a comparar PageRank local com ranking global.",
        },
        "query": """
MATCH (c:Character)
RETURN c.name AS personagem, c.race AS raca,
       round(coalesce(c.pagerank, 0), 5) AS pagerank,
       round(coalesce(c.weightedDegree, 0), 1) AS grau_ponderado,
       c.community AS comunidade
ORDER BY pagerank DESC
LIMIT 15
""".strip(),
    },
    {
        "id": "relationship-types",
        "title": "Tipos de relacao",
        "explain": "Ajuda a explicar que o grafo mistura ontologia, coocorrencia, falas e texto.",
        "lesson": "Esta query e boa para mostrar que o KG nao e uma rede social simples: ha arestas semanticas, textuais, estruturais e preditas.",
        "gnn": "Em grafos heterogeneos, o tipo da aresta muda a mensagem. Isso abre a conversa sobre relational GNNs.",
        "visual": {
            "center": "Gandalf",
            "hops": 1,
            "caption": "O grafo em Gandalf tende a exibir diversidade de relacoes em um personagem central.",
        },
        "query": """
MATCH ()-[r]->()
RETURN type(r) AS relacao, count(r) AS total
ORDER BY total DESC
LIMIT 20
""".strip(),
    },
    {
        "id": "strong-cooccurrence",
        "title": "Coocorrencias fortes",
        "explain": "Mostra as arestas ponderadas usadas como estrutura social/narrativa.",
        "lesson": "Coocorrencia e uma forma barata de inferir proximidade narrativa. Ela e util, mas nao significa amizade ou causalidade.",
        "gnn": "O peso da aresta pode modular a forca da mensagem entre dois personagens.",
        "visual": {
            "center": "Samwise",
            "hops": 1,
            "caption": "Samwise aparece como vizinho forte de Frodo e ajuda a discutir peso de aresta.",
        },
        "query": """
MATCH (a:Character)-[r:CO_OCCURS_WITH]-(b:Character)
WHERE a.name < b.name
RETURN a.name AS origem, b.name AS destino, r.weight AS peso
ORDER BY peso DESC
LIMIT 20
""".strip(),
    },
    {
        "id": "frodo-community",
        "title": "Comunidade de Frodo",
        "explain": "Lista personagens na mesma comunidade estrutural de Frodo.",
        "lesson": "Comunidades agrupam personagens que compartilham vizinhancas parecidas. E uma forma visual de falar de representacao de grupo.",
        "gnn": "Embeddings de nos tendem a aproximar entidades com contexto estrutural parecido.",
        "visual": {
            "center": "Frodo",
            "hops": 2,
            "caption": "A vizinhanca 2-hop de Frodo mostra muitos membros da mesma comunidade.",
        },
        "query": """
MATCH (f:Character {name: 'Frodo'})
MATCH (c:Character)
WHERE c.community = f.community
RETURN c.name AS personagem,
       c.race AS raca,
       round(coalesce(c.pagerank, 0), 5) AS pagerank,
       round(coalesce(c.weightedDegree, 0), 1) AS grau_ponderado,
       c.community AS comunidade
ORDER BY pagerank DESC
LIMIT 20
""".strip(),
    },
    {
        "id": "predicted-links",
        "title": "Links preditos",
        "explain": "Mostra arestas inferidas como camada separada de hipoteses.",
        "lesson": "Use para separar evidencia observada de hipotese predita. Em aula, isso abre a conversa sobre link prediction.",
        "gnn": "Link prediction e uma tarefa classica de GNN: aprender embeddings para estimar arestas ausentes.",
        "visual": {
            "center": "Sauron",
            "hops": 2,
            "caption": "A visualizacao em Sauron destaca onde links preditos podem aparecer como hipoteses.",
        },
        "query": """
MATCH (a:Character)-[r:PREDICTED_LINK]->(b:Character)
RETURN a.name AS origem,
       b.name AS destino,
       round(coalesce(r.confidence, 0), 3) AS confianca,
       r.method AS metodo
ORDER BY confianca DESC, origem, destino
LIMIT 20
""".strip(),
    },
    {
        "id": "similar-chapters",
        "title": "Capitulos similares",
        "explain": "Mostra similaridade estrutural entre capitulos dos livros.",
        "lesson": "Este exemplo amplia o grafo alem de personagens: capitulos tambem podem formar uma rede para recuperar contexto narrativo.",
        "gnn": "Grafos de documentos permitem propagar sinal entre trechos parecidos antes da etapa de RAG.",
        "visual": {
            "center": "Frodo",
            "hops": 1,
            "caption": "A tabela e documental; o grafo permanece em Frodo para manter a referencia narrativa.",
        },
        "query": """
MATCH (a:Chapter)-[r:SIMILAR_CHAPTER]->(b:Chapter)
RETURN a.bookTitle AS livro_a,
       a.title AS capitulo_a,
       b.bookTitle AS livro_b,
       b.title AS capitulo_b,
       round(coalesce(r.weight, 0), 3) AS peso
ORDER BY peso DESC
LIMIT 20
""".strip(),
    },
]


LECTURE_STEPS = [
    {
        "id": "opening",
        "title": "Abertura: a mesma pergunta, tres maquinas",
        "duration": "3 min",
        "mode": "overview",
        "question": "Qual a relacao de Frodo com Sauron?",
        "center": "Frodo",
        "hops": 2,
        "demoAction": "Mostre os numeros do corpus e anuncie que a mesma pergunta sera respondida por texto, grafo e GraphRAG.",
        "talkingPoints": [
            "O corpus mistura texto completo, scripts, ontologia e redes de personagens.",
            "A pergunta Frodo-Sauron e boa porque exige narrativa textual e estrutura relacional.",
            "A aula compara tres formas de recuperar contexto, nao tres modelos magicos.",
        ],
        "speakerNotes": [
            "Comece pela intuicao: texto lembra cenas; grafo lembra conexoes.",
            "Avise que a demo e deterministica e roda localmente com Neo4j e Ollama.",
        ],
        "quiz": {
            "question": "Por que essa pergunta nao e trivial para RAG puro?",
            "answer": "Porque ela mistura entidades, relacao estrutural e explicacao narrativa sobre Anel, Mordor e antagonismo.",
        },
    },
    {
        "id": "corpus",
        "title": "Corpus: de texto bruto para grafo navegavel",
        "duration": "4 min",
        "mode": "overview",
        "question": "Que fontes alimentam o grafo?",
        "center": "Frodo",
        "hops": 1,
        "demoAction": "Passe pelos contadores: entidades, relacoes, fontes, unidades RAG, chunks de livro e falas de script.",
        "talkingPoints": [
            "Livros viram TextChunks; scripts viram DialogueLines.",
            "Personagens detectados em texto criam arestas MENTIONS.",
            "Ontologia e redes sociais completam o KG com relacoes semanticas e ponderadas.",
        ],
        "speakerNotes": [
            "Use os tooltips dos contadores para explicar a diferenca entre unidades RAG, chunks e falas.",
            "Deixe claro que o grafo nao substitui o texto: ele indexa e organiza o texto.",
        ],
        "quiz": {
            "question": "Por que falas de script tambem entram como unidades RAG?",
            "answer": "Porque cada fala e uma evidencia textual pequena, com speaker e entidades mencionadas.",
        },
    },
    {
        "id": "rag",
        "title": "RAG vetorial: similaridade sem estrutura explicita",
        "duration": "5 min",
        "mode": "rag",
        "question": "Por que o Anel importa para Frodo?",
        "topK": 8,
        "demoAction": "Rode Buscar Evidencias e mostre ranking, cosine score, fonte e mencoes detectadas.",
        "talkingPoints": [
            "Chunks e falas viram vetores de embedding.",
            "A pergunta tambem vira vetor; a busca usa similaridade coseno.",
            "RAG encontra passagens boas, mas nao sabe caminhos estruturais por si so.",
        ],
        "speakerNotes": [
            "Aponte que o RAG e otimo para explicar narrativas e citar trechos.",
            "Mostre que o score vetorial nao diz qual relacao existe no KG.",
        ],
        "quiz": {
            "question": "O que muda entre BM25 e embedding retrieval?",
            "answer": "BM25 depende de termos; embedding retrieval aproxima significado no espaco vetorial.",
        },
    },
    {
        "id": "rag-failure",
        "title": "Limite do RAG: texto certo, estrutura implicita",
        "duration": "4 min",
        "mode": "rag",
        "question": "Quem sao os conectores entre Frodo e Mordor?",
        "topK": 8,
        "demoAction": "Mostre que a busca textual acha trechos, mas a nocao de conector fica espalhada e pouco auditavel.",
        "talkingPoints": [
            "Uma pergunta relacional pede entidades intermediarias, nao apenas trechos parecidos.",
            "RAG puro pode responder bem, mas e dificil auditar o caminho usado.",
            "Essa lacuna motiva abrir o grafo explicitamente.",
        ],
        "speakerNotes": [
            "Nao venda RAG como ruim; venda como incompleto para perguntas estruturais.",
            "Prepare a transicao para k-hop e caminhos.",
        ],
        "quiz": {
            "question": "Que evidencia falta quando so temos chunks?",
            "answer": "Falta uma estrutura explicita de caminhos, vizinhos, pesos e tipos de relacao.",
        },
    },
    {
        "id": "graph",
        "title": "Graph: vizinhanca, caminhos e centralidade",
        "duration": "6 min",
        "mode": "graph",
        "question": "Qual o caminho estrutural entre Frodo e Sauron?",
        "center": "Frodo",
        "hops": 2,
        "cypherExample": "frodo-sauron-path",
        "demoAction": "Selecione Menor caminho Frodo-Sauron, rode Cypher e mostre a tabela junto com o subgrafo.",
        "talkingPoints": [
            "k-hop define o campo receptivo, como em message passing.",
            "PageRank e comunidades resumem posicao estrutural.",
            "O grafo explica conexoes, mas nao substitui a narrativa textual.",
        ],
        "speakerNotes": [
            "Explique que cada query e uma pergunta operacional sobre a estrutura.",
            "Mostre como trocar hops muda cobertura e ruido.",
        ],
        "quiz": {
            "question": "Por que aumentar hops pode piorar a resposta?",
            "answer": "Mais hops aumentam cobertura, mas tambem trazem ruido e entidades pouco relevantes.",
        },
    },
    {
        "id": "message-passing",
        "title": "Ligacao com GNN: hops como campo receptivo",
        "duration": "5 min",
        "mode": "graph",
        "question": "O que muda quando aumento hops de 1 para 3?",
        "center": "Frodo",
        "hops": 1,
        "cypherExample": "frodo-neighbors",
        "demoAction": "Comece em 1-hop, depois altere para 2 ou 3 hops para mostrar cobertura versus ruido.",
        "talkingPoints": [
            "Em uma GNN, cada camada agrega informacao de vizinhos.",
            "1-hop e preciso, mas limitado; 3-hop cobre mais, mas dilui o sinal.",
            "GraphRAG usa a mesma intuicao para decidir quais entidades ativam contexto textual.",
        ],
        "speakerNotes": [
            "Use a palavra campo receptivo e conecte com CNNs se a turma conhecer.",
            "Destaque que aqui nao estamos treinando uma GNN; estamos usando a intuicao estrutural dela.",
        ],
        "quiz": {
            "question": "Qual analogia direta entre k-hop e GNN?",
            "answer": "k-hop representa quantas rodadas de vizinhos podem contribuir informacao para o no alvo.",
        },
    },
    {
        "id": "graphrag",
        "title": "GraphRAG: estrutura guia a recuperacao textual",
        "duration": "7 min",
        "mode": "hybrid",
        "question": "O que Frodo e de Sauron?",
        "center": "Frodo",
        "hops": 2,
        "topK": 8,
        "demoAction": "Execute GraphRAG e leia a esteira: entidades, k-hop, vetores e resposta.",
        "talkingPoints": [
            "Primeiro detectamos entidades.",
            "Depois expandimos k-hop no grafo.",
            "Por fim, buscamos evidencias vetoriais com boost do subgrafo.",
        ],
        "speakerNotes": [
            "A frase central: o grafo acha a estrutura; o texto explica a narrativa.",
            "Mostre entidades detectadas e evidencias antes de ler a resposta.",
        ],
        "quiz": {
            "question": "Onde a ideia de GNN aparece no GraphRAG?",
            "answer": "Na expansao k-hop: informacao de vizinhos e caminhos vira contexto para a resposta.",
        },
    },
    {
        "id": "compare",
        "title": "Comparacao final: RAG vs Graph vs GraphRAG",
        "duration": "4 min",
        "mode": "compare",
        "question": "Qual a relacao de Frodo com Sauron?",
        "center": "Frodo",
        "hops": 2,
        "topK": 8,
        "demoAction": "Rode Comparar Agora e leia cada coluna como uma tese diferente sobre recuperacao.",
        "talkingPoints": [
            "RAG textual recupera narrativa.",
            "Graph recupera estrutura.",
            "GraphRAG combina as duas evidencias.",
        ],
        "speakerNotes": [
            "A comparacao e o fechamento tecnico: nao e sobre vencer sempre, e sobre escolher retrieval para a pergunta.",
            "Use a coluna GraphRAG para conectar a aula com o objetivo de GNN.",
        ],
        "quiz": {
            "question": "Qual modo voce usaria para pergunta causal/narrativa com entidades conhecidas?",
            "answer": "GraphRAG, porque usa o grafo para selecionar contexto e o texto para explicar.",
        },
    },
    {
        "id": "closing",
        "title": "Takeaways: quando usar cada abordagem",
        "duration": "2 min",
        "mode": "compare",
        "question": "Qual a relacao de Frodo com Sauron?",
        "center": "Frodo",
        "hops": 2,
        "demoAction": "Feche com a regra pratica: RAG para narrativa, Graph para estrutura, GraphRAG para perguntas relacionais com explicacao textual.",
        "talkingPoints": [
            "RAG responde com texto, mas a estrutura fica implicita.",
            "Graph explica conexoes, mas pode faltar narrativa.",
            "GraphRAG e um compromisso auditavel para perguntas relacionais.",
        ],
        "speakerNotes": [
            "Termine lembrando que o mesmo padrao vale fora de LOTR: biomedicina, juridico, empresas e ciencia.",
            "Convide a turma a pensar qual seria o grafo do proprio dominio.",
        ],
        "quiz": {
            "question": "Qual e a regra de bolso final?",
            "answer": "Use o grafo para orientar o que procurar e use o texto para explicar com evidencia.",
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
        docs = VectorStore().search(
            payload.question,
            OllamaClient(),
            model=settings.ollama_embed_model,
            limit=payload.top_k,
            seed_entities=entities,
            graph_entities=graph_names,
            source_type=payload.source_type,
        )
        return {
            "question": payload.question,
            "entities": entities,
            "graphEntities": graph_names,
            "documents": docs,
            "embeddingModel": settings.ollama_embed_model,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        client.close()


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
    finally:
        client.close()


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
