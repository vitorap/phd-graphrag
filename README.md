# GraphRAG em Middle-earth

Demo pratica para uma aula/seminario de GNN, Knowledge Graphs e LLMs. A ideia e mostrar, em uma unica aplicacao local com Docker, como RAG vetorial por embeddings, Graph retrieval e GraphRAG hibrido se comportam em perguntas sobre Senhor dos Aneis, conectando chunks narrativos, subgrafos k-hop e intuicoes de GNN/message passing.

## Objetivo da Aula

O seminario parte de uma pergunta simples:

> O que muda quando o contexto entregue ao LLM deixa de ser apenas texto solto e passa a ser um subgrafo estruturado?

A demo usa quatro fontes complementares:

- **Raphtory LOTR interaction graph**: grafo de coocorrencia de personagens em sentencas da trilogia. Ele e melhor para visualizacao, centralidade, comunidades, k-hop neighborhood e conexao com GNN.
- **LOTRO OWL ontology**: ontologia RDF/OWL com classes, personagens, lugares, armas, linguas e relacoes semanticas como `friendOf`, `enemyOf`, `hasWeapon`, `inhabitant` e `speaks`.
- **SNA_LOTR**: livros limpos, scripts dos filmes, personagens, redes ponderadas, links por capitulo, sentimento e arquivos de link prediction.
- **Corpus textual local**: chunks dos livros e falas dos filmes ligados a entidades por `MENTIONS`.

No dashboard, **Fontes** sao os arquivos/obras de origem: 3 livros e 3 filmes. **Unidades RAG** sao as passagens indexadas e recuperaveis por embedding: `TextChunk` dos livros + `DialogueLine` dos scripts. Portanto, neste projeto `RetrievalDocument` e uma superclasse tecnica para "coisa recuperavel pelo RAG", nao um documento fonte/arquivo.

O resultado e um grafo hibrido e textual:

- `INTERACTS_WITH`: backbone narrativo vindo do dataset da Raphtory.
- `CO_OCCURS_WITH`: rede ponderada do SNA_LOTR.
- `PREDICTED_LINK`: links candidatos vindos de metricas de link prediction.
- `TextChunk` e `DialogueLine`: evidencias textuais para RAG.
- `MENTIONS`, `SPEAKS_LINE`, `IN_BOOK`, `IN_MOVIE`, `IN_CHAPTER`: ligam texto, personagens e fontes.
- relacoes semanticas: camada de conhecimento vinda da ontologia LOTRO.
- atributos/metricas: raca, genero, sentimento, word count, grau ponderado, PageRank e comunidade.

## Stack

- Docker Compose
- Neo4j Community
- FastAPI
- JavaScript/SVG nativo para visualizacao
- Ollama local no host para gerar respostas e embeddings
- Vector store local persistido em `data/vector_store/` com embeddings normalizados (`.npz` + metadata JSON)
- Python para ingestao, metricas e retrieval

Modelos Ollama detectados nesta maquina:

- `qwen3.6:latest` usado como default, com `context length 262144` reportado por `ollama show`
- `gemma4:26b` como alternativa
- `lfm2:latest` como alternativa
- `nomic-embed-text:latest` como default para embeddings do RAG vetorial

A UI consulta `/api/tags` do Ollama local no host e preenche o seletor **Modelo LLM** automaticamente. A chave **Sintese com LLM** decide se `/api/ask` e `/api/compare` ficam em retrieval-only ou usam Ollama para sintetizar as evidencias.

## Como Rodar

```bash
make help
make up
make data
make seed
make ollama-pull-embed
make vectors
make compare Q="Qual a relação de Frodo com Sauron?"
make ollama-warm
make app
```

URLs:

- App: http://localhost:8000
- Neo4j Browser: http://localhost:7474
- Bolt: `bolt://localhost:7687`

Credenciais Neo4j default:

- usuario: `neo4j`
- senha: `graphrag-lotr`

## Comandos Principais

```bash
make help         # lista comandos de apresentacao
make up           # sobe Neo4j e app
make data         # baixa datasets para data/raw
make seed         # importa e enriquece o grafo no Neo4j
make vectors      # gera embeddings e indice vetorial local
make stats        # mostra estatisticas do grafo
make ask Q="Como Frodo se conecta a Sauron?" MODE=hybrid
make ask Q="Como Frodo se conecta a Sauron?" MODE=hybrid STRATEGY=path
make ask Q="Como Frodo se conecta a Sauron?" MODE=rag
make ask Q="Como Frodo se conecta a Sauron?" MODE=graph
make compare Q="Qual a relação de Frodo com Sauron?"
make smoke-vectors
make ollama-pull-embed
make ollama-show  # mostra metadados do modelo local
make ollama-warm  # preaquece o modelo local antes da apresentacao
make logs         # acompanha logs
make reset        # remove containers e volume do Neo4j
```

## Perguntas Boas para a Demo

```bash
make ask Q="Como Frodo se conecta a Sauron?"
make compare Q="Qual a relação de Frodo com Sauron?"
make ask Q="Quais personagens sao mais centrais na rede?"
make ask Q="Que relacoes aproximam Frodo, Gandalf e Aragorn?"
make ask Q="O que muda quando aumento a vizinhanca para 3 saltos?"
make ask Q="Quais personagens conectam hobbits, elfos e homens?"
```

## Roteiro de 40 Minutos

1. **Motivacao: RAG vs GraphRAG (5 min)**
   - RAG tradicional recupera chunks.
   - GraphRAG recupera entidades, relacoes, caminhos e comunidades.
   - A diferenca pratica e que o contexto passa a ter estrutura.

2. **Modelo de Grafo + Texto (6 min)**
   - Mostrar `Character`, `Weapon`, `Place`, `Language`, `Race`, `TextChunk`, `DialogueLine`.
   - Mostrar `INTERACTS_WITH` e `CO_OCCURS_WITH` como backbone.
   - Mostrar relacoes semanticas da ontologia.
   - Mostrar `MENTIONS` como ponte entre texto e grafo.

3. **Visualizacao no Neo4j (7 min)**
   - Abrir Neo4j Browser.
   - Rodar consultas Cypher simples.
   - Mostrar vizinhanca de Frodo, Gandalf, Sauron.

4. **Comparacao RAG vs Graph vs GraphRAG (12 min)**
   - Rodar `make compare Q="Qual a relação de Frodo com Sauron?"`.
   - Mostrar RAG vetorial: bom para narrativa, sem estrutura explicita.
   - Mostrar Graph: bom para caminho/vizinhanca, mas pobre em explicacao.
   - Mostrar GraphRAG: entidades + subgrafo + chunks ligados ao subgrafo.
   - Na aba GraphRAG, alternar estrategias: `KG-as-Index`, `Vector-first`, `Graph filter`, `Paths`, `Community` e `Symbolic Cypher`.
   - Alterar `hops=1`, `hops=2`, `hops=3`.
   - Ligar a chave `Sintese com LLM` apenas quando o modelo ja estiver preaquecido.

5. **Conexao com GNN (8 min)**
   - `k-hop neighborhood` como campo receptivo.
   - Agregacao de vizinhos como analogia a message passing.
   - PageRank/comunidades como features estruturais.
   - Por que GraphRAG e GNN atacam problemas parecidos por mecanismos diferentes.

6. **Limitacoes e Extensoes (2 min)**
   - Coocorrencia nao e causalidade.
   - Ontologia e pequena, mas semanticamente rica.
   - Proximo passo: GDS, node classification, link prediction e ranking de subgrafos.

## Consultas Cypher para Mostrar

Vizinhanca de Frodo:

```cypher
MATCH p = (:Entity {name: "Frodo"})-[*1..2]-()
RETURN p
LIMIT 80;
```

Top personagens por PageRank:

```cypher
MATCH (c:Character)
RETURN c.name, c.race, c.pagerank, c.weightedDegree, c.community
ORDER BY c.pagerank DESC
LIMIT 15;
```

Caminhos entre Frodo e Sauron:

```cypher
MATCH p = shortestPath(
  (:Entity {name: "Frodo"})-[*..4]-(:Entity {name: "Sauron"})
)
RETURN p;
```

Relacoes semanticas da ontologia:

```cypher
MATCH p = (:Entity {name: "Frodo"})-[:FRIEND_OF|ENEMY_OF|HAS_WEAPON|SPEAKS]-()
RETURN p;
```

Chunks que mencionam Frodo e Sauron:

```cypher
MATCH (d:RetrievalDocument)-[:MENTIONS]->(:Entity {name: "Frodo"})
MATCH (d)-[:MENTIONS]->(:Entity {name: "Sauron"})
RETURN d.sourceTitle, d.chapterTitle, left(d.text, 220) AS trecho
LIMIT 10;
```

## Decisoes de Projeto

- **Neo4j em vez de RDF store puro**: facilita visualizacao, Cypher e demo em sala.
- **GraphRAG customizado em vez de framework pesado**: mais controlavel e explicavel para uma aula.
- **Dataset hibrido e rico**: Raphtory da densidade estrutural; LOTRO OWL da semantica; SNA_LOTR da corpus textual, rede ponderada e features.
- **Ollama local**: evita dependencia de API externa.
- **Docker-first**: parceiro consegue rodar com os mesmos comandos.
- **RAG vetorial local**: embeddings gerados no Ollama com `nomic-embed-text:latest` e cosine similarity em um indice persistido simples.
- **BM25 como fallback**: se o indice vetorial ainda nao existir, a demo ainda recupera evidencias textuais sem travar.

## Estrategias GraphRAG Implementadas

A aba **GraphRAG** nao trata GraphRAG como uma tecnica unica. Ela permite selecionar e comparar seis familias:

- `kg_index`: entidades da pergunta abrem subgrafo k-hop; o subgrafo da boost no ranking vetorial.
- `vector_first`: busca vetorial pura primeiro; entidades mencionadas nos chunks recuperados expandem o grafo depois.
- `graph_filter`: subgrafo vira filtro duro dentro da busca vetorial; so entram documentos ligados por `MENTIONS`.
- `path`: exige pelo menos duas entidades; caminhos curtos e conectores 2-hop recebem peso extra no reranking.
- `community`: comunidade estrutural dos personagens aproxima a ideia local-to-global e marca fallback quando nao ha comunidade.
- `cypher`: template Cypher simbolico e read-only busca documentos por `MENTIONS`; a geracao livre de Cypher fica na aba Graph.

Para auditar se as variantes nao colapsaram no mesmo codigo:

```bash
make smoke-strategies
```

Referencias usadas na UI: Microsoft GraphRAG Local/Global Search, From Local to Global GraphRAG, GRAG, KG2RAG, LightRAG, HippoRAG e GNN-RAG.

## Perguntas Centrais para Fechar com o Professor/Turma

- Quando uma vizinhanca k-hop e suficiente, e quando precisamos de busca global/comunidades?
- Em que tipo de pergunta o grafo ajuda mais que um RAG vetorial?
- Quando o texto sozinho explica melhor que o grafo?
- Qual e o limite de usar coocorrencia como relacao?
- Como features semanticas de KG poderiam alimentar uma GNN?
- Como uma GNN poderia ranquear melhores subgrafos para GraphRAG?

## Troubleshooting

O Ollama roda no host, nao em container. Se o app subir mas o LLM nao responder, verifique se o Ollama esta rodando localmente:

```bash
ollama list
```

O container usa por padrao:

```text
http://host.docker.internal:11434
```

Antes da aula, preaqueca o modelo:

```bash
make ollama-warm
```

Se estiver rodando tudo fora do Docker, use:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

Se quiser trocar o modelo:

```bash
make ask Q="Como Frodo se conecta a Sauron?" MODEL=gemma4:26b
```

Na UI, use o seletor **Modelo LLM**. Ele lista os modelos disponiveis no Ollama local da maquina.

Para o RAG vetorial, baixe o modelo de embedding e gere o indice:

```bash
make ollama-pull-embed EMBED_MODEL=nomic-embed-text:latest
make vectors EMBED_MODEL=nomic-embed-text:latest
```

O endpoint `/api/vector/status` mostra se o indice esta pronto. O endpoint `/api/vector-search` expõe a busca vetorial usada no playground.

Se quiser explorar a janela longa do `qwen3.6:latest`, aumente `NUM_CTX`. Para a demo, o default e propositalmente menor para reduzir latencia:

```bash
make ask Q="Como Frodo se conecta a Sauron?" NUM_CTX=262144 TIMEOUT=180
```
