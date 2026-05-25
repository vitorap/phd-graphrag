# GraphRAG em Middle-earth

Aplicacao local para demonstrar **RAG vetorial**, **Graph retrieval** e
**GraphRAG hibrido** usando dados de Senhor dos Aneis. O projeto foi feito para
uma aula/seminario de GNN, Knowledge Graphs e LLMs, mas qualquer pessoa pode
clonar o repositorio e rodar em sua propria maquina com Docker + Ollama.

A demo sobe:

- um Neo4j com grafo de personagens, entidades, livros, scripts e chunks;
- uma API FastAPI;
- uma UI web com abas Overview, RAG, Graph, GraphRAG, Compare e Flow;
- um indice vetorial local em `data/vector_store/`, gerado com embeddings do
  Ollama;
- consultas CLI via `make ask` e `make compare`.

URLs locais:

- App: http://localhost:8000
- Neo4j Browser: http://localhost:7474
- Neo4j Bolt: `bolt://localhost:7687`

Credenciais Neo4j:

- usuario: `neo4j`
- senha: `graphrag-lotr`

## Pre-requisitos

Instale antes de rodar:

1. **Git**
   - Para clonar o repositorio.

2. **Docker**
   - Docker Desktop no macOS/Windows, ou Docker Engine + Docker Compose v2 no
     Linux.
   - Verifique:

```bash
docker --version
docker compose version
```

3. **Ollama rodando no host**
   - O app nao sobe um container de Ollama por padrao.
   - O container acessa o Ollama da sua maquina por
     `http://host.docker.internal:11434`.
   - Verifique:

```bash
ollama list
```

4. **Modelo de embedding**
   - Default: `nomic-embed-text:latest`.
   - O comando `make bootstrap` tenta baixar esse modelo automaticamente.
   - Para baixar manualmente:

```bash
ollama pull nomic-embed-text:latest
```

5. **Modelo LLM opcional**
   - Default do projeto: `qwen3.6:latest`.
   - Se voce nao tiver esse modelo, use outro modelo local no app ou nos comandos:

```bash
make ask Q="How is Frodo connected to Sauron?" MODEL=llama3.1:8b
```

O projeto funciona em modo **retrieval-only** mesmo sem sintese por LLM. O
Ollama e essencial para gerar embeddings vetoriais com `make vectors`.

## Setup Rapido

Depois de clonar:

```bash
git clone <URL_DO_REPOSITORIO>
cd phd-graphrag
make bootstrap
make app
```

Abra:

```text
http://localhost:8000
```

O `make bootstrap` faz, em ordem:

1. baixa o modelo de embedding definido por `EMBED_MODEL`;
2. sobe Neo4j + app com Docker Compose;
3. garante os dados brutos em `data/raw/`;
4. importa o grafo no Neo4j;
5. gera o indice vetorial local em `data/vector_store/`;
6. imprime estatisticas do grafo.

Os dados brutos principais ja estao versionados em `data/raw/`, entao `make data`
normalmente apenas valida que os arquivos existem. Se algum arquivo estiver
faltando, ele tenta baixar novamente das fontes originais.

## Setup Passo a Passo

Use este fluxo se quiser acompanhar cada etapa.

1. Suba os containers:

```bash
make up
```

2. Garanta os datasets:

```bash
make data
```

3. Importe o grafo no Neo4j:

```bash
make seed
```

4. Baixe o modelo de embedding, se ainda nao tiver:

```bash
make ollama-pull-embed
```

5. Gere o indice vetorial:

```bash
make vectors
```

6. Confira estatisticas:

```bash
make stats
```

7. Abra a aplicacao:

```bash
make app
```

## Makefile

Rode:

```bash
make help
```

Principais comandos:

```bash
make bootstrap       # setup completo: Docker, dados, Neo4j e vetores
make up              # sobe Neo4j e app
make down            # para os containers sem apagar volume do Neo4j
make reset           # remove containers e volumes do Neo4j
make data            # garante datasets em data/raw
make seed            # importa o grafo no Neo4j
make vectors         # gera embeddings e indice vetorial local
make stats           # imprime estatisticas do grafo
make app             # mostra URL da aplicacao
make neo4j           # mostra URL e credenciais do Neo4j Browser
make logs            # acompanha logs
make ps              # lista containers
```

Comandos com Ollama:

```bash
make ollama-list        # lista modelos instalados no host
make ollama-pull        # baixa o modelo LLM definido por MODEL
make ollama-pull-embed  # baixa o modelo de embedding definido por EMBED_MODEL
make ollama-show        # mostra metadados do modelo definido por MODEL
make ollama-warm        # preaquece o modelo LLM antes da apresentacao
```

Comandos de pergunta:

```bash
make ask Q="How is Frodo connected to Sauron?" MODE=hybrid
make ask Q="How is Frodo connected to Sauron?" MODE=rag
make ask Q="How is Frodo connected to Sauron?" MODE=graph
make compare Q="How are Frodo and Sauron connected through the One Ring?"
```

Parametros uteis:

```bash
Q="..."                  # pergunta
MODE=rag|graph|hybrid    # modo de recuperacao
STRATEGY=kg_index        # estrategia GraphRAG
HOPS=2                   # profundidade do subgrafo
TOP_K=8                  # numero de evidencias textuais
MODEL=qwen3.6:latest     # modelo LLM do Ollama
EMBED_MODEL=nomic-embed-text:latest
NUM_CTX=16384            # janela de contexto enviada ao Ollama
TIMEOUT=60               # timeout de geracao
```

Exemplos:

```bash
make ask Q="Why does the One Ring matter to Frodo?" MODE=rag TOP_K=8
make ask Q="Who are the connectors between Frodo and Mordor?" MODE=graph HOPS=3
make ask Q="How is Frodo connected to Sauron?" MODE=hybrid STRATEGY=path
make compare Q="How are Frodo and Sauron connected through the One Ring?" TOP_K=8
```

## Usando a Interface

Abra http://localhost:8000.

Abas principais:

- **Overview**: resumo do corpus, grafo, exemplos e status local.
- **RAG**: busca vetorial pura em chunks dos livros e falas dos scripts.
- **Graph**: laboratorio Cypher read-only, visualizacao de subgrafos e sintese
  Graph-only.
- **GraphRAG**: estrategias hibridas que combinam subgrafo, embeddings,
  filtros, caminhos, comunidades e Cypher.
- **Compare**: compara RAG, Graph e GraphRAG para a mesma pergunta.
- **Flow**: roteiro guiado para apresentar a aula.

Na coluna lateral:

- **Pergunta**: texto usado nos modos RAG, GraphRAG e Compare.
- **Top-k**: quantas evidencias textuais entram no resultado.
- **Saltos / Centro**: parametros do subgrafo.
- **Modelo LLM**: lista modelos detectados no Ollama local.
- **Sintese com LLM**: se desligado, o app mostra apenas evidencias recuperadas;
  se ligado, o Ollama sintetiza a resposta.

## Estrategias GraphRAG

A aba GraphRAG implementa seis familias:

- `kg_index`: entidades da pergunta abrem um subgrafo k-hop; chunks que
  mencionam sementes/vizinhos ganham boost.
- `vector_first`: busca vetorial primeiro; entidades dos hits expandem o grafo
  depois.
- `graph_filter`: o subgrafo vira filtro duro para documentos ligados por
  `MENTIONS`.
- `path`: caminhos curtos e conectores 2-hop reforcam evidencias textuais.
- `community`: usa comunidade estrutural como contexto local-to-global.
- `cypher`: consulta simbolica por `MENTIONS`; a aba Graph mostra a geracao de
  Cypher revisavel.

Para comparar pelo terminal:

```bash
make smoke-strategies Q="How is Frodo connected to Sauron?"
```

## Dados

Fontes usadas:

- Raphtory LOTR interaction graph;
- LOTRO OWL ontology;
- SNA_LOTR, com livros limpos, scripts, redes, sentimento e predicoes.

Arquivos brutos:

```text
data/raw/
```

Indice vetorial:

```text
data/vector_store/
```

O indice vetorial nao precisa ser baixado: gere localmente com `make vectors`.
Ele usa embeddings normalizados e cosine similarity.

Terminologia da UI:

- **Fontes**: obras de origem, como livros e filmes.
- **Chunks livro**: `TextChunk`, pedaços dos livros completos.
- **Falas script**: `DialogueLine`, falas dos scripts dos filmes.
- **Unidades RAG**: tudo que pode ser recuperado pelo RAG: `TextChunk` +
  `DialogueLine`.
- **RetrievalDocument**: superclasse tecnica para uma unidade recuperavel; nao
  significa arquivo fonte.

## Neo4j Browser

Abra:

```text
http://localhost:7474
```

Credenciais:

```text
usuario: neo4j
senha: graphrag-lotr
```

Consultas uteis:

```cypher
MATCH p = (:Entity {name: "Frodo"})-[*1..2]-()
RETURN p
LIMIT 80;
```

```cypher
MATCH p = shortestPath(
  (:Entity {name: "Frodo"})-[*..4]-(:Entity {name: "Sauron"})
)
RETURN p;
```

```cypher
MATCH (d:RetrievalDocument)-[rf:MENTIONS]->(f:Entity {name: "Frodo"})
MATCH (d)-[rs:MENTIONS]->(s:Entity {name: "Sauron"})
WITH d, rf, f, rs, s
ORDER BY coalesce(d.mentionCount, 0) DESC
LIMIT 10
RETURN d, rf, f, rs, s;
```

Para visualizar grafo no Neo4j Browser, retorne objetos completos:

- bom: `RETURN p`
- bom: `RETURN n, r, m`
- ruim para visualizacao: `RETURN n.name`

`RETURN n.name` vira tabela, nao grafo.

## Validacao

Depois do setup:

```bash
make smoke
```

Para validar apenas o indice vetorial em uma amostra pequena:

```bash
make smoke-vectors
```

Para validar contratos de prompt/LLM:

```bash
make llm-check
```

## Troubleshooting

### O app abriu, mas o LLM nao responde

Verifique se o Ollama esta rodando no host:

```bash
ollama list
```

Se o modelo default nao existir, escolha outro:

```bash
make ask Q="How is Frodo connected to Sauron?" MODEL=llama3.1:8b
```

Na UI, escolha um modelo no seletor **Modelo LLM**.

### O embedding falhou

Baixe o modelo de embedding:

```bash
make ollama-pull-embed EMBED_MODEL=nomic-embed-text:latest
```

Depois gere os vetores:

```bash
make vectors
```

### Quero recomecar do zero

Isto apaga o volume do Neo4j:

```bash
make reset
make bootstrap
```

### Porta ocupada

As portas default sao:

- app: `8000`
- Neo4j Browser: `7474`
- Neo4j Bolt: `7687`

Se alguma porta estiver ocupada, altere `docker-compose.yml` ou pare o processo
que esta usando a porta.

### Estou no Linux e o container nao acessa o Ollama

O compose ja inclui:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Se ainda assim falhar, confirme que o Ollama esta aceitando conexoes no host e
que `OLLAMA_BASE_URL` no container aponta para:

```text
http://host.docker.internal:11434
```

### Quero trocar modelo ou contexto

```bash
make ask Q="How is Frodo connected to Sauron?" MODEL=llama3.1:8b NUM_CTX=32768 TIMEOUT=180
```

## Observacoes de Metodo

- Coocorrencia nao e causalidade.
- O modo Graph-only nao usa chunks textuais.
- O modo RAG puro nao usa subgrafo nem boost estrutural.
- O modo GraphRAG combina evidencia textual e estrutural.
- Scores de RAG puro (`cosine`) nao sao diretamente comparaveis com scores de
  GraphRAG (`cosine + graph boost`).
- As perguntas de demo ficam em ingles porque o corpus textual e os nomes do
  grafo estao majoritariamente em ingles.
