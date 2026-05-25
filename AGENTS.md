# AGENTS.md

Instrucoes para agentes trabalhando neste repositorio.

## Contexto

Este projeto e uma demo Docker-first para uma aula de doutorado sobre GNN, Knowledge Graphs, LLMs e GraphRAG. O foco e ser apresentavel, visual e confiavel em pouco tempo, nao ser um framework generico de GraphRAG.

## Objetivo do Produto

Entregar uma aplicacao local que:

- sobe com Docker Compose;
- baixa e importa dados de Senhor dos Aneis;
- cria um grafo hibrido no Neo4j;
- mostra o grafo visualmente;
- gera embeddings locais para chunks/falas;
- executa retrieval por subgrafo/k-hop;
- compara RAG vetorial, Graph retrieval e GraphRAG hibrido;
- usa Ollama local no host para gerar respostas em portugues;
- ajuda o apresentador a conectar GraphRAG com conceitos de GNN.

## Decisoes Fixas

- Banco: Neo4j.
- LLM: Ollama local no host, acessado pelo app em Docker via `host.docker.internal:11434`.
- Modelo default: `qwen3.6:latest`.
- Modelo default de embedding: `nomic-embed-text:latest`.
- A UI deve listar modelos do Ollama via `/api/models`, que usa `/api/tags` no host.
- Dataset principal: Raphtory LOTR cooccurrence graph + SNA_LOTR.
- Camada semantica: `Lotro/lotro.github.io` OWL ontology.
- Corpus textual: textos completos limpos e scripts do SNA_LOTR versionados em `data/raw/sna_lotr/`.
- App: FastAPI + frontend estatico.
- Vector store: indice local em `data/vector_store/`, gerado por `make vectors`, sem servico externo obrigatorio.
- Visualizacao: SVG/JavaScript nativo, sem CDN obrigatorio.

## Regras de Engenharia

- Manter a demo simples e robusta.
- Evitar dependencias pesadas se uma implementacao pequena resolver.
- Preferir scripts reproduziveis em `scripts/`.
- Nao depender de API paga.
- Nao exigir que o usuario instale nada fora Docker, exceto Ollama ja disponivel no host.
- Nao criar container de Ollama por padrao; o fluxo oficial usa o Ollama local do usuario.
- Comandos de apresentacao devem estar no `Makefile`.
- O app deve degradar bem se o Ollama falhar: mostrar contexto recuperado mesmo sem resposta gerada.

## Modelagem do Grafo

Labels principais:

- `Entity`
- `Character`
- `Weapon`
- `Place`
- `Language`
- `Race`
- `Name`
- `Book`
- `Movie`
- `Chapter`
- `RetrievalDocument`
- `TextChunk`
- `DialogueLine`

Terminologia de RAG:

- `Book`, `Movie` e `Chapter` sao fontes/estrutura narrativa.
- `TextChunk` e `DialogueLine` sao as unidades recuperaveis pelo RAG.
- `RetrievalDocument` e uma superclasse tecnica para `TextChunk` + `DialogueLine`; nao significa arquivo fonte.

Relacoes principais:

- `INTERACTS_WITH`
- `CO_OCCURS_WITH`
- `PREDICTED_LINK`
- `HAS_RACE`
- `FRIEND_OF`
- `ENEMY_OF`
- `HAS_WEAPON`
- `INHABITANT`
- `SPEAKS`
- `SPOKEN_BY`
- `SPOKEN_IN`
- `ADOPTED`
- `ADOPTED_BY`
- `NEPHEW_OF`
- `UNCLE_OF`
- `HAS_NAME`
- `MENTIONS`
- `SPEAKS_LINE`
- `IN_BOOK`
- `IN_MOVIE`
- `IN_CHAPTER`
- `SIMILAR_CHAPTER`

## Cuidados

- Nao vender coocorrencia como relacao causal.
- Separar claramente backbone narrativo de semantica ontologica.
- Preservar aliases importantes:
  - `Sam` -> `Samwise`
  - `Merry` -> `Meriadoc`
  - `Pippin` -> `Peregrin`
  - `Elessar` -> `Aragorn`
  - `Strider` -> `Aragorn`
  - `Smeagol`/`Sméagol` -> `Gollum`
- As respostas do LLM devem citar que sao baseadas no contexto recuperado, separando evidencia textual de evidencia estrutural quando for util.
- Cypher gerado para visualizacao no Neo4j Browser deve retornar objetos completos (`RETURN p` ou `RETURN a, r, b`), nao apenas propriedades escalares como `a.name AS nome`.
- `rag` usa embeddings Ollama + cosine similarity sobre `RetrievalDocument`.
- BM25 existe apenas como fallback quando o indice vetorial ainda nao foi gerado ou o modelo de embedding falha.
- `graph` usa entidades, subgrafo k-hop, caminhos e vizinhos.
- `hybrid` usa uma estrategia GraphRAG selecionavel:
  - `kg_index`: subgrafo k-hop da boost no ranking vetorial.
  - `vector_first`: RAG vetorial puro primeiro, grafo expandido apenas pelas mencoes recuperadas; fallback deve ser explicitado.
  - `graph_filter`: subgrafo como filtro duro dentro da busca vetorial para documentos ligados por `MENTIONS`.
  - `path`: caminhos/conectores como foco do reranking; deve degradar para `kg_index` se nao houver par/caminho/conector real.
  - `community`: comunidade estrutural como contexto local-to-global; fallback para k-hop deve aparecer no trace.
  - `cypher`: consulta simbolica deterministica por entidades/documentos via `MENTIONS`; geracao livre de Cypher fica na aba Graph.

## Validacao Antes de Finalizar

Rodar:

```bash
make up
make data
make seed
make vectors
make stats
make smoke-strategies
make ask Q="Como Frodo se conecta a Sauron?" MODE=hybrid
make compare Q="Qual a relação de Frodo com Sauron?"
```

Verificar:

- Neo4j Browser abre em http://localhost:7474.
- App abre em http://localhost:8000.
- `make stats` mostra nos e relacoes.
- A pergunta retorna entidades detectadas, contexto textual/estrutural e resposta ou fallback.
- A comparacao mostra diferencas claras entre RAG, Graph e GraphRAG.
