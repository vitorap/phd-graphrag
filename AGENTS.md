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
- executa retrieval por subgrafo/k-hop;
- compara RAG textual, Graph retrieval e GraphRAG hibrido;
- usa Ollama local no host para gerar respostas em portugues;
- ajuda o apresentador a conectar GraphRAG com conceitos de GNN.

## Decisoes Fixas

- Banco: Neo4j.
- LLM: Ollama local no host, acessado pelo app em Docker via `host.docker.internal:11434`.
- Modelo default: `qwen3.6:latest`.
- Dataset principal: Raphtory LOTR cooccurrence graph + SNA_LOTR.
- Camada semantica: `Lotro/lotro.github.io` OWL ontology.
- Corpus textual: textos completos limpos e scripts do SNA_LOTR versionados em `data/raw/sna_lotr/`.
- App: FastAPI + frontend estatico.
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
- `rag` usa BM25 sobre `RetrievalDocument`.
- `graph` usa entidades, subgrafo k-hop, caminhos e vizinhos.
- `hybrid` usa o subgrafo para orientar e reforcar chunks textuais ligados por `MENTIONS`.

## Validacao Antes de Finalizar

Rodar:

```bash
make up
make data
make seed
make stats
make ask Q="Como Frodo se conecta a Sauron?" MODE=hybrid
make compare Q="Qual a relação de Frodo com Sauron?"
```

Verificar:

- Neo4j Browser abre em http://localhost:7474.
- App abre em http://localhost:8000.
- `make stats` mostra nos e relacoes.
- A pergunta retorna entidades detectadas, contexto textual/estrutural e resposta ou fallback.
- A comparacao mostra diferencas claras entre RAG, Graph e GraphRAG.
