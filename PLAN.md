# Plano Detalhado

## Tese da Demo

GraphRAG e uma ponte natural entre Knowledge Graphs, LLMs e GNNs: em vez de recuperar apenas textos por similaridade, recuperamos uma vizinhanca estruturada, com entidades, relacoes, caminhos e metricas de grafo. Essa vizinhanca e o equivalente pratico do campo receptivo que uma GNN usaria para propagar informacao.

## Entrega Esperada

- Repositorio Docker-first.
- Neo4j com grafo LOTR hibrido.
- App local para visualizacao e perguntas.
- Makefile com comandos de apresentacao.
- README com roteiro da aula.
- Scripts reproduziveis para download, ingestao, metricas e perguntas.
- Comparacao explicita entre RAG textual, Graph e GraphRAG hibrido.

## Fase 1: Documentacao e Roteiro

Status: concluida.

Tarefas:

- Criar README com objetivo, stack, comandos e roteiro.
- Criar AGENTS com decisoes tecnicas para continuidade.
- Criar plano detalhado com arquitetura e sequencia de implementacao.
- Fixar perguntas centrais da aula.

## Fase 2: Infra Docker

Status: concluida.

Tarefas:

- Criar `docker-compose.yml` com Neo4j e app.
- Criar `Dockerfile` para app Python.
- Criar `requirements.txt`.
- Criar `.env.example`.
- Criar `Makefile` com comandos de apresentacao.

Critério de pronto:

- `make up` sobe Neo4j e app.
- `make help` mostra o fluxo da apresentacao.

## Fase 3: Dados

Status: concluida.

Tarefas:

- `scripts/download_data.py` baixa:
  - `lotr.csv`
  - `lotr_properties.csv`
  - `LOTROntology.owl`
  - textos completos do SNA_LOTR
  - scripts dos filmes
  - redes ponderadas, capitulos, sentimento e predicoes
- Guardar tudo em `data/raw/`.
- Validar tamanho minimo dos arquivos.

Critério de pronto:

- `make data` deixa os arquivos prontos localmente.

## Fase 4: Ingestao no Neo4j

Status: concluida.

Tarefas:

- Criar constraints.
- Limpar grafo quando `seed` rodar.
- Importar personagens e aliases do Raphtory.
- Importar personagens e atributos do SNA_LOTR.
- Agregar `INTERACTS_WITH` por par de personagens.
- Importar `CO_OCCURS_WITH` ponderado.
- Importar `PREDICTED_LINK` como camada de link prediction.
- Importar raca/genero.
- Importar `Book`, `Chapter`, `Movie`, `TextChunk` e `DialogueLine`.
- Ligar textos e falas a entidades por `MENTIONS`.
- Ligar falas a personagens por `SPEAKS_LINE`.
- Parsear OWL com `rdflib`.
- Criar entidades semanticas.
- Criar relacoes semanticas.
- Calcular metricas com `networkx`:
  - degree
  - weighted degree
  - PageRank
  - community id

Critério de pronto:

- `make seed` importa sem erro.
- `make stats` mostra contagens coerentes.

## Fase 5: GraphRAG

Status: concluida.

Tarefas:

- Resolver entidades a partir da pergunta.
- Recuperar subgrafo por `hops`.
- Recuperar chunks/falas por BM25.
- Oferecer modos `rag`, `graph` e `hybrid`.
- Gerar contexto textual com:
  - entidades detectadas;
  - relacoes recuperadas;
  - evidencias textuais recuperadas;
  - top vizinhos;
  - caminhos quando houver duas entidades;
  - metricas relevantes.
- Chamar Ollama via `/api/chat`.
- Criar fallback quando Ollama estiver indisponivel.

Critério de pronto:

- `make ask Q="Como Frodo se conecta a Sauron?" MODE=hybrid` retorna contexto e resposta.
- `make compare Q="Qual a relação de Frodo com Sauron?"` mostra as tres estrategias.

## Fase 6: App Visual

Status: concluida.

Tarefas:

- Criar endpoint de estatisticas.
- Criar endpoint de subgrafo.
- Criar endpoint de pergunta.
- Criar endpoint de comparacao.
- Criar UI com:
  - painel de pergunta;
  - seletor de hops;
  - grafo SVG interativo;
  - entidades detectadas;
  - contexto recuperado;
  - resposta do LLM;
  - comparacao lado a lado entre RAG, Graph e GraphRAG;
  - stats do grafo.

Critério de pronto:

- App abre em http://localhost:8000.
- Clique em no atualiza vizinhanca.
- Pergunta atualiza subgrafo e resposta.

## Fase 7: Validacao de Apresentacao

Status: concluida.

Tarefas:

- Rodar fluxo completo do zero.
- Testar perguntas do README.
- Ajustar texto de fallback.
- Garantir que `make help` seja suficiente como cola de apresentacao.

Critério de pronto:

- Uma pessoa consegue rodar a demo seguindo apenas `README.md`.

## Ordem de Execucao na Aula

1. `make help`
2. `make up`
3. `make data`
4. `make seed`
5. `make stats`
6. Abrir Neo4j Browser.
7. Abrir app.
8. Perguntar sobre Frodo/Sauron.
9. Rodar comparacao RAG vs Graph vs GraphRAG.
10. Alterar hops.
11. Conectar com k-hop/message passing.

## Riscos e Mitigacoes

- **Ollama indisponivel**: app mostra contexto recuperado e fallback sem travar.
- **Internet indisponivel**: rodar `make data` antes da aula.
- **Neo4j lento na primeira subida**: healthcheck e retry nos scripts.
- **Grafo grande demais para explicar tudo**: usar a UI para focar em Frodo/Sauron e deixar Neo4j para exploracao.
- **Coocorrencia confundida com relacao real**: explicar no roteiro e separar tipo de aresta.
- **Predicted links gerando ruido**: manter como tipo separado e explicar como camada de link prediction.

## Pontos Centrais que Ainda Podem Ser Ajustados

- Titulo final do seminario.
- Se a fala tera 20 ou 40 minutos.
- Qual modelo Ollama usar no dia.
- Se o parceiro vai cobrir introducao teorica.
- Se vale incluir uma comparacao explicita com RAG vetorial simples.

## Fase 8: Enriquecimento Maximo LOTR

Status: concluida.

Tarefas:

- Versionar `data/raw/sna_lotr/` completo.
- Criar corpus BM25 local sem embeddings externos.
- Criar `scripts/compare.py`.
- Atualizar `Makefile` com `MODE` e `make compare`.
- Atualizar app para exibir comparacao em tres colunas.
- Validar que a pergunta Frodo/Sauron produz resposta hibrida mais rica que grafo puro.
