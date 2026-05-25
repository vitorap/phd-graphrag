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
- Agregar `INTERACTS_WITH` por par de personagens.
- Importar raca/genero.
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
- Gerar contexto textual com:
  - entidades detectadas;
  - relacoes recuperadas;
  - top vizinhos;
  - caminhos quando houver duas entidades;
  - metricas relevantes.
- Chamar Ollama via `/api/chat`.
- Criar fallback quando Ollama estiver indisponivel.

Critério de pronto:

- `make ask Q="Como Frodo se conecta a Sauron?"` retorna contexto e resposta.

## Fase 6: App Visual

Status: concluida.

Tarefas:

- Criar endpoint de estatisticas.
- Criar endpoint de subgrafo.
- Criar endpoint de pergunta.
- Criar UI com:
  - painel de pergunta;
  - seletor de hops;
  - grafo SVG interativo;
  - entidades detectadas;
  - contexto recuperado;
  - resposta do LLM;
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
9. Alterar hops.
10. Conectar com k-hop/message passing.

## Riscos e Mitigacoes

- **Ollama indisponivel**: app mostra contexto recuperado e fallback sem travar.
- **Internet indisponivel**: rodar `make data` antes da aula.
- **Neo4j lento na primeira subida**: healthcheck e retry nos scripts.
- **Dataset pequeno da ontologia**: usar ontologia como camada semantica, nao como grafo principal.
- **Coocorrencia confundida com relacao real**: explicar no roteiro e separar tipo de aresta.

## Pontos Centrais que Ainda Podem Ser Ajustados

- Titulo final do seminario.
- Se a fala tera 20 ou 40 minutos.
- Qual modelo Ollama usar no dia.
- Se o parceiro vai cobrir introducao teorica.
- Se vale incluir uma comparacao explicita com RAG vetorial simples.
