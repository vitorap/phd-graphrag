# GraphRAG em Middle-earth

Demo pratica para uma aula/seminario de GNN, Knowledge Graphs e LLMs. A ideia e mostrar, em uma unica aplicacao local com Docker, como um GraphRAG pode recuperar subgrafos, caminhos e vizinhancas k-hop em um grafo de Senhor dos Aneis, e como isso se conecta diretamente com intuicoes de GNN/message passing.

## Objetivo da Aula

O seminario parte de uma pergunta simples:

> O que muda quando o contexto entregue ao LLM deixa de ser apenas texto solto e passa a ser um subgrafo estruturado?

A demo usa duas fontes complementares:

- **Raphtory LOTR interaction graph**: grafo de coocorrencia de personagens em sentencas da trilogia. Ele e melhor para visualizacao, centralidade, comunidades, k-hop neighborhood e conexao com GNN.
- **LOTRO OWL ontology**: ontologia RDF/OWL com classes, personagens, lugares, armas, linguas e relacoes semanticas como `friendOf`, `enemyOf`, `hasWeapon`, `inhabitant` e `speaks`.

O resultado e um grafo hibrido:

- `INTERACTS_WITH`: backbone narrativo vindo do dataset da Raphtory.
- relacoes semanticas: camada de conhecimento vinda da ontologia LOTRO.
- atributos/metricas: raca, genero, grau ponderado, PageRank e comunidade.

## Stack

- Docker Compose
- Neo4j Community
- FastAPI
- JavaScript/SVG nativo para visualizacao
- Ollama local no host para gerar respostas
- Python para ingestao, metricas e retrieval

Modelos Ollama detectados nesta maquina:

- `qwen3.6:latest` usado como default, com `context length 262144` reportado por `ollama show`
- `gemma4:26b` como alternativa
- `lfm2:latest` como alternativa

## Como Rodar

```bash
make help
make up
make data
make seed
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
make stats        # mostra estatisticas do grafo
make ask Q="Como Frodo se conecta a Sauron?"
make ollama-show  # mostra metadados do modelo local
make ollama-warm  # preaquece o modelo local antes da apresentacao
make logs         # acompanha logs
make reset        # remove containers e volume do Neo4j
```

## Perguntas Boas para a Demo

```bash
make ask Q="Como Frodo se conecta a Sauron?"
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

2. **Modelo de Grafo (5 min)**
   - Mostrar `Character`, `Weapon`, `Place`, `Language`, `Race`.
   - Mostrar `INTERACTS_WITH` como backbone.
   - Mostrar relacoes semanticas da ontologia.

3. **Visualizacao no Neo4j (7 min)**
   - Abrir Neo4j Browser.
   - Rodar consultas Cypher simples.
   - Mostrar vizinhanca de Frodo, Gandalf, Sauron.

4. **GraphRAG Local (10 min)**
   - Fazer pergunta no app.
   - Mostrar entidades detectadas.
   - Mostrar subgrafo recuperado.
   - Comparar resposta com `hops=1`, `hops=2`, `hops=3`.
   - Ligar a chave `LLM local` apenas quando o modelo ja estiver preaquecido.

5. **Conexao com GNN (8 min)**
   - `k-hop neighborhood` como campo receptivo.
   - Agregacao de vizinhos como analogia a message passing.
   - PageRank/comunidades como features estruturais.
   - Por que GraphRAG e GNN atacam problemas parecidos por mecanismos diferentes.

6. **Limitacoes e Extensoes (5 min)**
   - Coocorrencia nao e causalidade.
   - Ontologia e pequena, mas semanticamente rica.
   - Proximo passo: embeddings, GDS, node classification, link prediction.

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

## Decisoes de Projeto

- **Neo4j em vez de RDF store puro**: facilita visualizacao, Cypher e demo em sala.
- **GraphRAG customizado em vez de framework pesado**: mais controlavel e explicavel para uma aula.
- **Dataset hibrido**: Raphtory da densidade estrutural; LOTRO OWL da semantica.
- **Ollama local**: evita dependencia de API externa.
- **Docker-first**: parceiro consegue rodar com os mesmos comandos.

## Perguntas Centrais para Fechar com o Professor/Turma

- Quando uma vizinhanca k-hop e suficiente, e quando precisamos de busca global/comunidades?
- Em que tipo de pergunta o grafo ajuda mais que um RAG vetorial?
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

Se quiser explorar a janela longa do `qwen3.6:latest`, aumente `NUM_CTX`. Para a demo, o default e propositalmente menor para reduzir latencia:

```bash
make ask Q="Como Frodo se conecta a Sauron?" NUM_CTX=262144 TIMEOUT=180
```
