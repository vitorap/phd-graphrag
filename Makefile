SHELL := /bin/sh

Q ?= Como Frodo se conecta a Sauron?
HOPS ?= 2
MODE ?= hybrid
MODEL ?= qwen3.6:latest
NUM_CTX ?= 16384
TIMEOUT ?= 60

.PHONY: help up down reset build data seed stats ask compare app neo4j logs ps shell smoke bootstrap ollama-list ollama-show ollama-warm

help:
	@printf "\nGraphRAG em Middle-earth\n"
	@printf "=========================\n"
	@printf "make up       Sobe Neo4j e app\n"
	@printf "make data     Baixa datasets para data/raw\n"
	@printf "make seed     Importa o grafo hibrido no Neo4j\n"
	@printf "make stats    Mostra estatisticas do grafo\n"
	@printf "make ask      Pergunta via CLI. Use Q=\"...\" MODE=hybrid HOPS=2 MODEL=qwen3.6:latest NUM_CTX=16384\n"
	@printf "make compare  Compara RAG textual, Graph e GraphRAG. Use Q=\"...\" HOPS=2\n"
	@printf "make ollama-show  Mostra metadados do modelo local\n"
	@printf "make ollama-warm  Preaquece o modelo local antes da apresentacao\n"
	@printf "make app      Mostra URL do app\n"
	@printf "make neo4j    Mostra URL e credenciais do Neo4j Browser\n"
	@printf "make logs     Acompanha logs do app e Neo4j\n"
	@printf "make smoke    Roda um teste rapido de ponta a ponta\n"
	@printf "make reset    Remove containers e volumes\n\n"

build:
	docker compose build app

up:
	docker compose up -d --build

down:
	docker compose down

reset:
	docker compose down -v

data: build
	docker compose run --rm --no-deps app python scripts/download_data.py

seed:
	docker compose run --rm app python scripts/import_graph.py

stats:
	docker compose run --rm app python scripts/stats.py

ask:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/ask.py "$(Q)" --hops "$(HOPS)" --mode "$(MODE)"

compare:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/compare.py "$(Q)" --hops "$(HOPS)"

ollama-list:
	ollama list

ollama-show:
	ollama show "$(MODEL)"

ollama-warm:
	ollama run "$(MODEL)" "/no_think\nResponda exatamente: ok"

app:
	@printf "App: http://localhost:8000\n"

neo4j:
	@printf "Neo4j Browser: http://localhost:7474\n"
	@printf "Usuario: neo4j\n"
	@printf "Senha: graphrag-lotr\n"

logs:
	docker compose logs -f

ps:
	docker compose ps

shell:
	docker compose run --rm app sh

smoke:
	docker compose run --rm app python scripts/stats.py
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/ask.py "Como Frodo se conecta a Sauron?" --hops 2 --mode hybrid --no-llm
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/compare.py "Como Frodo se conecta a Sauron?" --hops 2

bootstrap: up data seed stats
