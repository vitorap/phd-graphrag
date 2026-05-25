SHELL := /bin/sh

Q ?= How is Frodo connected to Sauron?
HOPS ?= 2
TOP_K ?= 8
MODE ?= hybrid
STRATEGY ?= kg_index
MODEL ?= qwen3.6:latest
EMBED_MODEL ?= nomic-embed-text:latest
NUM_CTX ?= 16384
TIMEOUT ?= 60

.PHONY: help up down reset build data seed vectors stats ask compare app neo4j logs ps shell smoke smoke-vectors smoke-strategies llm-check bootstrap ollama-list ollama-show ollama-warm ollama-pull ollama-pull-embed

help:
	@printf "\nGraphRAG em Middle-earth\n"
	@printf "=========================\n"
	@printf "Setup rapido:\n"
	@printf "  make bootstrap       Sobe Docker, importa dados e gera o indice vetorial\n"
	@printf "  make app             Mostra a URL da aplicacao\n"
	@printf "  make neo4j           Mostra URL e credenciais do Neo4j Browser\n\n"
	@printf "Preparacao manual:\n"
	@printf "  make up              Sobe Neo4j e app\n"
	@printf "  make data            Garante datasets em data/raw\n"
	@printf "  make seed            Importa o grafo hibrido no Neo4j\n"
	@printf "  make vectors         Gera embeddings locais. Use EMBED_MODEL=nomic-embed-text:latest\n\n"
	@printf "Ollama no host:\n"
	@printf "  make ollama-list     Lista modelos locais\n"
	@printf "  make ollama-pull     Baixa o modelo LLM definido por MODEL\n"
	@printf "  make ollama-pull-embed  Baixa o modelo de embedding definido por EMBED_MODEL\n"
	@printf "  make ollama-show     Mostra metadados do modelo local\n"
	@printf "  make ollama-warm     Preaquece o modelo local antes da apresentacao\n\n"
	@printf "Uso e validacao:\n"
	@printf "  make stats           Mostra estatisticas do grafo\n"
	@printf "  make ask             Pergunta via CLI. Use Q=\"...\" MODE=hybrid STRATEGY=kg_index HOPS=2 TOP_K=8\n"
	@printf "  make compare         Compara RAG vetorial, Graph e GraphRAG. Use Q=\"...\" HOPS=2 TOP_K=8\n"
	@printf "  make smoke           Roda um teste rapido de ponta a ponta\n"
	@printf "  make smoke-vectors   Teste pequeno de geracao de vetores\n"
	@printf "  make smoke-strategies  Valida invariantes das seis variantes GraphRAG. Requer make vectors\n"
	@printf "  make llm-check       Valida contratos dos prompts LLM\n\n"
	@printf "Manutencao:\n"
	@printf "  make logs            Acompanha logs do app e Neo4j\n"
	@printf "  make ps              Lista containers\n"
	@printf "  make shell           Abre shell descartavel no container app\n"
	@printf "  make down            Para containers sem apagar dados\n"
	@printf "  make reset           Remove containers e volumes do Neo4j\n\n"

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

vectors:
	docker compose run --rm -e OLLAMA_EMBED_MODEL="$(EMBED_MODEL)" app python scripts/build_vectors.py --model "$(EMBED_MODEL)"

stats:
	docker compose run --rm app python scripts/stats.py

ask:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/ask.py "$(Q)" --hops "$(HOPS)" --top-k "$(TOP_K)" --mode "$(MODE)" --strategy "$(STRATEGY)"

compare:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/compare.py "$(Q)" --hops "$(HOPS)" --top-k "$(TOP_K)"

ollama-list:
	ollama list

ollama-show:
	ollama show "$(MODEL)"

ollama-pull:
	ollama pull "$(MODEL)"

ollama-pull-embed:
	ollama pull "$(EMBED_MODEL)"

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
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/ask.py "How is Frodo connected to Sauron?" --hops 2 --mode hybrid --no-llm
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/compare.py "How is Frodo connected to Sauron?" --hops 2

smoke-vectors:
	docker compose run --rm -e OLLAMA_EMBED_MODEL="$(EMBED_MODEL)" -e VECTOR_DIR=/tmp/phd-graphrag-smoke-vectors app sh -c 'python scripts/build_vectors.py --model "$(EMBED_MODEL)" --limit 16 && python scripts/ask.py "How is Frodo connected to Sauron?" --hops 2 --mode rag --no-llm'

smoke-strategies:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/validate_graphrag_strategies.py "$(Q)" --hops "$(HOPS)" --top-k "$(TOP_K)"

llm-check:
	docker compose run --rm -e OLLAMA_MODEL="$(MODEL)" -e OLLAMA_NUM_CTX="$(NUM_CTX)" -e OLLAMA_TIMEOUT="$(TIMEOUT)" app python scripts/validate_llm_contracts.py

bootstrap: ollama-pull-embed up data seed vectors stats
