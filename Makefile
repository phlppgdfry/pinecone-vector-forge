# Pinecone Vector Forge - Developer Automation
.PHONY: help install test seed status ui demo pc-login pc-status pc-indexes lint clean docker-build docker-up chat

PYTHON := arch -arm64 /Library/Frameworks/Python.framework/Versions/3.14/bin/python3

help:
	@echo "Pinecone Vector Forge - Available Commands:"
	@echo "  make install      Install Python dependencies & check Pinecone CLI"
	@echo "  make test         Execute test suite and latency benchmarks"
	@echo "  make seed         Ingest and vectorize the AI knowledge corpus"
	@echo "  make status       Inspect index health, vector counts, and namespaces"
	@echo "  make ui           Start the modern glassmorphism web dashboard on :8000"
	@echo "  make chat         Start interactive Pinecone Assistant terminal session"
	@echo "  make demo         Run automated CLI showcase tour"
	@echo "  make pc-login     Launch browser login via official Pinecone CLI"
	@echo "  make pc-status    Check Pinecone CLI authentication state"
	@echo "  make pc-indexes   List serverless indexes in your Pinecone account"
	@echo "  make docker-build Build minimal unprivileged Docker image"
	@echo "  make docker-up    Run application container via Docker Compose"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) -m src.cli.main evaluate

seed:
	$(PYTHON) -m src.seed.ingest

status:
	$(PYTHON) -m src.cli.main status

ui:
	$(PYTHON) -m src.web.app

chat:
	$(PYTHON) -m src.cli.main chat

demo:
	$(PYTHON) -m src.cli.main demo

docker-build:
	docker build -t pinecone-vector-forge:latest .

docker-up:
	docker compose up -d

pc-login:
	pc login

pc-status:
	pc auth status

pc-indexes:
	pc index list

push-cloud:
	./bin/push-to-pinecone.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
