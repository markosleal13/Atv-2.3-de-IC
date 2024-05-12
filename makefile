.PHONY: install
install: ## Install Python requirements.
	pip install  poetry
	poetry lock
	poetry install --no-root



.PHONY: run
run: ## Run the project.
	poetry  run python ./app/Modelos de Predicao - Analise.py

