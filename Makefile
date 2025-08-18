.PHONY: update init

update : pyproject.toml uv.lock .pre-commit-config.yaml
	uv self update;
	uv lock -U;
	uv sync;
	uv cache prune;
	uv run pre-commit autoupdate;
	uv run pre-commit gc;

init : pyproject.toml uv.lock
	curl -LsSf https://astral.sh/uv/install.sh | sh;
	uv venv;
	source .venv/bin/activate;
	uv sync;
	uv run pre-commit install;
