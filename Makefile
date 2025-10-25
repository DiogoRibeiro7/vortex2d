.PHONY: fmt lint type test bench docs

fmt:
	poetry run ruff check . --fix
	poetry run ruff format .

lint:
	poetry run ruff check .
	poetry run ruff format --check

type:
	poetry run mypy src

test:
	poetry run pytest -q

bench:
	poetry run pytest tests/test_bench_small.py --benchmark-only

docs:
	poetry run mkdocs serve -a 127.0.0.1:8000
