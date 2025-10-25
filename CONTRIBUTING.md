# Contributing

Thanks for helping improve **vortex2d**!

## Dev Environment

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Create and activate env
poetry install --with dev,plot
```

## Quality Gates

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

## Commit Style

- Conventional Commits are recommended.
- Run `pre-commit install` once; hooks will lint on commit.

## Releasing

- Bump the version in `pyproject.toml` and create a tag.
