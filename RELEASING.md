
# Releasing vortex2d

## Workflow
1. Follow Conventional Commits in PR titles & merge commits.
2. Merge to `main`. The **release-please** workflow will open a "release PR" proposing:
   - version bump in `pyproject.toml`
   - updated `CHANGELOG.md`
3. Merge the release PR. That action:
   - creates a GitHub release with tag `vX.Y.Z`
   - pushes the version commit to `main`

4. The `publish` workflow (on tag) builds wheels/sdist with Poetry and publishes to PyPI.
   - Uses Trusted Publishing (recommended). Enable this repo at PyPI (Project → "Publishing" → "Trusted Publishers").
   - Manual fallback: set `PYPI_API_TOKEN` repo secret and dispatch the workflow.

## Local checks
```bash
poetry build
poetry publish -r testpypi  # configure token via 'poetry config pypi-token.testpypi ...'
```

## Conda (optional)
- After PyPI release, update `conda/recipe/meta.yaml` sha256 with the sdist checksum.
- Submit a feedstock PR on conda-forge using grayskull / staged-recipes.
