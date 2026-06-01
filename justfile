# Recipes for llm-memory-bench.
# Install just from https://github.com/casey/just  (brew install just / apt install just).
# All recipes run through uv, so `just sync` once first.

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# Run one experiment: System x param-set x dataset x test-set.
# Writes results/<dataset>_<testset>/<system>_<paramset>/{memory,result}/.
# Examples:
#   just run
#   just run -- --system amem --params default --dataset comparison --testset comparison
run *ARGS="--system amem --params default --dataset comparison --testset comparison":
    uv run lmb run {{ARGS}}

# Re-render memory + retrieval HTML from frozen state.json/result.json.
# No LLM calls — use after editing the viz layer to iterate on visualisations.
#   just render
#   just render -- --dataset comparison --testset comparison
render *ARGS:
    uv run lmb render {{ARGS}}

# Convert the reference a-mem-reproduction dataset.json into our YAML
# dataset + test-set under data/.
convert *ARGS:
    uv run lmb convert {{ARGS}}

# Lint + format (ruff) and type-check.
fmt:
    uv run ruff format .
    uv run ruff check --fix .

lint:
    uv run ruff check .

# Run the test suite.
test *ARGS:
    uv run pytest {{ARGS}}
