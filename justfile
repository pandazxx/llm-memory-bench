# Recipes for llm-memory-bench.
# Install just from https://github.com/casey/just  (brew install just / apt install just).
# All recipes run through uv, so `just sync` once first.

default: help

help:
    @just --list

# Sync Python dependencies via uv (creates .venv).
sync:
    uv sync

# List the datasets available under data/datasets (by name).
datasets:
    @ls -1 data/datasets | sed 's/\.yaml$//'

# Run one experiment: System x param-set x dataset x test-set.
# Writes results/<dataset>_<testset>/<system>_<paramset>/{memory,result}/.
# Examples:
#   just run
#   just run -- --system amem --params default --dataset comparison --testset comparison
run *ARGS="--system amem --params default --dataset comparison --testset comparison":
    uv run lmb run {{ARGS}}

# Run every registered system (each with its default param-set) against one
# dataset/test-set, then build the shared comparison index.
#   just compare
#   just compare conversation conversation
compare DATASET="comparison" TESTSET="comparison":
    uv run lmb run --system amem      --params default           --dataset {{DATASET}} --testset {{TESTSET}}
    uv run lmb run --system hipporag  --params hipporag-default  --dataset {{DATASET}} --testset {{TESTSET}}
    uv run lmb run --system hipporag2 --params hipporag2-default --dataset {{DATASET}} --testset {{TESTSET}}
    @echo "Comparison index: results/{{DATASET}}_{{TESTSET}}/index.html"

# Re-render all memory + retrieval + comparison HTML for an experiment from
# frozen state.json/result.json — no LLM calls. Use after editing the viz layer.
#   just rerender
#   just rerender conversation conversation
rerender DATASET="comparison" TESTSET="comparison":
    uv run lmb rerender --dataset {{DATASET}} --testset {{TESTSET}}

# Low-level: re-render one system's memory HTML from a single frozen state.json.
# For a whole experiment (memory + retrieval + comparison) use `just rerender`.
#   just render -- --system hipporag --state-dir results/comparison_comparison/hipporag_default/memory
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
