# llm-memory-bench — Implementation Plan (Milestone 1)

Status: **proposal, awaiting review**. No framework code is written yet.

This plan turns the README into a concrete build. It is grounded in the
reference project `pandazxx/a-mem-reproduction`, whose three systems (A-Mem,
HippoRAG, HippoRAG2) we will port onto a cleaner, more general framework.

## 1. Agreed decisions

| Topic | Decision |
| --- | --- |
| Language / tooling | Python, managed with **uv** (standard uv project, `pyproject.toml` + `uv.lock`, `src/` layout) |
| Repo root | This repo **is** the project — no nested top-level folder |
| LLM backend | **NVIDIA NIM** (`meta/llama-3.1-70b-instruct`), wrapped behind a thin interface but the only wired provider |
| Embeddings | Local `sentence-transformers` (`all-MiniLM-L6-v2`), same as reference |
| 429 handling | NIM calls **retry indefinitely** on HTTP 429 (never give up) |
| Data format | **YAML** for datasets, test-sets, and param-sets |
| Milestone 1 scope | Framework skeleton + **A-Mem** end-to-end; HippoRAG / HippoRAG2 ported afterward |

## 2. Repository layout (uv standard, src layout)

```
pyproject.toml            # uv project, deps, console-script entrypoint
uv.lock
justfile                  # task runner: sync / run / render / convert / fmt / test
README.md
docs/plan.md              # this file
src/llm_memory_bench/
  __init__.py
  core/
    models.py             # Statement, Query, QueryResult dataclasses
    system.py             # System ABC
    experiment.py         # Experiment definition + runner
    scoring.py            # precision / recall / F1 vs required_retrieval
    storage.py            # result directory layout helpers
  io/
    dataset.py            # load + validate datasets (YAML)
    testset.py            # load + validate test-sets (YAML)
    params.py             # load param-sets, overlay on system defaults
  backend/
    nim.py                # NIM client, call() w/ infinite 429 retry, chat/chat_text
    embeddings.py         # local sentence-transformers helper
  systems/
    __init__.py           # name -> System registry
    amem.py               # A-Mem (Milestone 1)
    # hipporag.py, hipporag2.py  (later milestones)
  viz/
    common.py             # shared HTML scaffolding
    memory.py             # memory-structure visualization
    retrieval.py          # retrieval-trail visualization
  cli.py                  # `lmb run ...`, `lmb render ...`
data/
  datasets/comparison.yaml
  testsets/comparison.yaml
  params/amem-default.yaml
results/                  # generated; gitignored
```

## 3. Core abstractions

### 3.1 Data models (`core/models.py`)

```python
@dataclass
class Statement:        # README "Dataset" entry
    id: str             # S00001
    statement: str
    timestamp: str | None
    comment: str | None # human-only, ignored by systems

@dataclass
class Query:            # README "Test-set" entry
    id: str             # Q00001
    query: str
    required_retrieval: list[str]   # statement ids that should be retrieved
    expected_answer: str | None
    comment: str | None

@dataclass
class QueryResult:      # README "Query result"
    query_id: str
    retrieved_ids: list[str]
    trail: dict         # model-specific (seeds, PPR matrix, links followed, ...)
    answer: str
```

### 3.2 System interface (`core/system.py`)

Maps the README's System responsibilities onto the reference's proven
`ingest / query / dump_state / render_from_run` shape, plus first-class
parameters.

```python
class System(ABC):
    name: str   = "system"   # dir name + CLI id
    label: str  = "System"   # human label
    DEFAULT_PARAMS: dict = {}

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    @abstractmethod
    def construct_memory(self, statements: list[Statement]) -> None: ...

    @abstractmethod
    def retrieve(self, query: Query) -> QueryResult: ...

    # visualization (README "Visualizations")
    @abstractmethod
    def render_memory(self, out_dir: Path) -> None: ...
    @abstractmethod
    def render_retrieval(self, results: list[QueryResult], out_dir: Path) -> None: ...

    # persistence so we can re-render without re-running the LLM
    @abstractmethod
    def dump_memory_state(self, out_dir: Path) -> None: ...
    @classmethod
    @abstractmethod
    def render_from_state(cls, state_dir: Path, out_dir: Path) -> None: ...
```

Notable change vs the reference: `__init__(params)` so a system is tuned by a
param-set, and the README's separate `render_memory` / `render_retrieval`
instead of one `dump_state`.

### 3.3 Experiment (`core/experiment.py`)

```python
@dataclass
class Experiment:
    system: str        # registry key, e.g. "amem"
    params: str        # param-set name, e.g. "default"
    dataset: str       # dataset name
    testset: str       # test-set name
```

`run(experiment)`:
1. Load dataset, test-set, param-set.
2. Instantiate the system with params.
3. `construct_memory(statements)` → `render_memory()` + `dump_memory_state()`.
4. For each query: `retrieve(query)` → collect `QueryResult`.
5. Score, write `result.json`, `render_retrieval()`.
6. Emit comparison `index.html` across all `<system>_<paramset>` runs.

## 4. Storage / output layout (verbatim from README §"Test result storage")

```
results/
  <dataset>_<testset>/
    <system>_<paramset>/
      memory/
        index.html        # memory-structure visualization
        state.json        # frozen memory state for re-render
        <assets>
      result/
        index.html        # retrieval-trail visualization
        result.json       # scored query results
    index.html            # comparison across systems+paramsets
```

## 5. Data formats (YAML)

`data/datasets/comparison.yaml`:
```yaml
name: comparison
statements:
  - id: S00001
    statement: Sam works as a software engineer at TechCorp.
    timestamp: 2024-01-08T09:00:00
    comment: baseline employment fact
```

`data/testsets/comparison.yaml`:
```yaml
name: comparison
dataset: comparison
tests:
  - id: Q00001
    query: Where does Sam live?
    required_retrieval: [S00004]
    expected_answer: Oakland, California
    comment: single-hop direct fact
```

`data/params/amem-default.yaml`:
```yaml
system: amem
name: default
params:
  embedding_model: all-MiniLM-L6-v2
  retrieval_k: 5
  evo_threshold: 100
```

A small converter script will translate the reference `dataset.json`
(`memories` + `questions`, `requires_facts` → `required_retrieval`) into the
two YAML files above so we can run immediately.

## 6. Backend (`backend/nim.py`)

Ported from the reference `_nim.py`, trimmed to what A-Mem needs first
(`chat`, `chat_text`, `parse_json`), with HippoRAG helpers added later.

**429 retry — indefinite (per requirement):**
```python
def call(fn, *args, **kwargs):
    delay = 1.0
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            log.warning("429 rate-limited — retrying in %.1fs", delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)   # exponential backoff, capped; never gives up
```
Reference uses a fixed 5s sleep; we keep the infinite loop but add capped
exponential backoff to ease pressure on the free tier. Cap is configurable.

Embeddings stay local (`backend/embeddings.py` wraps sentence-transformers;
A-Mem keeps using ChromaDB internally).

## 7. Scoring (`core/scoring.py`)

Per query, compare `retrieved_ids` against `required_retrieval`:
- precision, recall, F1 (retrieval quality — the README's "judgemental factor")
- answer captured verbatim; exact/normalized match flag vs `expected_answer`
- LLM-as-judge answer grading deferred to a later milestone

Aggregate: macro precision/recall/F1 + per-category breakdown if the test-set
carries categories.

## 8. Visualizations

Milestone 1 keeps these minimal but functional; graph polish comes later.
- **Memory structure**: table of nodes (id, tags, context) + a graph diagram
  of nodes/edges with weights, side panel on click.
- **Retrieval trail**: table of query results + a per-query diagram (seed
  nodes → linkage retrieval) with a detail side panel.

Reuse the reference's mermaid + lightweight interactive-HTML approach;
`render_from_state` lets us iterate on visuals without re-calling the LLM.

## 8a. Task runner (`justfile`)

All common workflows go through `just` over `uv` (mirrors the reference repo):

| Recipe | Action |
| --- | --- |
| `just sync` | `uv sync` — create `.venv`, install deps |
| `just run` | run one experiment (defaults to amem/default/comparison/comparison) |
| `just render` | re-emit memory + retrieval HTML from frozen state (no LLM calls) |
| `just convert` | convert reference `dataset.json` → `data/` YAML dataset + test-set |
| `just fmt` / `just lint` | ruff format + lint |
| `just test` | pytest |

Recipes pass through extra args, e.g.
`just run -- --system amem --params default --dataset comparison --testset comparison`.

## 9. Milestones

- **M1 (this plan)**: skeleton (models, System ABC, Experiment runner,
  loaders, scoring, storage, NIM backend) + A-Mem end-to-end + converted
  comparison dataset/test-set + minimal viz + CLI. Deliverable: `lmb run
  --system amem --params default --dataset comparison --testset comparison`
  produces the full `results/` tree.
- **M2**: port HippoRAG + HippoRAG2; comparison `index.html`.
- **M3**: visualization polish (interactive graphs, PPR trail), LLM-judge
  answer scoring, param sweeps.

## 10. Open questions for review

1. Console-script name — `lmb` ok, or prefer `llm-memory-bench`?
2. Keep ChromaDB as A-Mem's internal vector store (as in the reference), or
   abstract a shared vector-store layer now? I lean: keep it internal for M1.
3. Backoff cap of 30s acceptable, or do you want the reference's flat 5s?
