"""HippoRAG v1 ported onto the framework's System interface.

Reproduces Jiménez et al. 2024 (NeurIPS, "HippoRAG"). Offline indexing turns
passages into an OpenIE knowledge graph (entity nodes + relation edges) plus
synonymy edges where entity-embedding cosine ≥ threshold. Retrieval seeds the
graph with the query's entities, runs Personalized PageRank, down-weights
popular entities by specificity, projects entity scores onto passages, then a
reader LLM answers from the top-K.

LLM + embeddings = NVIDIA NIM. KG math = numpy + scipy.sparse.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..backend import nim
from ..core.models import Query, QueryResult, Statement
from ..core.system import System
from ..viz import kg as kg_viz
from . import _kg_common as kgc

log = logging.getLogger(__name__)


class HippoRAG(System):
    name = "hipporag"
    label = "HippoRAG v1"
    DEFAULT_PARAMS = {
        "sim_threshold": 0.8,
        "retrieval_k": 5,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.item_ids: list[str] = []
        self.passages: list[str] = []
        self.triples_per_passage: dict[int, list[tuple]] = {}
        self.index: dict[str, Any] = {}
        self._edges: list[dict] = []

    # ── indexing ──────────────────────────────────────────────────────────

    def construct_memory(self, statements: list[Statement]) -> None:
        self.item_ids = [s.id for s in statements]
        self.passages = [s.statement for s in statements]
        sim_threshold = float(self.params["sim_threshold"])
        P = len(self.passages)

        # 1. OpenIE per passage.
        self.triples_per_passage = {}
        for i, passage in enumerate(self.passages):
            triples = nim.extract_triples(passage)
            self.triples_per_passage[i] = triples
            log.info("[%d/%d] %s: %d triple(s)", i + 1, P, self.item_ids[i], len(triples))

        # 2. Entity index + entity→passage membership.
        entity_to_passages: dict[str, set[int]] = defaultdict(set)
        for pidx, triples in self.triples_per_passage.items():
            for subj, _pred, obj in triples:
                entity_to_passages[subj].add(pidx)
                entity_to_passages[obj].add(pidx)
        entities = sorted(entity_to_passages.keys())
        entity_idx = {e: i for i, e in enumerate(entities)}
        N = len(entities)

        # 3. Entity embeddings (one batched NIM call).
        embeddings = nim.embed_batch(entities) if entities else np.zeros((0, 1024))

        # 4. Triple edges (symmetric, weight accumulates on repeats).
        graph: dict[tuple[int, int], float] = defaultdict(float)
        edge_records: list[dict] = []
        for triples in self.triples_per_passage.values():
            for subj, pred, obj in triples:
                si, oi = entity_idx[subj], entity_idx[obj]
                graph[(si, oi)] += 1.0
                graph[(oi, si)] += 1.0
                edge_records.append(
                    {"src": subj, "dst": obj, "kind": "relation", "predicate": pred}
                )

        # 5. Synonymy edges (cosine ≥ threshold).
        sims = embeddings @ embeddings.T if N > 0 else np.zeros((0, 0))
        for i in range(N):
            for j in range(i + 1, N):
                if sims[i, j] >= sim_threshold:
                    graph[(i, j)] = float(sims[i, j])
                    graph[(j, i)] = float(sims[i, j])
                    edge_records.append(
                        {
                            "src": entities[i],
                            "dst": entities[j],
                            "kind": "synonymy",
                            "weight": round(float(sims[i, j]), 3),
                        }
                    )

        # 6. Sparse CSR adjacency.
        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i)
            cols.append(j)
            data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)

        # 7. Specificity + entity→passage incidence matrix.
        specificity = (
            np.array([1.0 / len(entity_to_passages[e]) for e in entities], dtype=np.float64)
            if N > 0
            else np.zeros(0)
        )
        pr, pc = [], []
        for e, pidxs in entity_to_passages.items():
            ei = entity_idx[e]
            for pidx in pidxs:
                pr.append(ei)
                pc.append(pidx)
        P_matrix = sp.csr_matrix(
            (np.ones(len(pr), dtype=np.float64), (pr, pc)), shape=(max(N, 1), P)
        )

        self.index = dict(
            entities=entities,
            entity_idx=entity_idx,
            embeddings=embeddings,
            adj=adj,
            specificity=specificity,
            P_matrix=P_matrix,
        )
        self._edges = edge_records
        log.info("indexed %d entities, %d graph edges", N, len(graph) // 2)

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query: Query) -> QueryResult:
        k = int(self.params["retrieval_k"])
        q_ents = nim.extract_query_entities(query.query)
        entities = self.index.get("entities", [])
        embeddings = self.index.get("embeddings")
        adj = self.index.get("adj")

        seed_indices: list[int] = []
        seed_trace: list[dict[str, Any]] = []
        if entities and embeddings is not None and len(entities) > 0:
            for qe in q_ents:
                qe_emb = nim.embed(qe)
                sims = embeddings @ qe_emb
                best_idx = int(np.argmax(sims))
                seed_trace.append(
                    {
                        "query_entity": qe,
                        "matched": entities[best_idx],
                        "similarity": round(float(sims[best_idx]), 3),
                    }
                )
                if best_idx not in seed_indices:
                    seed_indices.append(best_idx)

        if adj is None or adj.shape[0] == 0 or not seed_indices:
            top: list[tuple[int, float]] = []
        else:
            ppr = kgc.ppr_from_indices(adj, seed_indices)
            weighted = ppr * self.index["specificity"]
            scores = np.array(self.index["P_matrix"].T.dot(weighted), dtype=np.float64).flatten()
            order = np.argsort(-scores)[:k]
            top = [(int(i), float(scores[i])) for i in order if scores[i] > 0]

        retrieved_ids = [self.item_ids[pidx] for pidx, _ in top]
        passages = [(self.item_ids[pidx], self.passages[pidx]) for pidx, _ in top]
        answer = kgc.read(query.query, passages)

        return QueryResult(
            query_id=query.id,
            retrieved_ids=retrieved_ids,
            answer=answer,
            trail={
                "kind": "hipporag",
                "query_entities": q_ents,
                "seed_entities": seed_trace,
                "scores": [round(s, 5) for _, s in top],
            },
        )

    # ── visualization + state ─────────────────────────────────────────────

    def _state(self) -> dict[str, Any]:
        items = [
            {
                "id": pid,
                "content": txt,
                "triples": [list(t) for t in self.triples_per_passage.get(i, [])],
            }
            for i, (pid, txt) in enumerate(zip(self.item_ids, self.passages, strict=True))
        ]
        return {
            "system": self.name,
            "label": self.label,
            "params": self.params,
            "items": items,
            "edges": self._edges,
        }

    def dump_memory_state(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "state.json").write_text(json.dumps(self._state(), indent=2))

    def render_memory(self, out_dir: Path) -> None:
        kg_viz.render_memory(self._state(), out_dir)

    def render_retrieval(self, results: list[QueryResult], out_dir: Path) -> None:
        kg_viz.render_retrieval(self._state(), results, out_dir)

    @classmethod
    def render_from_state(cls, state_dir: Path, out_dir: Path) -> None:
        state = json.loads((state_dir / "state.json").read_text())
        kg_viz.render_memory(state, out_dir)

    @classmethod
    def render_retrieval_from_state(
        cls, state_dir: Path, results: list[QueryResult], out_dir: Path
    ) -> None:
        state = json.loads((state_dir / "state.json").read_text())
        kg_viz.render_retrieval(state, results, out_dir)
