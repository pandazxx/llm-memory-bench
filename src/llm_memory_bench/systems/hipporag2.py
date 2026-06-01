"""HippoRAG v2 ported onto the framework's System interface.

What v2 changes vs v1: passages become first-class graph nodes (context edges
wire phrases to the passages they appear in); the query is matched to triples by
embedding cosine instead of NER; a recognition-memory LLM filter keeps only the
relevant triples, whose phrases seed PPR at weight 1.0 while every passage seeds
at a small weight scaled by query↔passage similarity; passage scores are read
directly off the passage portion of the PPR vector — no specificity projection.
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


class HippoRAG2(System):
    name = "hipporag2"
    label = "HippoRAG v2"
    DEFAULT_PARAMS = {
        "sim_threshold": 0.75,
        "top_k_triples": 10,
        "passage_seed_weight": 0.05,
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
            self.triples_per_passage[i] = nim.extract_triples(passage)
            log.info(
                "[%d/%d] %s: %d triple(s)",
                i + 1,
                P,
                self.item_ids[i],
                len(self.triples_per_passage[i]),
            )

        # 2. Phrase set + flat triple list (each remembers its source passage).
        phrase_set: set[str] = set()
        triples: list[tuple] = []  # (subj, pred, obj, pidx)
        for pidx, tps in self.triples_per_passage.items():
            for s, p, o in tps:
                phrase_set.add(s)
                phrase_set.add(o)
                triples.append((s, p, o, pidx))
        phrases = sorted(phrase_set)
        phrase_idx = {ph: i for i, ph in enumerate(phrases)}
        N_phrase = len(phrases)
        N = N_phrase + P

        # 3. Embeddings: phrases, passages, "s p o" triple texts.
        phrase_embs = nim.embed_batch(phrases) if phrases else np.zeros((0, 1024))
        passage_embs = nim.embed_batch(self.passages) if self.passages else np.zeros((0, 1024))
        triple_texts = [f"{s} {p} {o}" for s, p, o, _ in triples]
        triple_embs = (
            nim.embed_batch(triple_texts)
            if triples
            else np.zeros((0, phrase_embs.shape[1] if N_phrase else 1024))
        )

        # 4. Edges.
        graph: dict[tuple[int, int], float] = defaultdict(float)
        edge_records: list[dict] = []

        # 4a. Relation edges (phrase ↔ phrase, from triples).
        relation_pairs: set[tuple[int, int]] = set()
        for s, pred, o, _pidx in triples:
            si, oi = phrase_idx[s], phrase_idx[o]
            if si == oi:
                continue
            graph[(si, oi)] = max(graph[(si, oi)], 1.0)
            graph[(oi, si)] = max(graph[(oi, si)], 1.0)
            if tuple(sorted([si, oi])) not in relation_pairs:
                edge_records.append({"src": s, "dst": o, "kind": "relation", "predicate": pred})
            relation_pairs.add(tuple(sorted([si, oi])))

        # 4b. Synonymy edges (phrase ↔ phrase, cosine ≥ threshold).
        if N_phrase > 1:
            sims = phrase_embs @ phrase_embs.T
            for i in range(N_phrase):
                for j in range(i + 1, N_phrase):
                    if sims[i, j] >= sim_threshold and tuple(sorted([i, j])) not in relation_pairs:
                        graph[(i, j)] = max(graph[(i, j)], float(sims[i, j]))
                        graph[(j, i)] = max(graph[(j, i)], float(sims[i, j]))
                        edge_records.append(
                            {
                                "src": phrases[i],
                                "dst": phrases[j],
                                "kind": "synonymy",
                                "weight": round(float(sims[i, j]), 3),
                            }
                        )

        # 4c. Context edges (phrase ↔ passage node).
        seen_ctx: set[tuple[int, int]] = set()
        for s, _p, o, pidx in triples:
            pni = N_phrase + pidx
            for ei in (phrase_idx[s], phrase_idx[o]):
                if (ei, pni) in seen_ctx:
                    continue
                graph[(ei, pni)] = 1.0
                graph[(pni, ei)] = 1.0
                seen_ctx.add((ei, pni))

        # 5. Sparse CSR adjacency over the combined node space.
        rows, cols, data = [], [], []
        for (i, j), w in graph.items():
            rows.append(i)
            cols.append(j)
            data.append(w)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(max(N, 1), max(N, 1)), dtype=np.float64)

        self.index = dict(
            phrases=phrases,
            phrase_idx=phrase_idx,
            N_phrase=N_phrase,
            P=P,
            N=N,
            phrase_embs=phrase_embs,
            passage_embs=passage_embs,
            triples=triples,
            triple_embs=triple_embs,
            adj=adj,
        )
        self._edges = edge_records
        log.info(
            "indexed %d phrases + %d passages, %d triples",
            N_phrase,
            P,
            len(triples),
        )

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query: Query) -> QueryResult:
        idx = self.index
        k = int(self.params["retrieval_k"])
        top_k_triples = int(self.params["top_k_triples"])
        passage_seed_weight = float(self.params["passage_seed_weight"])

        q_emb = nim.embed(query.query, input_type="query")

        top_triples: list[tuple] = []
        top_triple_sims: list[float] = []
        if idx.get("triples"):
            triple_sims = idx["triple_embs"] @ q_emb
            order = np.argsort(-triple_sims)[:top_k_triples]
            top_triples = [
                (idx["triples"][i][0], idx["triples"][i][1], idx["triples"][i][2]) for i in order
            ]
            top_triple_sims = [float(triple_sims[i]) for i in order]

        filtered = nim.filter_triples(query.query, top_triples, top_k=4)

        seeds = np.zeros(idx["N"], dtype=np.float64) if idx.get("N") else np.zeros(1)
        for s, _p, o in filtered:
            if s in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][s]] += 1.0
            if o in idx["phrase_idx"]:
                seeds[idx["phrase_idx"][o]] += 1.0
        if idx.get("P") and idx["passage_embs"].size:
            passage_sims = idx["passage_embs"] @ q_emb
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] += passage_seed_weight * max(
                    0.0, float(passage_sims[pidx])
                )
        if seeds.sum() == 0 and idx.get("P"):
            for pidx in range(idx["P"]):
                seeds[idx["N_phrase"] + pidx] = 1.0 / idx["P"]

        if idx.get("adj") is not None and idx["adj"].shape[0] > 0:
            ppr = kgc.ppr_from_weights(idx["adj"], seeds)
            passage_scores = ppr[idx["N_phrase"] :]
            order = np.argsort(-passage_scores)[:k]
            top = [(int(i), float(passage_scores[i])) for i in order if passage_scores[i] > 0]
        else:
            top = []

        retrieved_ids = [self.item_ids[pidx] for pidx, _ in top]
        passages = [(self.item_ids[pidx], self.passages[pidx]) for pidx, _ in top]
        answer = kgc.read(query.query, passages)

        return QueryResult(
            query_id=query.id,
            retrieved_ids=retrieved_ids,
            answer=answer,
            trail={
                "kind": "hipporag2",
                "top_triples": [list(t) for t in top_triples],
                "top_triple_sims": [round(s, 3) for s in top_triple_sims],
                "filtered_triples": [list(t) for t in filtered],
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
