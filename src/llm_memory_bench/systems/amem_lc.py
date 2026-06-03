"""A-Mem with late-chunked corpus embeddings.

Identical to :class:`~llm_memory_bench.systems.amem.AMem` (LLM note construction,
write-time evolution, 1-hop link traversal at read time) except for *how* memory
contents are embedded for neighbour search. Vanilla A-Mem embeds each memory in
isolation; here we concatenate the corpus and apply **late chunking** — one
forward pass per packed window so a memory's tokens attend across neighbouring
memories — then mean-pool each memory's token span into its embedding.

Because late chunking needs the whole corpus up front, construction is three
phases: analyse every memory (LLM) → late-chunk embed the corpus → run write-time
evolution. Evolution only edits tags/context/links, never content, so embeddings
computed in phase two stay valid through phase three. Queries are embedded the
normal way (a single forward pass) and matched against the late-chunked vectors.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.models import Statement
from ._late_chunk import late_chunk_embed
from .amem import PROMPT_EVOLUTION, AMem, MemoryNote, _Engine, nim

log = logging.getLogger(__name__)


class _LateChunkEngine(_Engine):
    """A-Mem engine whose vector store is a numpy map of late-chunked embeddings.

    Neighbour and query search are plain cosine over :pyattr:`_emb`; ChromaDB is
    not used (the base class' collection methods are bypassed)."""

    def __init__(self, model_name: str, evo_threshold: int, max_tokens: int):
        from sentence_transformers import SentenceTransformer

        self.memories: dict[str, MemoryNote] = {}
        self.model_name = model_name
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.st_model = SentenceTransformer(model_name)
        model_max = int(getattr(self.st_model, "max_seq_length", 0) or 512)
        self.max_tokens = min(int(max_tokens), model_max)
        self._emb: dict[str, np.ndarray] = {}

    # ── late-chunk encoding ────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        return list(self.st_model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _forward(self, input_ids: list[int]) -> np.ndarray:
        import torch

        model = self.st_model[0].auto_model
        device = next(model.parameters()).device
        ids = torch.tensor([input_ids], device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state[0].cpu().numpy()

    def embed_corpus(self) -> None:
        """Late-chunk every memory's content (insertion order) into ``self._emb``."""
        notes = list(self.memories.values())
        if not notes:
            return
        tok = self.st_model.tokenizer
        embs = late_chunk_embed(
            [n.content for n in notes],
            tokenize=self._tokenize,
            forward=self._forward,
            cls_id=tok.cls_token_id,
            sep_id=tok.sep_token_id,
            max_tokens=self.max_tokens,
        )
        for note, emb in zip(notes, embs, strict=True):
            self._emb[note.id] = np.asarray(emb, dtype=np.float64)

    # ── cosine search over the late-chunked vectors ────────────────────────

    def _neighbors_by_emb(
        self, emb: np.ndarray | None, k: int, candidates: list[str]
    ) -> list[tuple[str, float]]:
        if emb is None or k <= 0 or not candidates:
            return []
        emb = np.asarray(emb, dtype=np.float64)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        scored = []
        for cid in candidates:
            v = self._emb.get(cid)
            if v is None:
                continue
            scored.append((cid, float(np.dot(emb, v))))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search(self, query: str, k: int) -> list[dict]:
        q_emb = np.asarray(self.st_model.encode(query, normalize_embeddings=True))
        hits = self._neighbors_by_emb(q_emb, k, list(self.memories.keys()))
        out = []
        for cid, sim in hits:
            note = self.memories[cid]
            out.append(
                {
                    "id": cid,
                    "content": note.content,
                    "distance": 1.0 - sim,
                    "keywords": list(note.keywords),
                    "tags": list(note.tags),
                    "context": note.context,
                    "links": list(note.links),
                    "timestamp": note.timestamp,
                }
            )
        return out

    # ── write-time evolution over the embedded corpus ──────────────────────

    def evolve_corpus(self, k: int) -> None:
        """Run A-Mem write-time evolution; memory *i* sees only memories before it,
        matching vanilla A-Mem's incremental insertion order."""
        processed: list[str] = []
        for note in list(self.memories.values()):
            if processed:
                self._evolve_one(note, processed, k)
                self.evo_cnt += 1
            processed.append(note.id)

    def _evolve_one(self, note: MemoryNote, candidates: list[str], k: int) -> None:
        hits = self._neighbors_by_emb(self._emb.get(note.id), k, candidates)
        neighbor_ids = [cid for cid, _ in hits]
        if not neighbor_ids:
            return
        prompt = PROMPT_EVOLUTION.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=self._neighbor_block(neighbor_ids),
            neighbor_number=len(neighbor_ids),
        )
        try:
            resp = nim.chat(prompt)
        except Exception as e:
            log.error("evolve LLM call failed: %s", e)
            return
        if not resp.get("should_evolve", False):
            return
        self._apply_evolution(note, resp, neighbor_ids)


class AMemLateChunk(AMem):
    name = "amem_lc"
    label = "A-Mem (late chunking)"
    DEFAULT_PARAMS = {
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_k": 5,
        "evo_threshold": 100,
        "late_chunk_max_tokens": 512,
    }

    @property
    def engine(self) -> _LateChunkEngine:
        if self._engine is None:
            self._engine = _LateChunkEngine(
                model_name=self.params["embedding_model"],
                evo_threshold=int(self.params["evo_threshold"]),
                max_tokens=int(self.params["late_chunk_max_tokens"]),
            )
        return self._engine

    def construct_memory(self, statements: list[Statement]) -> None:
        k = int(self.params["retrieval_k"])
        eng = self.engine

        # Phase 1 — analyse each memory and create its note (no embedding yet).
        for s in statements:
            analysis = eng.analyze_content(s.statement)
            eng.memories[s.id] = MemoryNote(
                id=s.id,
                content=s.statement,
                keywords=analysis["keywords"],
                context=analysis["context"],
                tags=analysis["tags"],
                timestamp=s.timestamp,
            )

        # Phase 2 — late-chunk embed the whole corpus in one pass.
        eng.embed_corpus()

        # Phase 3 — write-time evolution (edits tags/context/links only).
        eng.evolve_corpus(k)

        for i, s in enumerate(statements, 1):
            note = eng.memories[s.id]
            log.info(
                "[%d/%d] %s tags=%s links=%d evolved_neighbors=%d",
                i,
                len(statements),
                s.id,
                note.tags,
                len(note.links),
                len(note.evolution_history),
            )
