"""A-Mem (Agentic Memory) ported onto the framework's System interface.

Reproduces Xu et al. 2025 (arXiv 2502.12110). Each memory is an atomic
``MemoryNote``; three operations run per new memory: note construction (LLM
extracts keywords/tags/context), link generation + write-time evolution of
neighbours, and agentic retrieval (cosine + 1-hop link traversal + reader LLM).

LLM = NVIDIA NIM; embeddings = local sentence-transformers via ChromaDB
(kept internal to this system for Milestone 1).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..backend import nim
from ..core.models import Query, QueryResult, Statement
from ..core.system import System
from ..viz import memory as memory_viz
from ..viz import retrieval as retrieval_viz

log = logging.getLogger(__name__)


PROMPT_NOTE_CONSTRUCTION = """\
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Respond with ONLY a valid JSON object — no comments, no markdown fences, no
trailing commas — of exactly this shape:
{{"keywords": ["keyword1", "keyword2"], "context": "one sentence", "tags": ["tag1", "tag2"]}}

Where:
- "keywords": several specific, distinct keywords, ordered most to least important.
- "context": one sentence summarizing the main topic, key points, and purpose.
- "tags": several broad categories or themes for classification.

Content for analysis:
{content}"""

PROMPT_EVOLUTION = """\
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context: {context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. What specific actions should be taken (strengthen, update_neighbor)?
   1.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   1.2 If choose to update neighbor, you can update the context and tags of these memories based on the understanding of these memories.

Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true or false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1", "tag_n"],
    "new_context_neighborhood": ["new context", "new context"],
    "new_tags_neighborhood": [["tag_1", "tag_n"], ["tag_1", "tag_n"]]
}}"""


class MemoryNote:
    """One atomic Zettelkasten note."""

    def __init__(
        self,
        content: str,
        *,
        id: str | None = None,
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
        context: str = "General",
        links: list[str] | None = None,
        timestamp: str | None = None,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.tags = tags or []
        self.context = context
        self.links: list[str] = links or []
        now = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or now
        self.last_accessed = now
        self.retrieval_count = 0
        self.evolution_history: list[dict] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "keywords": list(self.keywords),
            "tags": list(self.tags),
            "context": self.context,
            "links": list(self.links),
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "retrieval_count": self.retrieval_count,
            "evolution_history": list(self.evolution_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryNote:
        note = cls(
            content=d["content"],
            id=d["id"],
            keywords=d.get("keywords", []),
            tags=d.get("tags", []),
            context=d.get("context", "General"),
            links=d.get("links", []),
            timestamp=d.get("timestamp"),
        )
        note.last_accessed = d.get("last_accessed", note.last_accessed)
        note.retrieval_count = d.get("retrieval_count", 0)
        note.evolution_history = list(d.get("evolution_history", []))
        return note


def _try_json_load(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return v


class _Engine:
    """Core A-Mem engine: in-memory note map + ChromaDB cosine index."""

    def __init__(self, model_name: str, evo_threshold: int):
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        self.memories: dict[str, MemoryNote] = {}
        self.model_name = model_name
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
        try:
            self.client.reset()
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="memories", embedding_function=self.embedding_fn
        )

    def analyze_content(self, content: str) -> dict[str, Any]:
        prompt = PROMPT_NOTE_CONSTRUCTION.format(content=content)
        try:
            result = nim.chat(prompt)
            return {
                "keywords": result.get("keywords", []),
                "context": result.get("context", "General"),
                "tags": result.get("tags", []),
            }
        except Exception as e:
            log.error("analyze_content failed: %s", e)
            return {"keywords": [], "context": "General", "tags": []}

    def _neighbor_block(self, ids: list[str]) -> str:
        """Format neighbour notes into the text block the evolution prompt expects."""
        parts = []
        for i, doc_id in enumerate(ids):
            note = self.memories.get(doc_id)
            if not note:
                continue
            parts.append(
                f"Memory {i + 1} (ID: {doc_id}):\n"
                f"  content: {note.content}\n"
                f"  keywords: {note.keywords}\n"
                f"  tags: {note.tags}\n"
                f"  context: {note.context}"
            )
        return "\n\n".join(parts)

    def _find_related(self, content: str, k: int) -> tuple[str, list[str]]:
        count = self.collection.count()
        if count == 0:
            return "", []
        results = self.collection.query(query_texts=[content], n_results=min(k, count))
        if not results or not results.get("ids") or not results["ids"][0]:
            return "", []
        ids = [doc_id for doc_id in results["ids"][0] if doc_id in self.memories]
        return self._neighbor_block(ids), ids

    def _apply_evolution(self, note: MemoryNote, resp: dict, neighbor_ids: list[str]) -> None:
        """Apply the LLM's evolution decision: strengthen links / rewrite neighbours."""
        for action in resp.get("actions", []):
            if action == "strengthen":
                note.links.extend(resp.get("suggested_connections", []))
                new_tags = resp.get("tags_to_update", [])
                if new_tags:
                    note.tags = new_tags
            elif action == "update_neighbor":
                new_contexts = resp.get("new_context_neighborhood", [])
                new_tags_all = resp.get("new_tags_neighborhood", [])
                for i, nid in enumerate(neighbor_ids):
                    neighbor = self.memories.get(nid)
                    if not neighbor:
                        continue
                    if i < len(new_contexts):
                        old = neighbor.context
                        neighbor.context = new_contexts[i]
                        neighbor.evolution_history.append(
                            {
                                "trigger": note.id,
                                "field": "context",
                                "old": old,
                                "new": new_contexts[i],
                            }
                        )
                    if i < len(new_tags_all):
                        old = neighbor.tags
                        neighbor.tags = new_tags_all[i]
                        neighbor.evolution_history.append(
                            {
                                "trigger": note.id,
                                "field": "tags",
                                "old": old,
                                "new": new_tags_all[i],
                            }
                        )

    def process_memory(self, note: MemoryNote, k: int) -> bool:
        if not self.memories:
            return False
        neighbors_text, neighbor_ids = self._find_related(note.content, k)
        if not neighbors_text or not neighbor_ids:
            return False
        prompt = PROMPT_EVOLUTION.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbors_text,
            neighbor_number=len(neighbor_ids),
        )
        try:
            resp = nim.chat(prompt)
        except Exception as e:
            log.error("process_memory LLM call failed: %s", e)
            return False
        if not resp.get("should_evolve", False):
            return False
        self._apply_evolution(note, resp, neighbor_ids)
        return True

    def _index(self, note: MemoryNote) -> None:
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "tags": json.dumps(note.tags),
            "context": note.context,
            "links": json.dumps(note.links),
            "timestamp": note.timestamp,
        }
        self.collection.add(documents=[note.content], metadatas=[metadata], ids=[note.id])

    def _consolidate(self) -> None:
        try:
            self.client.reset()
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="memories", embedding_function=self.embedding_fn
        )
        for note in self.memories.values():
            self._index(note)

    def add_note(self, content: str, k: int, timestamp: str | None, id: str | None) -> str:
        analysis = self.analyze_content(content)
        note = MemoryNote(
            id=id,
            content=content,
            keywords=analysis["keywords"],
            context=analysis["context"],
            tags=analysis["tags"],
            timestamp=timestamp,
        )
        if self.process_memory(note, k):
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self._consolidate()
        self.memories[note.id] = note
        self._index(note)
        return note.id

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        if not results or not results.get("ids"):
            return []
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            out.append(
                {
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    **{kk: _try_json_load(vv) for kk, vv in meta.items()},
                }
            )
        return out

    def read(self, question: str, k: int) -> dict[str, Any]:
        direct = self.search(question, k)
        if not direct:
            return {
                "answer": "I don't have any information about that.",
                "retrieved_ids": [],
                "links_followed": [],
                "seed_ids": [],
            }
        seen = {r["id"] for r in direct}
        seed_ids = [r["id"] for r in direct]
        links_followed: list[tuple[str, str]] = []
        bundle = list(direct)
        for r in direct:
            note = self.memories.get(r["id"])
            if not note:
                continue
            note.retrieval_count += 1
            for link_id in note.links:
                if link_id in seen:
                    continue
                seen.add(link_id)
                linked = self.memories.get(link_id)
                if linked:
                    links_followed.append((r["id"], link_id))
                    bundle.append(
                        {
                            "id": linked.id,
                            "content": linked.content,
                            "timestamp": linked.timestamp,
                            "via_link_from": r["id"],
                        }
                    )

        context = "\n".join(
            f"- [{b['id']}] (ts={b.get('timestamp', '?')}) {b['content']}" for b in bundle
        )
        prompt = (
            "You are answering a question using only the memories below. "
            "If multiple memories contradict, prefer the most recent one (later timestamp). "
            'If the answer is not contained in the memories, reply exactly: "unknown / not mentioned".\n\n'
            f"Memories:\n{context}\n\nQuestion: {question}\n\n"
            "Answer concisely in one short sentence."
        )
        answer = nim.chat_text(
            prompt,
            system="You are a precise question-answering assistant. Use only the provided memories.",
        )
        return {
            "answer": answer,
            "retrieved_ids": [b["id"] for b in bundle],
            "links_followed": links_followed,
            "seed_ids": seed_ids,
        }


class AMem(System):
    name = "amem"
    label = "A-Mem"
    DEFAULT_PARAMS = {
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval_k": 5,
        "evo_threshold": 100,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._engine: _Engine | None = None

    @property
    def engine(self) -> _Engine:
        if self._engine is None:
            self._engine = _Engine(
                model_name=self.params["embedding_model"],
                evo_threshold=int(self.params["evo_threshold"]),
            )
        return self._engine

    def construct_memory(self, statements: list[Statement]) -> None:
        k = int(self.params["retrieval_k"])
        for i, s in enumerate(statements, 1):
            self.engine.add_note(content=s.statement, k=k, timestamp=s.timestamp, id=s.id)
            note = self.engine.memories[s.id]
            log.info(
                "[%d/%d] %s tags=%s links=%d evolved_neighbors=%d",
                i,
                len(statements),
                s.id,
                note.tags,
                len(note.links),
                len(note.evolution_history),
            )

    def retrieve(self, query: Query) -> QueryResult:
        resp = self.engine.read(query.query, k=int(self.params["retrieval_k"]))
        return QueryResult(
            query_id=query.id,
            retrieved_ids=resp["retrieved_ids"],
            answer=resp["answer"],
            trail={
                "seed_ids": resp.get("seed_ids", []),
                "links_followed": resp.get("links_followed", []),
            },
        )

    # ── visualization + state ────────────────────────────────────────────

    def _state(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "label": self.label,
            "params": self.params,
            "memories": {mid: n.to_dict() for mid, n in self.engine.memories.items()},
        }

    def dump_memory_state(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "state.json").write_text(json.dumps(self._state(), indent=2))

    def render_memory(self, out_dir: Path) -> None:
        memory_viz.render(self._state(), out_dir)

    def render_retrieval(
        self, results: list[QueryResult], out_dir: Path, scores: list[dict] | None = None
    ) -> None:
        retrieval_viz.render(self._state(), results, out_dir, scores)

    @classmethod
    def render_from_state(cls, state_dir: Path, out_dir: Path) -> None:
        state = json.loads((state_dir / "state.json").read_text())
        memory_viz.render(state, out_dir)

    @classmethod
    def render_retrieval_from_state(
        cls,
        state_dir: Path,
        results: list[QueryResult],
        out_dir: Path,
        scores: list[dict] | None = None,
    ) -> None:
        state = json.loads((state_dir / "state.json").read_text())
        retrieval_viz.render(state, results, out_dir, scores)
