"""Shared pieces for the HippoRAG adapters: reader LLM + Personalized PageRank."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..backend import nim

_READER_SYSTEM = "You are a precise question-answering assistant. Use only the provided passages."


def read(question: str, passages: list[tuple[str, str]]) -> str:
    """Reader LLM: synthesise a one-sentence answer from retrieved passages."""
    if not passages:
        return "unknown / not mentioned"
    ctx = "\n".join(f"- [{pid}] {content}" for pid, content in passages)
    prompt = (
        "Answer the question using only the passages below.\n"
        'If the answer is not contained in them, reply exactly: "unknown / not mentioned".\n\n'
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in one short sentence."
    )
    return nim.chat_text(prompt, system=_READER_SYSTEM)


def _transition(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.array(adj.sum(axis=1), dtype=np.float64).flatten()
    row_sums[row_sums == 0] = 1.0
    return sp.diags(1.0 / row_sums) @ adj


def _power_iterate(
    T: sp.csr_matrix, s: np.ndarray, alpha: float, max_iter: int, tol: float
) -> np.ndarray:
    r = s.copy()
    for _ in range(max_iter):
        r_new = (1.0 - alpha) * T.T.dot(r) + alpha * s
        if np.linalg.norm(r_new - r, 1) < tol:
            return r_new
        r = r_new
    return r


def ppr_from_indices(
    adj: sp.csr_matrix,
    seed_indices: list[int],
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Personalized PageRank seeded by a uniform distribution over node indices (v1)."""
    N = adj.shape[0]
    s = np.zeros(N, dtype=np.float64)
    if seed_indices:
        s[seed_indices] = 1.0 / len(seed_indices)
    return _power_iterate(_transition(adj), s, alpha, max_iter, tol)


def ppr_from_weights(
    adj: sp.csr_matrix,
    seeds: np.ndarray,
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Personalized PageRank with arbitrary non-negative seed weights (v2)."""
    N = adj.shape[0]
    s = seeds / seeds.sum() if seeds.sum() > 0 else np.ones(N) / N
    return _power_iterate(_transition(adj), s, alpha, max_iter, tol)
