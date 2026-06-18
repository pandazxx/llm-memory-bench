"""Late chunking (Günther et al. 2024) as a pure, backend-agnostic routine.

Instead of embedding each chunk on its own, we concatenate several chunks into a
single transformer window so their tokens attend across chunk boundaries, then
mean-pool each chunk's contextualized token span to get that chunk's embedding.

The transformer is injected as two callables (``tokenize`` + ``forward``) so this
module stays free of torch/transformers and is testable with numpy fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def pack_windows(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Greedily group chunk indices into windows whose token totals fit ``budget``.

    ``lengths`` must already be truncated to ``budget`` so every chunk fits in a
    window on its own."""
    windows: list[list[int]] = []
    cur: list[int] = []
    cur_sum = 0
    for i, length in enumerate(lengths):
        length = min(length, budget)
        if cur and cur_sum + length > budget:
            windows.append(cur)
            cur, cur_sum = [], 0
        cur.append(i)
        cur_sum += length
    if cur:
        windows.append(cur)
    return windows


def late_chunk_embed(
    contents: Sequence[str],
    *,
    tokenize: Callable[[str], list[int]],
    forward: Callable[[list[int]], np.ndarray],
    cls_id: int,
    sep_id: int,
    max_tokens: int,
) -> list[np.ndarray]:
    """Embed every content via late chunking, returning one L2-normalized vector each.

    ``tokenize(text)`` yields token ids without special tokens; ``forward(input_ids)``
    runs the encoder over ``[cls] + ... + [sep]`` and returns the ``[T, D]`` matrix of
    contextualized token embeddings (one row per input position)."""
    budget = max(1, int(max_tokens) - 2)
    token_lists = [list(tokenize(c))[:budget] for c in contents]
    lengths = [len(t) for t in token_lists]
    windows = pack_windows(lengths, budget)

    embeddings: list[np.ndarray | None] = [None] * len(contents)
    for window in windows:
        input_ids = [cls_id]
        spans: list[tuple[int, int, int]] = []  # (content_idx, start, end)
        for ci in window:
            start = len(input_ids)
            input_ids.extend(token_lists[ci])
            spans.append((ci, start, len(input_ids)))
        input_ids.append(sep_id)

        hidden = np.asarray(forward(input_ids), dtype=np.float64)
        dim = hidden.shape[1]
        for ci, start, end in spans:
            vec = hidden[start:end].mean(axis=0) if end > start else np.zeros(dim)
            norm = float(np.linalg.norm(vec))
            embeddings[ci] = vec / norm if norm > 0 else vec

    return [e if e is not None else np.zeros(1) for e in embeddings]
