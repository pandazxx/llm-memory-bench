"""Tests for late chunking (pure numpy, no torch) + amem_lc registration."""

import numpy as np

from llm_memory_bench.systems import get_system
from llm_memory_bench.systems._late_chunk import late_chunk_embed, pack_windows

CLS, SEP = 100, 101
# Token id -> deterministic dense vector, so pooled spans are predictable.
VOCAB = {
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.0, 0.0, 1.0],
    4: [1.0, 1.0, 0.0],
    5: [0.0, 1.0, 1.0],
    CLS: [9.0, 9.0, 9.0],
    SEP: [-9.0, -9.0, -9.0],
}


def _tokenize(text):
    return [int(t) for t in text.split()]


def _forward(input_ids):
    return np.array([VOCAB[i] for i in input_ids], dtype=np.float64)


def _expected(token_ids):
    vec = np.mean([VOCAB[i] for i in token_ids], axis=0)
    return vec / np.linalg.norm(vec)


def test_pack_windows_groups_under_budget():
    # lengths 2,2,3 with budget 4 -> [2,2] then [3].
    assert pack_windows([2, 2, 3], budget=4) == [[0, 1], [2]]
    # everything fits in one window.
    assert pack_windows([1, 1, 1], budget=10) == [[0, 1, 2]]
    # each chunk already at budget -> one per window.
    assert pack_windows([4, 4], budget=4) == [[0], [1]]


def test_late_chunk_pools_each_span_and_normalizes():
    contents = ["1 2", "3", "4 5"]
    embs = late_chunk_embed(
        contents,
        tokenize=_tokenize,
        forward=_forward,
        cls_id=CLS,
        sep_id=SEP,
        max_tokens=6,  # budget 4: all three pack into one window
    )
    assert len(embs) == 3
    for e in embs:
        assert np.isclose(np.linalg.norm(e), 1.0)
    # Pooling must cover exactly each memory's own tokens (cls/sep excluded).
    assert np.allclose(embs[0], _expected([1, 2]))
    assert np.allclose(embs[1], _expected([3]))
    assert np.allclose(embs[2], _expected([4, 5]))


def test_late_chunk_spans_independent_of_window_packing():
    # Same contents, tight budget forces one window per memory; pooled vectors
    # (which span only a memory's own tokens) must be unchanged.
    contents = ["1 2", "3", "4 5"]
    packed = late_chunk_embed(
        contents, tokenize=_tokenize, forward=_forward, cls_id=CLS, sep_id=SEP, max_tokens=6
    )
    split = late_chunk_embed(
        contents, tokenize=_tokenize, forward=_forward, cls_id=CLS, sep_id=SEP, max_tokens=4
    )
    for a, b in zip(packed, split, strict=True):
        assert np.allclose(a, b)


def test_truncation_respects_budget():
    # max_tokens=3 -> budget 1: only the first token of each content survives.
    embs = late_chunk_embed(
        ["1 2 3"], tokenize=_tokenize, forward=_forward, cls_id=CLS, sep_id=SEP, max_tokens=3
    )
    assert np.allclose(embs[0], _expected([1]))


def test_amem_lc_registered_without_torch():
    cls = get_system("amem_lc")
    assert cls.name == "amem_lc"
    assert "late_chunk_max_tokens" in cls.DEFAULT_PARAMS
