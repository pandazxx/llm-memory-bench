"""NVIDIA NIM LLM client.

The only wired LLM provider. All calls go through ``call()`` which retries
**indefinitely** on HTTP 429 with capped exponential backoff (the free NIM
tier rate-limits aggressively; we never give up).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import numpy as np
from openai import OpenAI, RateLimitError

log = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("NIM_LLM_MODEL", "meta/llama-3.1-70b-instruct")
EMBED_MODEL = os.environ.get("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

# 429 backoff: start at 1s, double each retry, cap at 30s. Never stop retrying.
BACKOFF_START = 1.0
BACKOFF_CAP = 30.0

_client: OpenAI | None = None


def nim() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set — export your NIM key before running.")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def call(fn, *args, **kwargs):
    """Invoke ``fn`` retrying indefinitely on 429 with capped exponential backoff."""
    delay = BACKOFF_START
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            log.warning("429 rate-limited — retrying in %.1fs", delay)
            time.sleep(delay)
            delay = min(delay * 2, BACKOFF_CAP)


def _strip_json_comments(s: str) -> str:
    """Drop ``//`` and ``/* */`` comments that aren't inside a JSON string.

    LLMs frequently echo the comment placeholders from a schema example; this is
    string-aware so it won't corrupt values like ``"https://..."``."""
    out: list[str] = []
    i, n = 0, len(s)
    in_str = escape = False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first {...} object from raw LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"no JSON object found in LLM response:\n{text[:200]}")
    blob = match.group()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        cleaned = _strip_json_comments(blob)
        # Comment removal can leave a trailing comma before a } or ]; drop those.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        return json.loads(cleaned)


def chat(
    prompt: str,
    *,
    system: str = "You must respond with a JSON object.",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    resp = call(
        nim().chat.completions.create,
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_json(resp.choices[0].message.content)


def chat_text(
    prompt: str,
    *,
    system: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Free-text chat (no JSON parsing). Used by the QA reader."""
    resp = call(
        nim().chat.completions.create,
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _chat_raw(messages: list[dict], *, temperature: float = 0.0) -> str:
    """Few-shot chat with a full message list; returns raw text content."""
    resp = call(
        nim().chat.completions.create,
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ── Embeddings (HippoRAG v1 + v2) ──────────────────────────────────────────


def embed_batch(texts: list[str], input_type: str = "passage") -> np.ndarray:
    """Embed N strings; returns (N, dim) L2-normalised float64 array.

    ``input_type`` is "passage" for index-side text, "query" for lookups —
    required by NIM's asymmetric embedding models.
    """
    response = call(
        nim().embeddings.create,
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": input_type},
    )
    vecs = np.array(
        [d.embedding for d in sorted(response.data, key=lambda x: x.index)],
        dtype=np.float64,
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed(text: str, input_type: str = "query") -> np.ndarray:
    return embed_batch([text], input_type=input_type)[0]


# ── Passage NER + OpenIE (HippoRAG) ────────────────────────────────────────
# Prompts are verbatim from OSU-NLP-Group/HippoRAG (identical in HippoRAG 2's
# ner / triple_extraction templates).

_ONE_SHOT_PASSAGE = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music "
    "portal - PlanetRadiocity.com that offers music related news, videos, songs, and "
    "other music-related features."
)

_ONE_SHOT_ENTITIES = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

_ONE_SHOT_TRIPLES = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

_NER_SYSTEM = (
    "Your task is to extract named entities from the given paragraph. \n"
    "Respond with a JSON list of entities.\n"
)

_OPENIE_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the "
    "given passages and named entity lists. \n"
    "Respond with a JSON list of triples, with each triple representing a relationship "
    "in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities "
    "in the list for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n"
)

_OPENIE_FRAME = (
    "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
    "Paragraph:\n"
    "```\n"
    "{passage}\n"
    "```\n\n"
    "{named_entity_json}\n"
)


def _ner_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": _NER_SYSTEM},
        {"role": "user", "content": f"Paragraph:\n```\n{_ONE_SHOT_PASSAGE}\n```\n"},
        {"role": "assistant", "content": _ONE_SHOT_ENTITIES},
        {"role": "user", "content": f"Paragraph:```\n{text}\n```"},
    ]


def _openie_messages(passage: str, entities: list[str]) -> list[dict]:
    one_shot_input = _OPENIE_FRAME.format(
        passage=_ONE_SHOT_PASSAGE, named_entity_json=_ONE_SHOT_ENTITIES
    )
    user_input = _OPENIE_FRAME.format(
        passage=passage, named_entity_json=json.dumps({"named_entities": entities})
    )
    return [
        {"role": "system", "content": _OPENIE_SYSTEM},
        {"role": "user", "content": one_shot_input},
        {"role": "assistant", "content": _ONE_SHOT_TRIPLES},
        {"role": "user", "content": user_input},
    ]


def extract_triples(passage: str) -> list[tuple]:
    """Two-step OpenIE (NER then OpenIE conditioned on those entities)."""
    try:
        entities = parse_json(_chat_raw(_ner_messages(passage))).get("named_entities", [])
    except (json.JSONDecodeError, ValueError):
        entities = []

    try:
        data = parse_json(_chat_raw(_openie_messages(passage, entities)))
        return [tuple(t[:3]) for t in data.get("triples", []) if len(t) >= 3]
    except (json.JSONDecodeError, ValueError):
        return []


# ── Query NER (HippoRAG v1) ────────────────────────────────────────────────
# Broader than the original — also extracts key concepts, not just NEs.

_QUERY_NER_SYSTEM = (
    "Your task is to extract named entities and key concepts from a question "
    "that are useful for knowledge-graph retrieval. "
    "Include person names, organisations, places, and domain concepts. "
    'Respond with a JSON object: {"named_entities": ["...", ...]}\n'
)

_QUERY_NER_ONE_SHOT_Q = "Who founded the company that makes the iPhone?"
_QUERY_NER_ONE_SHOT_A = '{"named_entities": ["iPhone", "company"]}'


def _query_ner_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _QUERY_NER_SYSTEM},
        {"role": "user", "content": _QUERY_NER_ONE_SHOT_Q},
        {"role": "assistant", "content": _QUERY_NER_ONE_SHOT_A},
        {"role": "user", "content": question},
    ]


def extract_query_entities(question: str) -> list[str]:
    try:
        return parse_json(_chat_raw(_query_ner_messages(question))).get("named_entities", [])
    except (json.JSONDecodeError, ValueError):
        return []


# ── Recognition memory filter (HippoRAG v2) ────────────────────────────────

_FILTER_SYSTEM = (
    "You are a critical component of a high-stakes question-answering system. "
    "Your task is to filter facts based on their relevance to a given query. "
    "The query may require multi-hop reasoning to connect different pieces of information. "
    "You must select up to 4 relevant facts from the provided candidate list that have a "
    "strong connection to the query, aiding in reasoning and providing an accurate answer. "
    "The output should be in JSON format, e.g., "
    '{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, '
    'and if no facts are relevant, return an empty list, {"fact": []}. '
    "You must only use facts from the candidate list and not generate new facts."
)

_FILTER_ONE_SHOT_Q = "When did the director of film S.O.B. (Film) die?"
_FILTER_ONE_SHOT_BEFORE = json.dumps(
    {
        "fact": [
            ["allan dwan", "died on", "28 december 1981"],
            ["s o b", "written and directed by", "blake edwards"],
            ["robert aldrich", "died on", "december 5 1983"],
            ["robert siodmak", "died on", "10 march 1973"],
            ["bernardo bertolucci", "died on", "26 november 2018"],
        ]
    }
)
_FILTER_ONE_SHOT_AFTER = '{"fact":[["s o b","written and directed by","blake edwards"]]}'


def _filter_messages(question: str, facts_json: str) -> list[dict]:
    return [
        {"role": "system", "content": _FILTER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {_FILTER_ONE_SHOT_Q}\n"
                f"Fact Before Filter: {_FILTER_ONE_SHOT_BEFORE}\nFact After Filter:"
            ),
        },
        {"role": "assistant", "content": _FILTER_ONE_SHOT_AFTER},
        {
            "role": "user",
            "content": (
                f"Question: {question}\nFact Before Filter: {facts_json}\nFact After Filter:"
            ),
        },
    ]


def filter_triples(question: str, triples: list[tuple], top_k: int = 4) -> list[tuple]:
    """HippoRAG 2 recognition memory: LLM keeps the triples relevant to the query."""
    if not triples:
        return []
    facts_json = json.dumps({"fact": [list(t) for t in triples]})
    try:
        data = parse_json(_chat_raw(_filter_messages(question, facts_json)))
        kept = [tuple(t[:3]) for t in data.get("fact", []) if len(t) >= 3]
        return kept[:top_k]
    except (json.JSONDecodeError, ValueError):
        return []
