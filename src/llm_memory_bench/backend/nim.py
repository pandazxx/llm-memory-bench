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

from openai import OpenAI, RateLimitError

log = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("NIM_LLM_MODEL", "meta/llama-3.1-70b-instruct")
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


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first {...} object from raw LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"no JSON object found in LLM response:\n{text[:200]}")


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
