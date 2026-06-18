"""parse_json robustness against the messy JSON LLMs emit."""

from llm_memory_bench.backend import nim


def test_plain_json():
    assert nim.parse_json('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_markdown_fenced():
    assert nim.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_line_comments_and_trailing_commas():
    raw = """{
        "keywords": [
            // several specific keywords
        ],
        "context": "a sentence", // summary
        "tags": ["x", "y",]
    }"""
    assert nim.parse_json(raw) == {"keywords": [], "context": "a sentence", "tags": ["x", "y"]}


def test_comment_marker_inside_string_is_preserved():
    # The inline comment forces the strip fallback; the // inside the URL must survive.
    raw = '{"url": "https://example.com/x", // see here\n "n": 1}'
    assert nim.parse_json(raw) == {"url": "https://example.com/x", "n": 1}
