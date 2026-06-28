"""Offline tests for the LLM-judge response parsing/clamping — no API calls.

The judge talks to the model, but the JSON extraction and score clamping are pure and
must be robust to fenced code blocks, trailing prose, and out-of-range scores."""
from evallib.llm_judge import _clamp, _parse_json


def test_parse_plain_json():
    assert _parse_json('{"groundedness": 5}') == {"groundedness": 5}


def test_parse_fenced_json():
    txt = "```json\n{\"a\": 1, \"b\": 2}\n```"
    assert _parse_json(txt) == {"a": 1, "b": 2}


def test_parse_json_with_surrounding_prose():
    txt = "Here is my judgment:\n{\"groundedness\": 4, \"comment\": \"ok\"}\nThanks!"
    assert _parse_json(txt)["groundedness"] == 4


def test_parse_garbage_returns_empty():
    assert _parse_json("no json here at all") == {}
    assert _parse_json("") == {}


def test_clamp_into_range():
    assert _clamp(5) == 5
    assert _clamp(9) == 5      # above range -> capped
    assert _clamp(0) == 1      # below range -> floored
    assert _clamp("3") == 3    # string number coerced
    assert _clamp(None) is None
    assert _clamp("nonsense") is None
