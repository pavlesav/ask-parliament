"""Unit tests for the pattern-based relevance judge — deterministic, offline."""
from evallib.judge import compile_item, grade_hits, relevance


def _item(**over):
    base = {
        "id": "t",
        "question": "q",
        "patterns": ["ukrain", "invasion|aggression|war"],
    }
    base.update(over)
    return compile_item(base)


def test_requires_all_patterns():
    it = _item()
    assert relevance("the invasion of Ukraine was condemned", "AT", "2022-03-01", it) == 1.0
    # Missing the second pattern group -> not relevant.
    assert relevance("Ukraine joined the trade talks", "AT", "2022-03-01", it) == 0.0


def test_case_insensitive_and_crosslingual_text():
    it = _item()
    # Native-language text still matches because the pattern stems are language-neutral.
    assert relevance("Die INVASION der Ukraine", "AT", None, it) == 1.0


def test_strong_pattern_promotes_to_grade_2():
    it = _item(strong_patterns=["invasion of ukraine"])
    assert relevance("the invasion of Ukraine", "GB", None, it) == 2.0
    assert relevance("war in ukraine territory", "GB", None, it) == 1.0


def test_country_constraint():
    it = _item(countries=["AT", "DE"])
    assert relevance("invasion of ukraine", "AT", None, it) == 1.0
    assert relevance("invasion of ukraine", "GB", None, it) == 0.0


def test_date_window():
    it = _item(date_from="2022-02-24", date_to="2022-12-31")
    assert relevance("invasion of ukraine", "AT", "2022-03-01", it) == 1.0
    assert relevance("invasion of ukraine", "AT", "2021-01-01", it) == 0.0
    # A constrained item rejects undated speeches (can't confirm the window).
    assert relevance("invasion of ukraine", "AT", None, it) == 0.0


def test_grade_hits_over_objects_and_dicts():
    it = _item()
    class S:
        def __init__(self, text, country, date):
            self.text, self.country, self.date = text, country, date
    hits = [
        S("invasion of ukraine", "AT", "2022-03-01"),
        {"text": "unrelated budget debate", "country": "AT", "date": "2022-03-01"},
        S("war in ukraine", "FR", "2022-04-01"),
    ]
    assert grade_hits(hits, it) == [1.0, 0.0, 1.0]
