"""Shared evaluation library for Ask Parliament.

Small, dependency-light helpers used by the eval scripts (`retrieval_eval.py`,
`generation_eval.py`, `build_golden_set.py`, `report.py`) and exercised directly
by the unit tests in `eval/tests/`. Kept import-light on purpose: the metric and
judging code must run with no Qdrant server and no API key, so regressions in the
maths are caught fast and offline.
"""
