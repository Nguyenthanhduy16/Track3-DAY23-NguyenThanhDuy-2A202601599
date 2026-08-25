"""Pytest configuration.

Load .env before any test module is collected, so skipif conditions that check
os.getenv(...) (e.g. test_graph_smoke.py) see the configured API keys.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
