"""Rebuild fixtures/analyst/ridgeline.db (deterministic seed).

    uv run python fixtures/analyst/generate.py
"""

from langchain_agents.analyst.fixture_db import build_db
from langchain_agents.analyst.db import default_db_path

if __name__ == "__main__":
    path = build_db(default_db_path())
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
