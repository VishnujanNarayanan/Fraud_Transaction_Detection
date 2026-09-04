"""Load the PaySim CSV into SQLite and run the analytics in src/queries.sql.

Why a database at all, for a file that pandas can read? Two reasons that hold outside
this project too: the aggregations become readable by anyone who knows SQL but not
pandas, and they run against 6.36M rows without holding the frame in memory. The
loader chunks the CSV so the import itself never does either.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Fraud.csv"
DEFAULT_DB = ROOT / "artifacts" / "fraud.db"
QUERIES_PATH = Path(__file__).resolve().parent / "queries.sql"

TABLE = "transactions"
CHUNK_ROWS = 250_000

_NAME_HEADER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def load_queries(path: Path = QUERIES_PATH) -> dict[str, str]:
    """Split queries.sql on its `-- name: <key>` headers into {name: sql}."""
    text = path.read_text(encoding="utf8")
    parts = _NAME_HEADER.split(text)
    # parts == [preamble, name1, body1, name2, body2, ...]
    return {
        name: body.strip()
        for name, body in zip(parts[1::2], parts[2::2], strict=True)
        if body.strip()
    }


def build(csv_path: Path = DEFAULT_CSV, db_path: Path = DEFAULT_DB) -> Path:
    """Load the CSV into SQLite in chunks, then index the columns the queries group on."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download PaySim from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and unzip it in the "
            "project root."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rows = 0
    with sqlite3.connect(db_path) as conn:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_ROWS)):
            chunk.to_sql(
                TABLE, conn, if_exists="replace" if i == 0 else "append", index=False
            )
            rows += len(chunk)
            print(f"  loaded {rows:,} rows", end="\r", flush=True)
        print(f"  loaded {rows:,} rows")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_type ON {TABLE}(type)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_fraud ON {TABLE}(isFraud)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_step ON {TABLE}(step)")
    return db_path


def run(name: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Run one named query from queries.sql and return it as a DataFrame."""
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"unknown query {name!r}; have {sorted(queries)}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(queries[name], conn)


def run_all(db_path: Path = DEFAULT_DB) -> dict[str, pd.DataFrame]:
    """Run every named query. Used by the report below and by the dashboard build."""
    with sqlite3.connect(db_path) as conn:
        return {
            name: pd.read_sql_query(sql, conn) for name, sql in load_queries().items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--build", action="store_true", help="(re)load the CSV first")
    parser.add_argument("--query", help="run one named query instead of all of them")
    args = parser.parse_args()

    if args.build:
        print(f"Building {args.db} from {args.csv}")
        build(args.csv, args.db)

    if args.query:
        print(run(args.query, args.db).to_string(index=False))
        return

    for name, frame in run_all(args.db).items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
