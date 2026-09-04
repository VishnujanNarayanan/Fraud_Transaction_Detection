"""Tests for the SQLite analytics layer.

Every query in queries.sql is executed against a small synthetic table. That catches
the failure this layer is most prone to -- a query that is syntactically fine but
references a column the table does not have -- without needing the 470 MB CSV.
"""

import sqlite3

import pandas as pd
import pytest

from src import db

from .conftest import make_frame


@pytest.fixture
def database(tmp_path, frame):
    path = tmp_path / "test.db"
    with sqlite3.connect(path) as conn:
        frame.to_sql(db.TABLE, conn, index=False)
    return path


def test_every_query_parses_and_runs(database):
    results = db.run_all(database)
    assert set(results) == set(db.load_queries())
    for name, result in results.items():
        assert isinstance(result, pd.DataFrame), name


def test_queries_are_named_and_non_empty():
    """Each named body must be a real statement once its leading comment is stripped."""
    queries = db.load_queries()
    assert len(queries) >= 9
    for name, sql in queries.items():
        body = "\n".join(
            line for line in sql.splitlines() if not line.strip().startswith("--")
        ).strip()
        assert body.upper().startswith(("SELECT", "WITH")), name
        assert body.endswith(";"), name


def test_unknown_query_name_raises(database):
    with pytest.raises(KeyError):
        db.run("no_such_query", database)


def test_row_count_matches_the_source_frame(database, frame):
    result = db.run("row_count", database)
    assert result.loc[0, "transactions"] == len(frame)
    assert result.loc[0, "fraud_rows"] == frame["isFraud"].sum()


def test_fraud_by_type_agrees_with_pandas(database, frame):
    """The SQL cut and the pandas cut must give the same answer, or one of them is wrong."""
    sql_result = db.run("fraud_by_type", database).set_index("type")["fraud_rows"]
    pandas_result = frame.groupby("type")["isFraud"].sum()
    pd.testing.assert_series_equal(
        sql_result.sort_index().astype(int),
        pandas_result.sort_index().astype(int),
        check_names=False,
    )


def test_contingency_rows_sum_to_the_channel_total(database, frame):
    result = db.run("contingency_type_fraud", database)
    totals = (result["not_fraud"] + result["fraud"]).sum()
    assert totals == len(frame)


def test_fraud_by_hour_covers_only_real_clock_hours(database):
    hours = db.run("fraud_by_hour", database)["hour_of_day"]
    assert hours.between(0, 23).all()


def test_balance_anomaly_query_matches_the_engineered_flag(database, frame):
    """queries.sql and preprocessor.py must define suspicious_flag identically."""
    sql_result = db.run("balance_anomaly_rate", database).set_index("suspicious_flag")
    expected = (
        ((frame["amount"] > 0) & (frame["newbalanceDest"] == frame["oldbalanceDest"]))
        .astype(int)
        .value_counts()
    )
    for flag, count in expected.items():
        assert sql_result.loc[flag, "transactions"] == count


def test_amount_percentiles_are_monotonic(database):
    result = db.run("amount_percentiles", database)
    for _, row in result.iterrows():
        values = [row["p25"], row["p50"], row["p75"], row["p95"], row["p99"]]
        assert values == sorted(values)


def test_build_reports_a_missing_csv_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        db.build(tmp_path / "nope.csv", tmp_path / "out.db")


def test_build_loads_a_csv_end_to_end(tmp_path):
    csv = tmp_path / "Fraud.csv"
    make_frame(n=120, seed=11).to_csv(csv, index=False)
    path = db.build(csv, tmp_path / "built.db")
    assert db.run("row_count", path).loc[0, "transactions"] == 120
