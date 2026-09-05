"""Run the page's ACTUAL JavaScript and compare it to scikit-learn.

test_export_web.py checks a Python mirror of the browser arithmetic. That is only
worth something if the mirror matches the real thing, so this test extracts the
scoring functions straight out of docs/index.html, executes them in Node, and
compares the numbers to the model's own.

Skipped when Node is unavailable, so the suite still runs anywhere.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.export_web import export
from src.predict import score
from src.train import train

from .conftest import make_frame

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def extract_script(html: str) -> str:
    """Pull the page's inline script, minus the fetch() that needs a browser."""
    body = html.split("<script>", 1)[1].split("</script>", 1)[0]
    return body.split('fetch("model.json")', 1)[0]


@pytest.fixture
def prepared(tmp_path):
    train(artifacts=tmp_path, frame=make_frame(n=2000, seed=51))
    bundle = export(tmp_path, tmp_path / "model.json")
    return bundle, tmp_path


def test_page_javascript_matches_sklearn(prepared, tmp_path):
    bundle, artifacts = prepared
    rows = make_frame(n=40, seed=61).drop(columns=["isFraud"])
    expected = score(rows, artifacts)["fraud_score"].tolist()

    payload = [
        {
            "type": row["type"],
            "step": float(row["step"]),
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
        }
        for _, row in rows.iterrows()
    ]

    script = tmp_path / "parity.mjs"
    script.write_text(
        extract_script(PAGE.read_text(encoding="utf8"))
        + textwrap.dedent(
            f"""
            M = {json.dumps(bundle)};
            const rows = {json.dumps(payload)};
            console.log(JSON.stringify(rows.map(r => score(r).probability)));
            """
        )
    )

    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=True
    )
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_page_javascript_handles_an_unknown_channel(prepared, tmp_path):
    """The page must degrade exactly as handle_unknown="ignore" does."""
    bundle, artifacts = prepared
    rows = make_frame(n=5, seed=62).drop(columns=["isFraud"])
    rows["type"] = "CRYPTO_OUT"
    expected = score(rows, artifacts)["fraud_score"].tolist()

    payload = [
        {
            "type": "CRYPTO_OUT",
            "step": float(row["step"]),
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
        }
        for _, row in rows.iterrows()
    ]

    script = tmp_path / "parity_unknown.mjs"
    script.write_text(
        extract_script(PAGE.read_text(encoding="utf8"))
        + textwrap.dedent(
            f"""
            M = {json.dumps(bundle)};
            const rows = {json.dumps(payload)};
            console.log(JSON.stringify(rows.map(r => score(r).probability)));
            """
        )
    )
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=True
    )
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
