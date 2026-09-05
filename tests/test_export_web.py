"""The browser must compute the same score as scikit-learn.

This is the test that makes the static demo trustworthy. The page reimplements
FraudPreprocessor.transform and the model's sigmoid in JavaScript; if that drifts
from the Python, the demo shows a confident number that the model would not agree
with. The reference implementation in export_web mirrors the JavaScript step for
step, and here it is checked against the real pipeline.
"""

import json

import numpy as np
import pytest

from src.export_web import export, score_like_browser
from src.predict import score
from src.train import train

from .conftest import make_frame


@pytest.fixture
def bundle(tmp_path):
    train(artifacts=tmp_path, frame=make_frame(n=2000, seed=41))
    return export(tmp_path, tmp_path / "model.json"), tmp_path


def test_bundle_has_everything_the_browser_needs(bundle):
    data, _ = bundle
    for key in (
        "feature_columns",
        "coefficients",
        "intercept",
        "numeric_features",
        "scaler_mean",
        "scaler_scale",
        "feature_mean",
        "feature_scale",
        "type_categories",
        "threshold",
    ):
        assert key in data, key


def test_coefficients_line_up_with_columns(bundle):
    data, _ = bundle
    assert len(data["coefficients"]) == len(data["feature_columns"])


def test_scaler_moments_line_up_with_numeric_features(bundle):
    data, _ = bundle
    assert len(data["scaler_mean"]) == len(data["numeric_features"])
    assert len(data["scaler_scale"]) == len(data["numeric_features"])


def test_bundle_is_small_enough_to_load_instantly(bundle):
    """The whole point of exporting is that the page needs no server."""
    _, artifacts = bundle
    assert (artifacts / "model.json").stat().st_size < 32_000


def test_browser_arithmetic_matches_sklearn(bundle):
    """The load-bearing test: same rows, same scores, to floating-point tolerance."""
    data, artifacts = bundle
    rows = make_frame(n=60, seed=88).drop(columns=["isFraud"])

    sklearn_scores = score(rows, artifacts)["fraud_score"].to_numpy()
    browser_scores = np.array(
        [score_like_browser(data, row) for _, row in rows.iterrows()]
    )
    np.testing.assert_allclose(browser_scores, sklearn_scores, rtol=1e-9, atol=1e-9)


def test_browser_arithmetic_holds_for_an_unseen_transaction_type(bundle):
    """An unknown channel must one-hot to all zeros, exactly as handle_unknown does."""
    data, artifacts = bundle
    rows = make_frame(n=10, seed=89).drop(columns=["isFraud"])
    rows["type"] = "CRYPTO_OUT"

    sklearn_scores = score(rows, artifacts)["fraud_score"].to_numpy()
    browser_scores = np.array(
        [score_like_browser(data, row) for _, row in rows.iterrows()]
    )
    np.testing.assert_allclose(browser_scores, sklearn_scores, rtol=1e-9, atol=1e-9)


def test_browser_arithmetic_holds_when_the_balance_anomaly_fires(bundle):
    data, artifacts = bundle
    rows = make_frame(n=10, seed=90).drop(columns=["isFraud"])
    rows["amount"] = 900_000.0
    rows["oldbalanceDest"] = 500.0
    rows["newbalanceDest"] = 500.0

    sklearn_scores = score(rows, artifacts)["fraud_score"].to_numpy()
    browser_scores = np.array(
        [score_like_browser(data, row) for _, row in rows.iterrows()]
    )
    np.testing.assert_allclose(browser_scores, sklearn_scores, rtol=1e-9, atol=1e-9)


def test_exported_json_round_trips(bundle):
    _, artifacts = bundle
    reloaded = json.loads((artifacts / "model.json").read_text())
    assert isinstance(reloaded["intercept"], float)
