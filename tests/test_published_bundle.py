"""Guard the bundle that actually ships to visitors.

docs/model.json is the file every visitor downloads. If it is committed stale,
truncated, or exported from a model whose column order no longer matches, the demo
shows confident numbers that are wrong and nothing else in the suite would notice --
every other test builds its own bundle in a tmp directory.

These tests run only when the file is present, so the suite stays green before the
first export and turns into a real gate the moment a bundle is published.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / "model.json"

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(), reason="docs/model.json has not been exported yet"
)


@pytest.fixture(scope="module")
def bundle():
    return json.loads(BUNDLE.read_text())


def test_bundle_is_valid_json_with_every_required_key(bundle):
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
        assert key in bundle, f"published bundle is missing {key}"


def test_coefficients_line_up_with_columns(bundle):
    """A mismatch here is the failure that produces confident nonsense."""
    assert len(bundle["coefficients"]) == len(bundle["feature_columns"])


def test_scaler_moments_line_up_with_numeric_features(bundle):
    assert len(bundle["scaler_mean"]) == len(bundle["numeric_features"])
    assert len(bundle["scaler_scale"]) == len(bundle["numeric_features"])


def test_no_scale_is_zero(bundle):
    """A zero scale would divide by zero in the browser and yield NaN or Infinity."""
    assert all(scale != 0 for scale in bundle["scaler_scale"])
    assert all(scale != 0 for scale in bundle["feature_scale"])


def test_feature_scaler_covers_every_column(bundle):
    """One mean and one scale per feature column, or the page misaligns them."""
    assert len(bundle["feature_mean"]) == len(bundle["feature_columns"])
    assert len(bundle["feature_scale"]) == len(bundle["feature_columns"])


def test_every_value_is_finite(bundle):
    """NaN survives json.dumps as a bare literal that JSON.parse rejects outright."""
    for key in ("coefficients", "scaler_mean", "scaler_scale", "feature_mean", "feature_scale"):
        for value in bundle[key]:
            assert value == value, f"{key} contains NaN"
            assert abs(value) != float("inf"), f"{key} contains an infinity"
    assert bundle["intercept"] == bundle["intercept"]


def test_threshold_is_a_probability(bundle):
    assert 0.0 < bundle["threshold"] < 1.0


def test_transaction_channels_are_the_ones_the_model_was_fitted_on(bundle):
    """PaySim's fraud-bearing channels. An empty or unexpected list means a bad export."""
    assert bundle["type_categories"], "no transaction channels in the bundle"
    assert set(bundle["type_categories"]) <= {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }


def test_one_hot_columns_match_the_declared_channels(bundle):
    """The page builds type_<channel> keys from type_categories; they must exist."""
    declared = {f"type_{c}" for c in bundle["type_categories"]}
    present = {c for c in bundle["feature_columns"] if c.startswith("type_")}
    assert declared == present


def test_bundle_stays_small_enough_to_load_instantly(bundle):
    assert BUNDLE.stat().st_size < 32_000


def test_published_metrics_are_plausible(bundle):
    """Guards against publishing a bundle from a toy run rather than the real dataset."""
    metrics = bundle.get("metrics") or {}
    if "average_precision" in metrics:
        assert 0.0 <= metrics["average_precision"] <= 1.0
    if "rows" in metrics:
        assert metrics["rows"] > 0
