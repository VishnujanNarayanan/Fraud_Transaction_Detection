"""End-to-end training test.

Runs the real entry point on a synthetic PaySim-shaped frame and asserts that every
artifact needed to score new data is written. The notebook persisted only the
preprocessor, so a score could not be reproduced without retraining; that regression
is what these assertions prevent.
"""

import json

import joblib
import pytest

from src.preprocessor import REDUCED_DROP
from src.train import train

from .conftest import make_frame


@pytest.fixture
def trained(tmp_path):
    frame = make_frame(n=1500, seed=21)
    metrics = train(artifacts=tmp_path, frame=frame)
    return metrics, tmp_path


def test_writes_every_artifact_needed_to_score(trained):
    _, artifacts = trained
    for name in (
        "fraud_preprocessor.pkl",
        "fraud_model.joblib",
        "feature_columns.json",
        "metrics.json",
        "coefficients.json",
    ):
        assert (artifacts / name).exists(), f"{name} was not written"


def test_persisted_model_and_columns_agree(trained):
    _, artifacts = trained
    model = joblib.load(artifacts / "fraud_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())
    assert model.coef_.shape[1] == len(columns)


def test_persisted_preprocessor_can_score_unseen_rows(trained):
    """The artifacts must work together on data neither of them has seen."""
    _, artifacts = trained
    pre = joblib.load(artifacts / "fraud_preprocessor.pkl")
    model = joblib.load(artifacts / "fraud_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())

    fresh = make_frame(n=100, seed=99).drop(columns=["isFraud"])
    features = pre.transform(fresh)[columns]
    scores = model.predict_proba(features)[:, 1]
    assert len(scores) == 100
    assert ((scores >= 0) & (scores <= 1)).all()


def test_metrics_include_the_honest_summary(trained):
    metrics, _ = trained
    assert "average_precision" in metrics
    assert "threshold" in metrics
    assert metrics["variant"] == "engineered"


def test_reduced_variant_drops_the_weak_features(tmp_path):
    frame = make_frame(n=1500, seed=22)
    train(artifacts=tmp_path, frame=frame, reduced=True)
    columns = json.loads((tmp_path / "feature_columns.json").read_text())
    assert not set(columns) & set(REDUCED_DROP)


def test_coefficients_are_ranked_by_absolute_weight(trained):
    _, artifacts = trained
    coefficients = json.loads((artifacts / "coefficients.json").read_text())
    magnitudes = [abs(v) for v in coefficients.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_missing_csv_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        train(csv_path=tmp_path / "nope.csv", artifacts=tmp_path)
