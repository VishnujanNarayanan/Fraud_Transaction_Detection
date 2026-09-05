"""End-to-end training test.

Runs the real entry point on a synthetic PaySim-shaped frame and asserts that every
artifact needed to score new data is written. The notebook persisted only the
preprocessor, so a score could not be reproduced without retraining; that regression
is what these assertions prevent.
"""

import json
import warnings

import joblib
import pytest
from sklearn.exceptions import ConvergenceWarning

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


def test_persisted_artifacts_score_unseen_rows_together(trained):
    """All four artifacts must work as a set on data none of them has seen.

    The preprocessor alone is not enough: the feature scaler sits between it and the
    model, and skipping it feeds raw engineered magnitudes to coefficients fitted on
    standardised ones - which produces confident, wrong numbers rather than an error.
    """
    _, artifacts = trained
    pre = joblib.load(artifacts / "fraud_preprocessor.pkl")
    feature_scaler = joblib.load(artifacts / "feature_scaler.joblib")
    model = joblib.load(artifacts / "fraud_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())

    fresh = make_frame(n=100, seed=99).drop(columns=["isFraud"])
    features = feature_scaler.transform(pre.transform(fresh)[columns])
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


def test_features_are_standardised_before_fitting(trained):
    """FraudPreprocessor scales four columns; the model needs all of them scaled.

    The engineered columns - diff_orig, log_amount, step, hour - leave transform() on
    their raw scales, spanning several orders of magnitude. Fitting on those directly
    made lbfgs exhaust max_iter without converging and cost real recall on the full
    dataset, so the second scaler is load-bearing rather than the redundant step it
    looks like.
    """
    _, artifacts = trained
    scaler = joblib.load(artifacts / "feature_scaler.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())
    assert len(scaler.mean_) == len(columns)


def test_training_converges(tmp_path):
    """A ConvergenceWarning means the reported coefficients are wherever lbfgs stopped."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        train(artifacts=tmp_path, frame=make_frame(n=3000, seed=23))


def test_missing_csv_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        train(csv_path=tmp_path / "nope.csv", artifacts=tmp_path)
