"""Tests for scoring new data with the persisted artifacts."""

import json

import pandas as pd
import pytest

from src.predict import load_artifacts, score
from src.train import train

from .conftest import make_frame


@pytest.fixture
def artifacts(tmp_path):
    train(artifacts=tmp_path, frame=make_frame(n=1500, seed=31))
    return tmp_path


def test_scoring_appends_a_score_and_an_alert(artifacts):
    fresh = make_frame(n=80, seed=77).drop(columns=["isFraud"])
    scored = score(fresh, artifacts)
    assert len(scored) == len(fresh)
    assert scored["fraud_score"].between(0, 1).all()
    assert scored["alert"].isin([0, 1]).all()


def test_scoring_tolerates_a_labelled_frame(artifacts):
    """A frame that still carries isFraud must score, not raise."""
    labelled = make_frame(n=50, seed=78)
    scored = score(labelled, artifacts)
    assert "fraud_score" in scored.columns


def test_raising_the_threshold_never_adds_alerts(artifacts):
    fresh = make_frame(n=200, seed=79).drop(columns=["isFraud"])
    low = score(fresh, artifacts, threshold=0.2)["alert"].sum()
    high = score(fresh, artifacts, threshold=0.8)["alert"].sum()
    assert high <= low


def test_scoring_does_not_mutate_the_input(artifacts):
    fresh = make_frame(n=40, seed=80).drop(columns=["isFraud"])
    before = fresh.copy()
    score(fresh, artifacts)
    pd.testing.assert_frame_equal(fresh, before)


def test_threshold_defaults_to_the_one_chosen_at_training_time(artifacts):
    *_, threshold = load_artifacts(artifacts)
    expected = json.loads((artifacts / "metrics.json").read_text())["threshold"]
    assert threshold == expected


def test_missing_artifacts_name_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraud_model.joblib"):
        load_artifacts(tmp_path)
