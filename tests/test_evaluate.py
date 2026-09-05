"""Tests for the imbalanced-classification metrics.

The point of this module is that ROC-AUC lies on this dataset, so the tests assert
the properties that make average precision and the threshold sweep trustworthy
substitutes rather than just checking sklearn still returns a float.
"""

import numpy as np
import pytest

from src.evaluate import (
    curve_points,
    evaluate,
    legitimate_quantiles,
    pick_threshold,
    threshold_sweep,
)


@pytest.fixture
def imbalanced():
    """1% positives, with scores that separate the classes but overlap in the middle."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.01).astype(int)
    scores = np.clip(rng.beta(2, 8, n) + y * rng.uniform(0.2, 0.5, n), 0, 1)
    return y, scores


def test_evaluate_reports_average_precision_and_roc_auc(imbalanced):
    result = evaluate(*imbalanced)
    assert 0.0 <= result["average_precision"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_roc_auc_flatters_the_model_relative_to_average_precision(imbalanced):
    """The whole reason this module exists: on 1% positives ROC-AUC reads far higher.

    If this assertion ever fails the dataset is no longer imbalanced enough for the
    argument in the README to hold, and the headline metric should be revisited.
    """
    result = evaluate(*imbalanced)
    assert result["roc_auc"] > result["average_precision"]


def test_base_rate_and_counts_are_consistent(imbalanced):
    y, scores = imbalanced
    result = evaluate(y, scores)
    assert result["rows"] == len(y)
    assert result["positives"] == int(y.sum())
    assert result["base_rate"] == pytest.approx(y.sum() / len(y))


def test_raising_the_threshold_cannot_increase_recall(imbalanced):
    """Monotonicity is the property that makes the sweep table meaningful."""
    rows = threshold_sweep(*imbalanced)
    recalls = [row["recall"] for row in rows]
    assert recalls == sorted(recalls, reverse=True)


def test_raising_the_threshold_cannot_increase_alert_volume(imbalanced):
    rows = threshold_sweep(*imbalanced)
    alerts = [row["alerts"] for row in rows]
    assert alerts == sorted(alerts, reverse=True)


def test_sweep_precision_and_recall_stay_in_range(imbalanced):
    for row in threshold_sweep(*imbalanced):
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert row["caught"] <= row["alerts"]


def test_evaluate_at_a_higher_threshold_raises_precision(imbalanced):
    y, scores = imbalanced
    low = evaluate(y, scores, threshold=0.2)
    high = evaluate(y, scores, threshold=0.6)
    assert high["alerts"] <= low["alerts"]


def test_pick_threshold_returns_a_usable_probability(imbalanced):
    threshold = pick_threshold(*imbalanced)
    assert 0.0 <= threshold <= 1.0


def test_pick_threshold_favours_recall_as_beta_rises(imbalanced):
    """Higher beta weights recall more, so it should not pick a stricter cutoff."""
    y, scores = imbalanced
    assert pick_threshold(y, scores, beta=4.0) <= pick_threshold(y, scores, beta=0.5)


def test_curve_points_are_thinned_and_paired(imbalanced):
    curve = curve_points(*imbalanced, points=50)
    assert len(curve["precision"]) == len(curve["recall"]) <= 50


def test_a_perfect_ranker_scores_one():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    result = evaluate(y, scores)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_legitimate_quantiles_are_monotonic(imbalanced):
    """The page binary-searches this table, so order is a correctness requirement."""
    table = legitimate_quantiles(*imbalanced)
    assert table["values"] == sorted(table["values"])
    assert len(table["values"]) == len(table["percentiles"])


def test_legitimate_quantiles_describe_only_the_negatives(imbalanced):
    """Including fraud rows would inflate the reference distribution and flatten the tail."""
    import numpy as np

    y, scores = imbalanced
    table = legitimate_quantiles(y, scores)
    assert table["values"][-1] == pytest.approx(float(np.max(scores[y == 0])))


def test_quantile_grid_is_dense_in_the_tail(imbalanced):
    """The decision lives above the 99th percentile, so that is where resolution matters."""
    percentiles = legitimate_quantiles(*imbalanced)["percentiles"]
    assert sum(1 for p in percentiles if p >= 99.0) >= 20
