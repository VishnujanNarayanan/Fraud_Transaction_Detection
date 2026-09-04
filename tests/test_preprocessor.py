"""Contract tests for FraudPreprocessor.

These pin the three things that silently break a fraud model: the output column set
drifting between train and score time, test statistics leaking into the fitted
scaler, and the balance-anomaly flag mis-firing.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessor import DROP_COLUMNS, FraudPreprocessor

from .conftest import make_frame


@pytest.fixture
def fitted(features):
    return FraudPreprocessor().fit(features)


def test_is_a_sklearn_transformer():
    pre = FraudPreprocessor()
    assert isinstance(pre, BaseEstimator)
    assert isinstance(pre, TransformerMixin)


def test_columns_are_stable_across_disjoint_batches(fitted):
    """Scoring next month's extract must produce the training column set exactly.

    A drifting column set is the failure that turns a working model into silent
    garbage in production, because the coefficient vector no longer lines up.
    """
    a = fitted.transform(make_frame(n=200, seed=1))
    b = fitted.transform(make_frame(n=200, seed=2))
    assert list(a.columns) == list(b.columns)


def test_unseen_transaction_type_does_not_add_a_column(fitted, features):
    """handle_unknown="ignore" must absorb a new channel, not widen the frame."""
    baseline = list(fitted.transform(features).columns)
    novel = make_frame(n=50, seed=3)
    novel.loc[novel.index[:10], "type"] = "CRYPTO_OUT"
    assert list(fitted.transform(novel.drop(columns=["isFraud"])).columns) == baseline


def test_identifiers_and_leakage_are_dropped(fitted, features):
    out = fitted.transform(features)
    for col in DROP_COLUMNS:
        assert col not in out.columns, f"{col} survived transform"


def test_isflaggedfraud_is_never_a_feature(fitted, features):
    """isFlaggedFraud is the dataset's own rule-based label, not a pre-hoc signal."""
    assert "isFlaggedFraud" not in fitted.transform(features).columns


def test_scaler_is_fitted_on_train_only(features):
    """Refitting on a superset must move the scaler; that is what leakage would look like."""
    train = features.iloc[:200]
    pre_train = FraudPreprocessor().fit(train)
    pre_all = FraudPreprocessor().fit(features)
    assert not np.allclose(pre_train.scaler.mean_, pre_all.scaler.mean_)


def test_transform_does_not_mutate_its_input(fitted, features):
    before = features.copy()
    fitted.transform(features)
    pd.testing.assert_frame_equal(features, before)


def test_transform_is_pure(fitted, features):
    """Two calls on the same rows must agree, or scoring is not reproducible."""
    pd.testing.assert_frame_equal(fitted.transform(features), fitted.transform(features))


def test_suspicious_flag_fires_when_destination_balance_does_not_move(fitted):
    """Money left the sender but the recipient's balance never changed."""
    frame = make_frame(n=20, seed=4).drop(columns=["isFraud"])
    frame["amount"] = 5_000.0
    frame["oldbalanceDest"] = 1_000.0
    frame["newbalanceDest"] = 1_000.0
    assert (fitted.transform(frame)["suspicious_flag"] == 1).all()


def test_suspicious_flag_stays_down_on_a_normal_transfer(fitted):
    frame = make_frame(n=20, seed=5).drop(columns=["isFraud"])
    frame["amount"] = 5_000.0
    frame["oldbalanceDest"] = 1_000.0
    frame["newbalanceDest"] = 6_000.0
    assert (fitted.transform(frame)["suspicious_flag"] == 0).all()


def test_error_flag_fires_on_a_negative_balance(fitted):
    frame = make_frame(n=10, seed=6).drop(columns=["isFraud"])
    frame["oldbalanceOrg"] = -1.0
    assert (fitted.transform(frame)["error_flag"] == 1).all()


def test_hour_is_derived_from_step_and_encoded_cyclically(fitted):
    frame = make_frame(n=48, seed=7).drop(columns=["isFraud"])
    frame["step"] = list(range(48))
    out = fitted.transform(frame)
    assert out["hour"].tolist() == [s % 24 for s in range(48)]
    # 23:00 and 00:00 must be neighbours on the circle, not opposite ends of a line.
    assert np.isclose(
        np.hypot(
            out.loc[23, "hour_sin"] - out.loc[0, "hour_sin"],
            out.loc[23, "hour_cos"] - out.loc[0, "hour_cos"],
        ),
        np.hypot(
            out.loc[11, "hour_sin"] - out.loc[12, "hour_sin"],
            out.loc[11, "hour_cos"] - out.loc[12, "hour_cos"],
        ),
    )


def test_log_amount_is_present_and_monotonic(fitted, features):
    out = fitted.transform(features)
    assert "log_amount" in out.columns
    order_raw = features["amount"].rank()
    order_log = out["log_amount"].rank()
    pd.testing.assert_series_equal(order_raw, order_log, check_names=False)


def test_log_transform_can_be_switched_off(features):
    pre = FraudPreprocessor(log_transform=False).fit(features)
    assert "log_amount" not in pre.transform(features).columns


def test_output_is_entirely_numeric(fitted, features):
    out = fitted.transform(features)
    assert not out.select_dtypes(exclude="number").columns.tolist()


def test_no_nulls_are_introduced(fitted, features):
    assert not fitted.transform(features).isnull().any().any()
