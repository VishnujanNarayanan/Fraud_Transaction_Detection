"""The FraudPreprocessor transformer, lifted out of Fraud_Detection_Model.ipynb.

Behaviour is deliberately identical to the notebook cell it came from, including the
two features the README documents as provably dead (`error_flag` and
`always_nonfraud_type`, both zero-coefficient). They are kept so that a model trained
here matches one trained in the notebook column for column; `train.py --reduced` is
where they get dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: Transaction channels in which PaySim never labels a row fraudulent. Rows of these
#: types are excluded when fitting the encoder and scaler, so the fitted statistics
#: describe the population the model is actually asked to score.
ALWAYS_NONFRAUD_TYPES = ["PAYMENT", "DEBIT", "CASH_IN"]

#: Dropped on transform: two account identifiers, the dataset's own rule-based fraud
#: flag (leakage — it is a label, not a feature available before the fact), and one
#: half of a perfectly correlated balance pair.
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud", "newbalanceOrig"]

#: Weak or provably zero-coefficient columns, dropped by the "reduced" model variant.
REDUCED_DROP = [
    "oldbalanceDest",
    "error_flag",
    "always_nonfraud_type",
    "day_of_week",
    "newbalanceDest",
    "amount",
    "hour",
]


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Fit encoders and scalers on training rows only, then apply them unchanged.

    Fitting on the full frame would leak test statistics into training, which is the
    failure mode this class exists to prevent: ``fit`` learns the one-hot categories
    and the scaler moments, and ``transform`` is a pure application of them.
    """

    def __init__(self, log_transform: bool = True):
        self.log_transform = log_transform
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_features = [
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "newbalanceDest",
        ]
        self.categorical_features = ["type"]

    def fit(self, X: pd.DataFrame, y=None) -> FraudPreprocessor:
        X_ = X.copy()
        X_ = X_[~X_["type"].isin(ALWAYS_NONFRAUD_TYPES)]
        self.ohe.fit(X_[["type"]])
        self.scaler.fit(X_[self.numeric_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()

        # Balance deltas, computed before newbalanceOrig is dropped.
        if "newbalanceOrig" in X_.columns:
            X_["diff_orig"] = X_["oldbalanceOrg"] - X_["newbalanceOrig"]
        else:
            X_["diff_orig"] = 0
        X_["diff_dest"] = X_["newbalanceDest"] - X_["oldbalanceDest"]

        X_.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")

        # `step` is an hour counter over a 744-hour simulation. Sine/cosine keep 23:00
        # and 00:00 adjacent instead of maximally distant.
        if "step" in X_.columns:
            X_["hour"] = X_["step"] % 24
            X_["day_of_week"] = X_["step"] % 168
            X_["hour_sin"] = np.sin(2 * np.pi * X_["hour"] / 24)
            X_["hour_cos"] = np.cos(2 * np.pi * X_["hour"] / 24)

        type_encoded = self.ohe.transform(X_[["type"]])
        type_encoded = pd.DataFrame(
            type_encoded,
            columns=self.ohe.get_feature_names_out(["type"]),
            index=X_.index,
        )
        X_ = pd.concat([X_.drop(columns=["type"]), type_encoded], axis=1)

        matching = [
            c
            for c in type_encoded.columns
            if any(t in c for t in ALWAYS_NONFRAUD_TYPES)
        ]
        if matching:
            X_["always_nonfraud_type"] = (
                X_[matching].sum(axis=1).clip(upper=1).astype(int)
            )
        else:
            # The encoder was fitted without these channels, so no such column exists
            # and the flag is identically zero — which is why its coefficient is 0.000.
            X_["always_nonfraud_type"] = 0

        if self.log_transform and "amount" in X_.columns:
            X_["log_amount"] = np.log1p(X_["amount"])

        # Money left the sender but the recipient's balance never moved.
        X_["suspicious_flag"] = (
            (X_["amount"] > 0) & (X_["newbalanceDest"] == X_["oldbalanceDest"])
        ).astype(int)

        X_["error_flag"] = (
            (X_["oldbalanceOrg"] < 0)
            | (X_["oldbalanceDest"] < 0)
            | (X_["newbalanceDest"] < 0)
        ).astype(int)

        X_[self.numeric_features] = self.scaler.transform(X_[self.numeric_features])

        return X_
