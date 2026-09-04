"""Synthetic PaySim-shaped fixtures.

Fraud.csv is a ~470 MB Kaggle download and is gitignored, so CI can never see it.
Every test therefore builds its own frame with the same column contract as the real
dataset. That is a feature, not a workaround: it pins the contract the preprocessor
depends on, so a schema change breaks a fast test instead of a 6.36M-row notebook run.
"""

import numpy as np
import pandas as pd
import pytest

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def make_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A frame with PaySim's columns, its channel mix, and its fraud concentration.

    Fraud is confined to TRANSFER and CASH_OUT, matching the simulation, so tests that
    assert on channel behaviour are exercising the same shape the model was built for.
    """
    rng = np.random.default_rng(seed)
    step = rng.integers(1, 744, n)
    kind = rng.choice(TYPES, n, p=[0.34, 0.08, 0.35, 0.01, 0.22])
    amount = np.round(rng.lognormal(mean=9.0, sigma=1.6, size=n), 2)
    old_org = np.round(rng.lognormal(mean=9.5, sigma=1.8, size=n), 2)
    new_org = np.maximum(old_org - amount, 0.0)
    old_dest = np.round(rng.lognormal(mean=9.0, sigma=2.0, size=n), 2)
    new_dest = old_dest + amount

    risky = np.isin(kind, ["TRANSFER", "CASH_OUT"])
    is_fraud = (risky & (rng.random(n) < 0.05)).astype(int)

    # The balance anomaly the model keys on: money left the sender, the recipient's
    # balance never moved.
    new_dest = np.where(is_fraud == 1, old_dest, new_dest)

    return pd.DataFrame(
        {
            "step": step,
            "type": kind,
            "amount": amount,
            "nameOrig": [f"C{i:09d}" for i in range(n)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": [f"C{i + n:09d}" for i in range(n)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": ((amount > 200_000) & (kind == "TRANSFER")).astype(int),
        }
    )[RAW_COLUMNS]


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def features(frame) -> pd.DataFrame:
    return frame.drop(columns=["isFraud"])


@pytest.fixture
def target(frame) -> pd.Series:
    return frame["isFraud"]
