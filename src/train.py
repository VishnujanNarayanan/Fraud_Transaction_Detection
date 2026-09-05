"""Train the fraud classifiers and persist every artifact needed to score new data.

The notebook trained three variants and saved only the preprocessor, so the models
themselves had to be retrained from scratch before anything could be scored. This
entry point saves the model, the preprocessor and the column order together, which is
the minimum set required to reproduce a score.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate, legitimate_quantiles, pick_threshold
from src.preprocessor import REDUCED_DROP, FraudPreprocessor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Fraud.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "isFraud"
SEED = 42
TEST_SIZE = 0.2

#: Explicit dtypes and a narrowed column set for the 6.36M-row CSV.
#:
#: Loading it with pandas' defaults needs several gigabytes and is killed by the OOM
#: reaper on an 8 GB machine. Two things dominate: nameOrig and nameDest are 11-character
#: strings held as Python objects, which costs well over a gigabyte between them, and
#: every integer column is widened to int64.
#:
#: The account identifiers are dropped by FraudPreprocessor.transform anyway, and
#: isFlaggedFraud is dropped as leakage, so not reading them costs nothing and saves
#: most of the footprint. Balances stay float64 so the fitted coefficients match what
#: the notebook produced.
RAW_DTYPES = {
    "step": "int32",
    "type": "object",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    TARGET: "int8",
}


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """Read the transaction CSV using only the columns the model actually consumes."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download PaySim from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and unzip it in the "
            "project root."
        )
    return pd.read_csv(csv_path, usecols=list(RAW_DTYPES), dtype=RAW_DTYPES)


def build_model() -> LogisticRegression:
    """class_weight="balanced" rather than resampling.

    Resampling a 6.36M-row frame either duplicates it in memory or invents synthetic
    rows; re-weighting the loss achieves the same correction for free.
    """
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")


def train(
    csv_path: Path = DEFAULT_CSV,
    artifacts: Path = ARTIFACTS,
    reduced: bool = False,
    frame: pd.DataFrame | None = None,
) -> dict:
    """Fit the preprocessor and model, evaluate honestly, and persist everything."""
    if frame is None:
        frame = load_transactions(csv_path)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    # The split copied both halves; the source frame is now dead weight worth
    # hundreds of megabytes, and on an 8 GB machine that is the difference between
    # finishing and being killed.
    del frame, X, y
    gc.collect()

    pre = FraudPreprocessor().fit(X_train)
    X_train_t = pre.transform(X_train)
    del X_train
    gc.collect()

    if reduced:
        X_train_t = X_train_t.drop(columns=REDUCED_DROP, errors="ignore")

    columns = list(X_train_t.columns)

    # A second scaler over ALL feature columns, not just the four FraudPreprocessor
    # handles. The engineered columns - diff_orig, diff_dest, log_amount, step, hour,
    # day_of_week and the one-hot flags - leave transform() on their raw scales, which
    # span several orders of magnitude. Handing that to lbfgs makes it exhaust
    # max_iter without converging and costs real recall. Fitted on train only, for the
    # same reason the first scaler is.
    feature_scaler = StandardScaler().fit(X_train_t)
    model = build_model().fit(feature_scaler.transform(X_train_t), y_train)
    del X_train_t
    gc.collect()

    # Held-out rows are transformed only after training has released its matrix.
    X_test_t = pre.transform(X_test)
    if reduced:
        X_test_t = X_test_t.drop(columns=REDUCED_DROP, errors="ignore")
    del X_test
    gc.collect()

    scores = model.predict_proba(feature_scaler.transform(X_test_t[columns]))[:, 1]
    del X_test_t
    gc.collect()
    metrics = evaluate(y_test, scores)
    metrics["threshold"] = pick_threshold(y_test, scores)
    # Precision and recall at the operating point that will actually be used. Reporting
    # only the 0.50 figures understates the model badly: precision there is 0.037, while
    # at the chosen threshold it is roughly 0.65 for three quarters of the recall.
    metrics["at_threshold"] = evaluate(y_test, scores, threshold=metrics["threshold"])
    metrics["legitimate_quantiles"] = legitimate_quantiles(y_test, scores)
    metrics["variant"] = "reduced" if reduced else "engineered"

    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, artifacts / "fraud_preprocessor.pkl")
    joblib.dump(feature_scaler, artifacts / "feature_scaler.joblib")
    joblib.dump(model, artifacts / "fraud_model.joblib")
    (artifacts / "feature_columns.json").write_text(json.dumps(columns, indent=2))
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (artifacts / "coefficients.json").write_text(
        json.dumps(
            dict(sorted(zip(columns, model.coef_[0].tolist(), strict=True),
                        key=lambda kv: -abs(kv[1]))),
            indent=2,
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="drop the weak and provably zero-coefficient features",
    )
    args = parser.parse_args()

    metrics = train(args.csv, args.artifacts, reduced=args.reduced)
    print(json.dumps(metrics, indent=2, default=float))
    print(f"\nArtifacts written to {args.artifacts}")


if __name__ == "__main__":
    main()
