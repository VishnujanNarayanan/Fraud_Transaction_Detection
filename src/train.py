"""Train the fraud classifiers and persist every artifact needed to score new data.

The notebook trained three variants and saved only the preprocessor, so the models
themselves had to be retrained from scratch before anything could be scored. This
entry point saves the model, the preprocessor and the column order together, which is
the minimum set required to reproduce a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate, pick_threshold
from src.preprocessor import REDUCED_DROP, FraudPreprocessor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "Fraud.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "isFraud"
SEED = 42
TEST_SIZE = 0.2


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
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Download PaySim from "
                "https://www.kaggle.com/datasets/ealaxi/paysim1 and unzip it in the "
                "project root."
            )
        frame = pd.read_csv(csv_path)

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    pre = FraudPreprocessor().fit(X_train)
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)

    if reduced:
        X_train_t = X_train_t.drop(columns=REDUCED_DROP, errors="ignore")
        X_test_t = X_test_t.drop(columns=REDUCED_DROP, errors="ignore")

    columns = list(X_train_t.columns)
    model = build_model().fit(X_train_t, y_train)

    scores = model.predict_proba(X_test_t)[:, 1]
    metrics = evaluate(y_test, scores)
    metrics["threshold"] = pick_threshold(y_test, scores)
    metrics["variant"] = "reduced" if reduced else "engineered"

    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, artifacts / "fraud_preprocessor.pkl")
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
