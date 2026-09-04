"""Score a CSV of transactions with the persisted artifacts.

This is the piece the notebook could not offer: it saved the preprocessor but not the
model, so scoring new data meant retraining first. Here the model, the preprocessor
and the column order are loaded together and applied unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load_artifacts(artifacts: Path = ARTIFACTS) -> tuple:
    """Load the preprocessor, model, column order and chosen threshold together.

    They are only meaningful as a set: a model applied to columns in a different
    order silently produces confident nonsense.
    """
    missing = [
        name
        for name in ("fraud_preprocessor.pkl", "fraud_model.joblib", "feature_columns.json")
        if not (artifacts / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing artifacts in {artifacts}: {', '.join(missing)}. "
            "Run `python -m src.train` first."
        )
    pre = joblib.load(artifacts / "fraud_preprocessor.pkl")
    model = joblib.load(artifacts / "fraud_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())

    threshold = 0.5
    metrics_path = artifacts / "metrics.json"
    if metrics_path.exists():
        threshold = json.loads(metrics_path.read_text()).get("threshold", 0.5)
    return pre, model, columns, threshold


def score(
    frame: pd.DataFrame,
    artifacts: Path = ARTIFACTS,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return the input frame with a fraud score and an alert flag appended."""
    pre, model, columns, chosen = load_artifacts(artifacts)
    cutoff = chosen if threshold is None else threshold

    features = pre.transform(frame.drop(columns=["isFraud"], errors="ignore"))
    scores = model.predict_proba(features[columns])[:, 1]

    out = frame.copy()
    out["fraud_score"] = scores
    out["alert"] = (scores >= cutoff).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV of transactions to score")
    parser.add_argument("--out", type=Path, help="write scored rows here")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--threshold",
        type=float,
        help="override the threshold chosen at training time",
    )
    parser.add_argument("--alerts-only", action="store_true")
    args = parser.parse_args()

    scored = score(pd.read_csv(args.input), args.artifacts, args.threshold)
    if args.alerts_only:
        scored = scored[scored["alert"] == 1]

    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"{len(scored):,} rows written to {args.out}")
    else:
        print(scored.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
