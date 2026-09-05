"""Export the trained model to JSON so a static page can score in the browser.

Why not a hosted app? A Streamlit or Hugging Face free tier sleeps, so the first
visitor waits 30-50 seconds on a cold start and concludes the demo is broken. A
logistic regression is a coefficient vector and an intercept, and the preprocessor is
a set of one-hot categories plus per-column means and scales. All of it fits in a few
kilobytes of JSON, so the model can run client-side on a static page that loads
instantly, costs nothing, and never sleeps.

This module exports exactly the parameters the browser needs to reproduce
FraudPreprocessor.transform followed by the model's sigmoid. tests/test_export_web.py
asserts the two agree to within floating-point tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUT = ROOT / "docs" / "model.json"


def export(artifacts: Path = ARTIFACTS, out: Path = DEFAULT_OUT) -> dict:
    """Write the model, scaler moments and one-hot categories as a JSON bundle."""
    pre = joblib.load(artifacts / "fraud_preprocessor.pkl")
    feature_scaler = joblib.load(artifacts / "feature_scaler.joblib")
    model = joblib.load(artifacts / "fraud_model.joblib")
    columns = json.loads((artifacts / "feature_columns.json").read_text())

    metrics = {}
    metrics_path = artifacts / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    bundle = {
        "feature_columns": columns,
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "numeric_features": list(pre.numeric_features),
        "scaler_mean": [float(m) for m in pre.scaler.mean_],
        "scaler_scale": [float(s) for s in pre.scaler.scale_],
        "log_transform": bool(pre.log_transform),
        # The second scaler, over every feature column. Without it the page would
        # feed raw engineered values to coefficients fitted on standardised ones.
        "feature_mean": [float(m) for m in feature_scaler.mean_],
        "feature_scale": [float(s) for s in feature_scaler.scale_],
        "type_categories": [str(c) for c in pre.ohe.categories_[0]],
        "threshold": float(metrics.get("threshold", 0.5)),
        "metrics": {
            key: metrics[key]
            for key in (
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "rows",
                "positives",
                "at_threshold",
            )
            if key in metrics
        },
        # Lets the page say where a score sits among legitimate transactions, rather
        # than showing a raw score whose scale is an artefact of the class weighting.
        "legitimate_quantiles": metrics.get("legitimate_quantiles", {}),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    return bundle


def score_like_browser(bundle: dict, row: dict) -> float:
    """A reference implementation of the browser's arithmetic, used by the tests.

    Kept in Python next to the exporter so the JavaScript in docs/index.html has
    something to be checked against; if the two ever diverge, a test fails rather
    than the demo quietly showing a different number from the model.
    """
    features: dict[str, float] = {}

    amount = float(row["amount"])
    old_org = float(row["oldbalanceOrg"])
    new_org = float(row["newbalanceOrig"])
    old_dest = float(row["oldbalanceDest"])
    new_dest = float(row["newbalanceDest"])
    step = float(row["step"])

    features["step"] = step
    features["amount"] = amount
    features["oldbalanceOrg"] = old_org
    features["oldbalanceDest"] = old_dest
    features["newbalanceDest"] = new_dest
    features["diff_orig"] = old_org - new_org
    features["diff_dest"] = new_dest - old_dest
    features["hour"] = step % 24
    features["day_of_week"] = step % 168
    features["hour_sin"] = float(np.sin(2 * np.pi * (step % 24) / 24))
    features["hour_cos"] = float(np.cos(2 * np.pi * (step % 24) / 24))

    for category in bundle["type_categories"]:
        features[f"type_{category}"] = 1.0 if row["type"] == category else 0.0

    features["always_nonfraud_type"] = 0.0
    if bundle["log_transform"]:
        features["log_amount"] = float(np.log1p(amount))
    features["suspicious_flag"] = 1.0 if (amount > 0 and new_dest == old_dest) else 0.0
    features["error_flag"] = 1.0 if (old_org < 0 or old_dest < 0 or new_dest < 0) else 0.0

    for name, mean, scale in zip(
        bundle["numeric_features"], bundle["scaler_mean"], bundle["scaler_scale"], strict=True
    ):
        features[name] = (features[name] - mean) / scale

    total = bundle["intercept"]
    for name, coefficient, mean, scale in zip(
        bundle["feature_columns"],
        bundle["coefficients"],
        bundle["feature_mean"],
        bundle["feature_scale"],
        strict=True,
    ):
        total += coefficient * ((features.get(name, 0.0) - mean) / scale)
    # Numerically stable sigmoid: exp(-total) overflows for strongly negative
    # scores, and the browser's Math.exp has the same problem. Both sides branch.
    if total >= 0:
        return float(1.0 / (1.0 + np.exp(-total)))
    exponential = np.exp(total)
    return float(exponential / (1.0 + exponential))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    bundle = export(args.artifacts, args.out)
    size = args.out.stat().st_size
    print(f"Wrote {args.out} ({size:,} bytes, {len(bundle['feature_columns'])} features)")


if __name__ == "__main__":
    main()
