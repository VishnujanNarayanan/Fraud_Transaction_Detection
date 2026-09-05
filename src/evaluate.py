"""Honest evaluation for a heavily imbalanced classifier.

The README's Limitations section already concedes the problem this module fixes:
ROC-AUC was the headline metric, and at 0.129% positives it is dominated by the
1.27M easy negatives, so it flatters the model. Average precision is the honest
summary, and the operating threshold is a business decision rather than a default.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """Metrics at a given operating point, plus both summary scores.

    average_precision is reported first because it is the one that means something
    here: it is the area under the precision-recall curve, which ignores the easy
    negatives that ROC-AUC spends most of its range on.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    positives = int(y_true.sum())
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "positives": positives,
        "rows": int(len(y_true)),
        "base_rate": float(positives / len(y_true)) if len(y_true) else 0.0,
        "alerts": int(predicted.sum()),
    }


def threshold_sweep(y_true, scores, points: int = 50) -> list[dict]:
    """Precision, recall and alert volume across the range of decision thresholds.

    This is the table a fraud desk actually needs. Precision of 0.037 at the default
    cutoff means roughly one alert in 27 is genuine; whether that is acceptable
    depends on the cost of a missed fraud against the cost of reviewing a false one,
    and that trade cannot be read off a single number.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rows = []
    for threshold in np.linspace(0.05, 0.95, points):
        predicted = (scores >= threshold).astype(int)
        alerts = int(predicted.sum())
        caught = int(((predicted == 1) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "alerts": alerts,
                "caught": caught,
                "precision": float(caught / alerts) if alerts else 0.0,
                "recall": float(caught / y_true.sum()) if y_true.sum() else 0.0,
            }
        )
    return rows


def pick_threshold(y_true, scores, beta: float = 2.0) -> float:
    """The threshold maximising F-beta, defaulting to beta=2.

    beta=2 weights recall twice as heavily as precision, which is the right default
    for fraud: a missed fraud costs the value of the transfer, while a false alert
    costs a few minutes of an analyst's time. The caller can override it, and should,
    once they know their real costs.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more point than thresholds; drop the last.
    precision, recall = precision[:-1], recall[:-1]
    denominator = (beta**2 * precision) + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = np.where(
            denominator > 0,
            (1 + beta**2) * precision * recall / denominator,
            0.0,
        )
    return float(thresholds[int(np.argmax(f_beta))])


def curve_points(y_true, scores, points: int = 200) -> dict:
    """A precision-recall curve thinned to `points`, for plotting or export."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    if len(precision) > points:
        index = np.linspace(0, len(precision) - 1, points).astype(int)
        precision, recall = precision[index], recall[index]
    return {
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
    }
