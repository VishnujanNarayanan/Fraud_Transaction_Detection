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


def legitimate_quantiles(y_true, scores) -> dict:
    """The score distribution of LEGITIMATE rows, as a quantile table.

    class_weight="balanced" upweights the positive class by roughly 775 to 1, so the
    model's output is a risk ranking, not a calibrated probability. On held-out data the
    median legitimate transaction scores 0.0000 and the 99th percentile scores 0.81 — so
    a raw score of 0.977, which reads alarmingly high to a person, is beaten by 0.22% of
    perfectly ordinary transactions.

    Publishing this table lets a reader be told where a score sits in the legitimate
    population, which is the question they actually have, instead of a number whose
    scale is an artefact of the class weighting.

    The grid is deliberately dense above the 99th percentile, because that is where the
    entire decision lives.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    legitimate = scores[y_true == 0]
    grid = np.concatenate([np.arange(0.0, 99.0, 1.0), np.arange(99.0, 100.001, 0.05)])
    return {
        "percentiles": [round(float(p), 3) for p in grid],
        "values": [float(v) for v in np.percentile(legitimate, grid)],
    }


def input_ranges(frame) -> dict:
    """What each raw input actually looks like in the training data.

    The demo lets anyone type any number, and a model asked to score a transaction
    unlike anything it was trained on will still answer confidently. Publishing the
    observed range lets the page say so instead of pretending the answer is sound.

    p999 rather than the maximum is the upper guide: PaySim's maxima are single
    extreme rows, so warning against them would fire on almost nothing.
    """
    import numpy as np

    out = {}
    for column in (
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ):
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy()
        out[column] = {
            "min": float(np.min(values)),
            "p999": float(np.percentile(values, 99.9)),
            "max": float(np.max(values)),
        }
    return out


#: The channels the model is actually asked to score. FraudPreprocessor.fit drops the
#: other three, and PaySim never labels them fraudulent, so prevalence figures quoted
#: over all six million rows would describe a population the model never sees.
SCORED_CHANNELS = ("TRANSFER", "CASH_OUT")


def balance_patterns(frame, target: str = "isFraud") -> dict:
    """How common each balance shape is, and how fraudulent, in the scored channels.

    This exists to stop the demo warning about normal data. Intuition says a transfer
    larger than the sender's balance, or a recipient whose balance does not move, is
    broken input. In PaySim both are ordinary: 89.8% and 22.7% of scored rows
    respectively. Telling a visitor those are unreal would be false, and would steer
    them away from the shapes that carry the signal -- a recipient balance that does not
    move is roughly six times more likely to be fraud, which is exactly what the
    engineered suspicious_flag encodes.

    So the page reports prevalence instead of warning. Measured, not assumed.
    """
    import numpy as np

    scored = frame[frame["type"].isin(SCORED_CHANNELS)]
    if scored.empty:
        return {}

    labels = scored[target].to_numpy() if target in scored.columns else None
    amount = scored["amount"].to_numpy()

    old_org = scored["oldbalanceOrg"].to_numpy()
    new_org = scored["newbalanceOrig"].to_numpy()
    old_dest = scored["oldbalanceDest"].to_numpy()
    new_dest = scored["newbalanceDest"].to_numpy()

    masks = {
        "amount_over_sender_balance": amount > old_org,
        "sender_balance_mismatch": np.abs(old_org - amount - new_org) > 0.01,
        "recipient_balance_mismatch": np.abs(old_dest + amount - new_dest) > 0.01,
        # Rare shapes worth naming individually, because their fraud rates are extreme
        # in both directions and a visitor who lands on one should be told.
        "amount_zero": amount == 0,
        "recipient_empty_throughout": (old_dest == 0) & (new_dest == 0),
        "recipient_balance_fell": new_dest < old_dest,
        "recipient_gained_more_than_sent": new_dest > old_dest + amount + 0.01,
        "sender_balance_rose": new_org > old_org,
        "sender_started_empty": old_org == 0,
    }

    out = {"rows": int(len(scored))}
    if labels is not None:
        out["base_rate"] = float(labels.mean())
    for name, mask in masks.items():
        entry = {"share": float(mask.mean()), "rows": int(mask.sum())}
        if labels is not None and mask.any():
            entry["fraud_rate"] = float(labels[mask].mean())
        out[name] = entry
    return out


def channel_summary(frame, target: str = "isFraud") -> dict:
    """Every transaction channel in the data, with its size and fraud rate.

    The demo previously offered only the two channels the encoder was fitted on. The
    other three exist in the data and a visitor may reasonably want to try them, so
    they are offered too -- labelled, because PaySim contains no fraud at all in
    PAYMENT, DEBIT or CASH_IN and the model was never fitted on them.
    """
    out = {}
    for channel, group in frame.groupby("type"):
        entry = {"rows": int(len(group)), "fitted": channel in SCORED_CHANNELS}
        if target in group.columns:
            entry["fraud_rate"] = float(group[target].mean())
        out[str(channel)] = entry
    return out


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
