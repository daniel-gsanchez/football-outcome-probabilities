"""Metrics for probability forecasts."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def brier(y, probs):
    """Multiclass Brier score: mean squared error against the one-hot outcome."""
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def scores(y, probs):
    """Proper scoring rules first (they reward good probabilities), then
    classification metrics (they only look at the most likely outcome)."""
    pred = probs.argmax(axis=1)
    return {
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
        "brier": brier(y, probs),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


def score_table(y, forecasts):
    """One row per model: {name: probs} -> DataFrame of metrics."""
    return pd.DataFrame({name: scores(y, p) for name, p in forecasts.items()}).T


def match_log_loss(y, probs):
    """Log-loss of each individual match."""
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def paired_bootstrap(y, probs_a, probs_b, n_boot=5000, seed=0):
    """Mean log-loss difference (a - b) on the same matches, with a bootstrap
    95% interval. Negative means `a` is better."""
    d = match_log_loss(y, probs_a) - match_log_loss(y, probs_b)
    boot = np.random.default_rng(seed).choice(d, (n_boot, len(d))).mean(axis=1)
    return {"diff": d.mean(), "low": np.quantile(boot, 0.025),
            "high": np.quantile(boot, 0.975)}


def calibration_curve(y, probs, outcome, bins=10):
    """Mean forecast vs. observed frequency in equal-count bins."""
    p = probs[:, outcome]
    hit = (y == outcome).astype(float)
    groups = pd.qcut(p, bins, duplicates="drop")
    return (pd.DataFrame({"forecast": p, "observed": hit})
              .groupby(groups, observed=True).mean())
