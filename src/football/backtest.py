"""Flat-stake value-betting backtest."""
import numpy as np
import pandas as pd


def value_bets(probs, decimal_odds, min_edge):
    """Bet 1 unit on the outcome with the highest expected value, if any.

    Expected value of a 1-unit bet = p * odds - 1. A bet is placed only when
    it exceeds `min_edge` (e.g. 0.05 = the model sees a 5% edge).
    Returns one row per match with the chosen outcome and the profit.
    """
    ev = probs * decimal_odds - 1
    best = np.nanargmax(np.where(np.isnan(ev), -np.inf, ev), axis=1)
    best_ev = ev[np.arange(len(ev)), best]
    return pd.DataFrame({
        "bet": best_ev > min_edge,
        "outcome": best,
        "odds": decimal_odds[np.arange(len(ev)), best],
        "edge": best_ev,
    })


def settle(bets, y):
    """Profit per bet: odds - 1 if it wins, -1 if it loses."""
    placed = bets[bets["bet"]].copy()
    won = placed["outcome"].to_numpy() == np.asarray(y)[placed.index]
    placed["profit"] = np.where(won, placed["odds"] - 1, -1.0)
    return placed


def summary(placed, n_boot=5000, seed=0):
    """Number of bets, hit rate, ROI and a bootstrap 95% interval for the ROI."""
    profit = placed["profit"].to_numpy()
    if len(profit) == 0:
        return {"bets": 0, "hit_rate": np.nan, "roi": np.nan,
                "roi_low": np.nan, "roi_high": np.nan}
    rng = np.random.default_rng(seed)
    boot = rng.choice(profit, (n_boot, len(profit))).mean(axis=1)
    return {
        "bets": len(profit),
        "hit_rate": float((profit > 0).mean()),
        "roi": float(profit.mean()),
        "roi_low": float(np.quantile(boot, 0.025)),
        "roi_high": float(np.quantile(boot, 0.975)),
    }
