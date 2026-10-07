import numpy as np
import pandas as pd

from football.backtest import settle, value_bets
from football.data import implied_probabilities
from football.features import build_sequences


def toy_matches():
    """Teams 1 and 2 play each other three times; results are all different."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        "home_team": [1, 2, 1],
        "away_team": [2, 1, 2],
        "league_id": [9, 9, 9],
        "home_goals": [3, 0, 1],
        "away_goals": [0, 0, 2],
        "result": [0, 1, 2],
    })


def test_sequences_only_use_past_matches():
    s = build_sequences(toy_matches(), window=2)
    # Only the third match has two previous matches for both teams.
    assert list(s["index"]) == [2]
    home_history = s["home_seq"][0]  # team 1 before match 3
    assert home_history[:, 0].tolist() == [3, 0]  # goals for in matches 1 and 2
    assert 1 not in home_history[:, 0]  # match 3 itself (1-2) is not included


def test_implied_probabilities_remove_margin():
    probs, overround = implied_probabilities(np.array([[2.0, 3.5, 4.0]]))
    assert np.isclose(probs.sum(), 1.0)
    assert np.isclose(overround[0], 1 / 2 + 1 / 3.5 + 1 / 4)


def test_value_bet_profit():
    probs = np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4]])
    odds = np.array([[2.0, 4.0, 5.0], [2.0, 3.0, 2.0]])
    bets = value_bets(probs, odds, min_edge=0.0)
    placed = settle(bets, y=[0, 1])
    # Match 1: best EV is home (0.6 * 2 - 1 = 0.2) and it wins -> +1.0
    # Match 2: all EVs are <= 0 -> no bet
    assert placed["profit"].tolist() == [1.0]
