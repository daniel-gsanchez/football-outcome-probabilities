"""Build leakage-free inputs: each match only sees matches played before it."""
from collections import defaultdict

import numpy as np
import pandas as pd

# Per-match features stored in each team's history.
FEATURES = ("goals_for", "goals_against", "goal_diff", "points", "was_home")

# Chronological split by season: train up to 2013/14, validate on 2014/15,
# test on 2015/16 (the last season in the database).
VAL_START = pd.Timestamp("2014-07-01")
TEST_START = pd.Timestamp("2015-07-01")


def match_row(goals_for, goals_against, was_home):
    """History row for one team in one match (order = FEATURES)."""
    diff = goals_for - goals_against
    points = 3 if diff > 0 else (1 if diff == 0 else 0)
    return [goals_for, goals_against, diff, points, int(was_home)]


def build_sequences(matches, window):
    """Last `window` matches of each team before every match.

    Matches are processed in chronological order: a sample is created from
    the history *before* the match is added to it, so the model never sees
    the result it has to predict. Matches where either team has fewer than
    `window` previous games are skipped.
    """
    history = defaultdict(list)
    rows = {k: [] for k in ("index", "home_seq", "away_seq")}

    for i, m in enumerate(matches.itertuples(index=False)):
        h, a = history[m.home_team], history[m.away_team]
        if len(h) >= window and len(a) >= window:
            rows["index"].append(i)
            rows["home_seq"].append(h[-window:])
            rows["away_seq"].append(a[-window:])
        h.append(match_row(m.home_goals, m.away_goals, True))
        a.append(match_row(m.away_goals, m.home_goals, False))

    idx = np.array(rows["index"])
    return {
        "index": idx,  # row position in `matches`
        "home_seq": np.array(rows["home_seq"], dtype=np.float32),
        "away_seq": np.array(rows["away_seq"], dtype=np.float32),
        "home_team": matches["home_team"].to_numpy()[idx],
        "away_team": matches["away_team"].to_numpy()[idx],
        "league": matches["league_id"].to_numpy()[idx],
        "date": matches["date"].to_numpy()[idx],
        "y": matches["result"].to_numpy()[idx],
    }


def split_masks(dates):
    dates = pd.to_datetime(dates)
    return {
        "train": dates < VAL_START,
        "val": (dates >= VAL_START) & (dates < TEST_START),
        "test": dates >= TEST_START,
    }


class SequenceScaler:
    """Standardize each feature using training sequences only."""

    def fit(self, *seqs):
        flat = np.concatenate([s.reshape(-1, s.shape[-1]) for s in seqs])
        self.mean_, self.std_ = flat.mean(axis=0), flat.std(axis=0)
        return self

    def transform(self, seq):
        return ((seq - self.mean_) / self.std_).astype(np.float32)


def summary_features(home_seq, away_seq):
    """Tabular baseline input: window averages (home, away, home - away)."""
    h = home_seq[:, :, :4].mean(axis=1)  # goals, goal difference and points
    a = away_seq[:, :, :4].mean(axis=1)
    return np.concatenate([h, a, h - a], axis=1)


def index_map(values):
    """Map raw ids (teams, leagues) to 0..n-1 for the embedding layers."""
    return {int(v): i for i, v in enumerate(np.unique(values))}
