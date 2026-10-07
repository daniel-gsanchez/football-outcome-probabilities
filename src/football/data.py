"""Load matches and bookmaker odds from the Kaggle European Soccer Database."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DATASET = "hugomathien/soccer"
BOOKMAKERS = {"B365": "Bet365", "PS": "Pinnacle"}


def database_path():
    """Download the dataset with kagglehub (cached after the first call)."""
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET)) / "database.sqlite"


def load_matches(db_path=None):
    """One row per match, sorted chronologically, with result and odds.

    result: 0 = home win, 1 = draw, 2 = away win (same order as the odds).
    """
    odds_cols = [f"{b}{o}" for b in BOOKMAKERS for o in "HDA"]
    query = f"""
        SELECT m.match_api_id, m.date, m.season, m.league_id, l.name AS league,
               m.home_team_api_id AS home_team, m.away_team_api_id AS away_team,
               th.team_long_name AS home_name, ta.team_long_name AS away_name,
               m.home_team_goal AS home_goals, m.away_team_goal AS away_goals,
               {", ".join("m." + c for c in odds_cols)}
        FROM Match m
        JOIN League l ON l.id = m.league_id
        JOIN Team th ON th.team_api_id = m.home_team_api_id
        JOIN Team ta ON ta.team_api_id = m.away_team_api_id
    """
    with sqlite3.connect(db_path or database_path()) as con:
        df = pd.read_sql(query, con)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_api_id"], kind="stable").reset_index(drop=True)
    df["result"] = np.select(
        [df.home_goals > df.away_goals, df.home_goals == df.away_goals], [0, 1], 2
    )
    return df


def odds(df, book):
    """(n, 3) array of decimal odds for home, draw, away (NaN if missing)."""
    return df[[f"{book}H", f"{book}D", f"{book}A"]].to_numpy(dtype=float)


def implied_probabilities(decimal_odds):
    """Remove the bookmaker margin by proportional normalization.

    Returns the implied probabilities (rows sum to 1) and the overround
    (sum of 1/odds; 1.06 means a 6% margin).
    """
    raw = 1.0 / decimal_odds
    overround = raw.sum(axis=1)
    return raw / overround[:, None], overround
