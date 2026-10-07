"""REST API: pre-match probabilities from each team's most recent matches.

Run from the repository root:
    uvicorn api.app:app
Interactive docs: http://127.0.0.1:8000/docs
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football import CLASSES  # noqa: E402
from football.features import match_row  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
model = tf.keras.models.load_model(ARTIFACTS / "model.keras")
prep = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
WINDOW = prep["window"]
MEAN, STD = np.array(prep["mean"]), np.array(prep["std"])
TEAMS = {int(k): v for k, v in prep["teams"].items()}
TEAM_NAMES = {int(k): v for k, v in prep["team_names"].items()}
LEAGUES = {int(k): v for k, v in prep["leagues"].items()}


class PastMatch(BaseModel):
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    was_home: bool


class MatchRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    league_id: int
    home_last_matches: list[PastMatch] = Field(description="Oldest first")
    away_last_matches: list[PastMatch] = Field(description="Oldest first")


class Forecast(BaseModel):
    match: str
    probabilities: dict[str, float]
    fair_odds: dict[str, float]


app = FastAPI(title="Football outcome probabilities", version="1.0")


def encode(history):
    rows = [match_row(m.goals_for, m.goals_against, m.was_home) for m in history]
    return ((np.array(rows, dtype=float) - MEAN) / STD)[None].astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "window": WINDOW}


@app.post("/predict", response_model=Forecast)
def predict(req: MatchRequest):
    for team in (req.home_team_id, req.away_team_id):
        if team not in TEAMS:
            raise HTTPException(404, f"Unknown team id {team}")
    if req.league_id not in LEAGUES:
        raise HTTPException(404, f"Unknown league id {req.league_id}")
    for side in (req.home_last_matches, req.away_last_matches):
        if len(side) != WINDOW:
            raise HTTPException(422, f"Each team needs exactly its last {WINDOW} matches")

    inputs = [encode(req.home_last_matches), encode(req.away_last_matches),
              np.array([TEAMS[req.home_team_id]]), np.array([TEAMS[req.away_team_id]]),
              np.array([LEAGUES[req.league_id]])]
    probs = model.predict(inputs, verbose=0)[0].astype(float)
    return Forecast(
        match=f"{TEAM_NAMES[req.home_team_id]} vs {TEAM_NAMES[req.away_team_id]}",
        probabilities={c: round(p, 4) for c, p in zip(CLASSES, probs)},
        fair_odds={c: round(1 / p, 2) for c, p in zip(CLASSES, probs)},
    )
