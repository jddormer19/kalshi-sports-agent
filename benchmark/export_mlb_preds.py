"""Export point-in-time MLB model predictions for the 2026 season (walk-forward: trained only on earlier seasons).

Run with the MLB model project's environment, from that project's root (a separate private repo; PYTHONPATH=.):
    .venv/bin/python <this file> <out.csv>
"""
import sys

import pandas as pd

from mlbpred.backtest import walk_forward
from mlbpred.cli import ALL_GAME_TYPES
from mlbpred.config import load_settings
from mlbpred.features import FeatureStore
from mlbpred.history import load_history
from mlbpred.teams import _TEAMS

settings = load_settings()
games, logs = load_history(settings.data_dir)
store = FeatureStore(games, logs)
first = settings.history_first_season + 1
frame = store.build_training_frame(ALL_GAME_TYPES, first)
pred = walk_forward(frame, [2026], "logreg", first)
print("frame columns:", list(frame.columns))
print("2026 predictions:", len(pred), "| trained on seasons", first, "-", 2025)

abbr = {t.id: t.abbrev for t in _TEAMS}
keep = [c for c in ("game_pk", "season", "game_type", "game_datetime", "home_id", "away_id", "home_win", "p_model")
        if c in pred.columns]
out = pred[keep].copy()
if "game_datetime" not in out.columns and "game_pk" in out.columns:
    out = out.merge(games[["game_pk", "game_datetime"]], on="game_pk", how="left")
out["home_abbr"] = out.home_id.map(abbr)
out["away_abbr"] = out.away_id.map(abbr)
out.to_csv(sys.argv[1], index=False)
print("wrote", len(out), "rows ->", sys.argv[1])
print(out.head(3).to_string(index=False))
