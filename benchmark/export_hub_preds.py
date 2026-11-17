"""Export point-in-time NBA and NFL model predictions (walk-forward: each season predicted by a model fit only on
earlier seasons). Run from the NBA/NFL model project's root (a separate private repo) with HUB_DATA_DIR pointing at a COPY of data/ (the live server holds the lock):
    HUB_DATA_DIR=<copy of data> PYTHONPATH=. .venv/bin/python <this file> <out_dir>
NBA uses the same frame as the live predictor (historical games + current-season games from ESPN), because the
historical table drops a season's games until teams reach 40 games (the exhibition filter).
"""
import sys
from pathlib import Path

import pandas as pd

from sports.nba import backtest as nba_bt
from sports.nba import features as nba_features
from sports.nba.engine import NBAEngine
from sports.nfl import backtest as nfl_bt

out_dir = Path(sys.argv[1])

# NBA: 2026 (2025-26 incl. playoffs) and 2027 (2026-27, started Oct 2026)
games, tg = NBAEngine()._frames()
nba = games.merge(nba_features.build(games, tg), on="game_id")
nba["margin"] = nba.home_score - nba.away_score
nba["total"] = nba.home_score + nba.away_score
nba_pred = nba_bt.walk_forward_preds(nba, [2026, 2027])
nba_out = nba_pred[["game_id", "season", "start", "home", "away", "home_score", "away_score", "pred_p_home"]].copy()
nba_out["home_win"] = (nba_out.home_score > nba_out.away_score).astype("Int64").where(nba_out.home_score.notna())
nba_out.to_csv(out_dir / "model_preds_nba.csv", index=False)
print("NBA:", len(nba_out), "predictions; by season:", nba_out.season.value_counts().sort_index().to_dict())

# NFL: 2026 season
nfl = nfl_bt.dataset()
nfl_pred = nfl_bt.walk_forward_preds(nfl, [2026])
cols = [c for c in ("game_id", "season", "week", "gameday", "gametime", "home", "away", "home_team", "away_team",
                    "margin", "pred_p_home") if c in nfl_pred.columns]
nfl_out = nfl_pred[cols].copy()
nfl_out["home_win"] = (nfl_out.margin > 0).astype("Int64").where(nfl_out.margin.notna() & (nfl_out.margin != 0))
nfl_out.to_csv(out_dir / "model_preds_nfl.csv", index=False)
print("NFL:", len(nfl_out), "predictions; columns:", list(nfl_out.columns))
