# Kalshi mispricing model

Predicts P(YES) for Kalshi binary markets from point-in-time market state, compares with the
quoted price, and backtests a fee-aware taker strategy. Read-only: it never places orders.

**Start here:** open `kalshi_model.ipynb` (Jupyter or VS Code) for the full walkthrough with charts,
or `kalshi_model.html` in any browser. To re-run the notebook from scratch:
`.venv/bin/jupyter nbconvert --to notebook --execute --inplace kalshi_model.ipynb`

```bash
.venv/bin/python run.py collect     # crawl recurring series + hourly candles (resumable, ~1h)
.venv/bin/python run.py features    # snapshots + leakage check
.venv/bin/python run.py train       # time split, LightGBM correction on market log-odds
.venv/bin/python run.py backtest    # fee-aware, 1 trade/market, event-bootstrap CI
.venv/bin/python run.py predict     # score open markets -> data/live_edges.csv
```

## Design decisions (and why)
- **Universe = recurring short-dated series** (games, daily weather, commodity/index ranges,
  weekly entertainment). A settled-only dataset is survivorship-biased for long-dated markets
  (they're only settled if resolved early, mostly YES). Series whose YES markets close earlier
  than NO markets are dropped.
- **Snapshots sampled across each market's life**, not "N hours before close": close time
  often *is* the resolution moment, so anchoring to it leaks.
- **Model = market log-odds + boosted correction**, Platt-calibrated on validation.
- **Split by time**: train on markets closed before T1, validate after T1, test on snapshots after T2.
- Fees: Kalshi taker `ceil(0.07·C·P·(1−P))`; trades pay the full spread.

## Results (first run, 2026-09-23; 12.4k markets, 317 series)
Test period: snapshots on or after 2026-09-16 (2,450 markets).

| | Brier | vs market |
|---|---|---|
| Market mid | 0.1840 | — |
| Market, recalibrated | 0.1840 | +0.0% |
| Model | 0.1875 | **+1.9% (worse)** |

- Beats the market on test in Weather (−4%), Commodities (−10%) and Financials (−6%).
  Loses in Sports (+3%) and Economics (+8%).
- On validation slices *inside* the training period, every variant beat the market by about 1.5%.
  That edge **did not carry forward** to the later test period (regime shift: football season started).
- Backtest: flat ROI ≈ −2% at small margins (CI includes 0). At a 10¢ margin, +39% on 84 trades
  (CI 9–71%), but that margin was picked from a sweep, so treat it as a hypothesis, not a result.

## ⚠ Known problem: live edges are NOT trustworthy
The model inflates longshots. Its probabilities for mutually exclusive brackets sum well above 1
(WTI weekly brackets: 2.95; weather highs: about 1.2). The top rows of `live_edges.csv` are mostly
this artifact.

## Next steps
1. Normalize p_model within mutually exclusive events (or model brackets jointly), then re-run
   train and backtest.
2. Walk-forward evaluation over several test windows instead of one.
3. Per-category models for Weather/Commodities, with domain features (NOAA forecasts, futures prices).
   Market microstructure alone looks close to efficient.
