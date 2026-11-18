# Benchmark spec: can an agent beat the market on sports game winners?

Frozen 2026-11-13, before any agent code. Changing a rule after the holdout has been scored invalidates the
benchmark; any change before that is logged at the bottom with its date and reason.

## 1. Question
At a fixed time before each game, which forecast of "the home team wins" is most accurate: the Kalshi market price,
my own models, or an LLM research agent that can use my models and point-in-time news? The market is the bar to beat.

## 2. Universe
- Settled Kalshi game-winner series: `KXMLBGAME`, `KXNBAGAME`, `KXNFLGAME` (crawled 2026-11-13: 6,870 settled games,
  13,740 markets, Apr 2025 – Nov 12, 2026; `run.py collect-games`, raw data in `data/games/`).
- **Included:** games whose scheduled start time can be determined from Kalshi's own data (rule 4). As of the crawl that
  is 2,989 games: MLB 2,481 (Mar 25 – Oct 28, 2026), NBA 322 (from Apr 17, 2026), NFL 186 (from Aug 6, 2026).
- **Excluded, with a count printed for each reason:**
  - games with no determinable start time (all 3,133 games from 2025, and NBA/NFL 2026 games missing `occurrence_datetime`);
  - NFL preseason games (my NFL model covers regular season and playoffs only);
  - NBA preseason games (Sep 1 - Oct 18, 2026): no official injury reports, and starters rest unpredictably;
  - voided, postponed or tied games, i.e. anything where exactly one side did not settle YES;
  - games with no usable price at the decision time (rule 5).
- Why only Kalshi's own data for start times: a single source applied the same way to every game, with no extra
  scraping or licensing questions. Adding 2025 games from an open-licensed schedule (e.g. nflverse) is a possible v2.

## 3. Unit of analysis
- **One row per game**, not per market. Kalshi lists every game twice ("Team A wins", "Team B wins"); the two prices
  are near mirror images, so counting both would double the sample size without adding information and make
  confidence intervals look tighter than they are.
- The row uses the **home team's market**. Label = 1 if the home team won, 0 if it lost.
- Home/away is read from the event code (`…NYYMIN` = away NYY, home MIN), with the convention checked by hand on at
  least 3 games per sport against a published schedule.

## 4. Decision time t
- **t = scheduled start − 60 minutes.**
- Start time: the ticker time when present (`KXMLBGAME-26SEP161340…` = Sep 16, 13:40 **US Eastern**); otherwise
  `occurrence_datetime − 3 hours`. Checked on 2026-11-13: `occurrence_datetime` is exactly 3 h after the ticker time on
  3,884 MLB markets (5th–95th percentile: 180 min), and minus 3 h it lands on standard start times in every sport
  (NFL 13:00 / 16:25 / 20:15 ET, NBA 19:00 / 19:30 / 20:00 ET). It is a scheduled end time, not a start time.
- Why 60 minutes: by then MLB lineups and NFL inactives are public, so the market has its information and the test is
  hard. Agents can't win just by being earlier than the news. A t = start − 24 h variant may be added later as a second,
  separately reported benchmark.

## 5. Market baseline at t
- Market probability = **mid of the YES bid and ask** from the last hourly candle whose period **ended at or before t**.
  A candle ending after t is never used, even if it is only minutes late.
- Excluded: no candle before t, a missing bid or ask, or a spread wider than **10¢** (the price is not a real consensus).
- The spread at t is kept in the table so results can later be split by liquidity.

## 6. Information rule (no lookahead)
- Every input a system uses (prices, model features, news, injury reports) must be timestamped **before t**.
- Every retrieval and tool call is logged with the timestamps of what it returned. Any item dated after t marks the
  run as **failed**, not just penalized, and the leakage count is reported next to the scores.
- The leakage audit runs automatically on every evaluation, including the baselines.

## 7. Split
- **Chronological, per sport:** within each sport, the earliest 70% of games by start time are **dev**, the latest 30%
  are **holdout**, cut at a **date boundary** (US Eastern) so no single day's games are split across the two sets;
  the resulting shares are approximate and reported per sport. Per sport so every sport appears in both sets (a single global date cutoff would put most NBA
  and NFL games in the holdout and almost none in dev).
- Dev is for building, debugging and tuning prompts and tools. The holdout is scored **once**, at the end, with the
  final frozen system. If the holdout is ever looked at early, that is written down here.

## 8. Systems compared
1. **Market:** mid at t (rule 5).
2. **Recalibrated market:** the market price passed through a calibration curve fit on dev only.
3. **My model alone:** my existing MLB / NFL / NBA win-probability models, using only data available before t.
4. **Agent + my model:** the LangGraph agent with my models, market data and point-in-time news as tools.
5. **Agent without my model (ablation):** same agent, model tools removed. Shows what the model adds.

## 9. Metrics and decision rule
- **Primary:** Brier score (lower is better), pooled across sports and reported per sport.
- **Secondary:** log loss; calibration (reliability plot, 10 bins); cost in $ and seconds per forecast; tool calls per run;
  leakage failures (should be 0).
- **Comparison with the market:** paired bootstrap over games (10,000 resamples) of the Brier difference
  (system − market), with a 95% confidence interval.
- **Decision rule, fixed now:** a system "beats the market" only if the whole 95% interval is below zero on the holdout.
  Anything else is reported as no detectable edge, with the interval. A null result is reported as fully as a win.
- Per-sport results for NBA (~95 holdout games) and NFL (~55) are directional only: too few games for firm conclusions.

## 10. Known limitations
- Hourly candles are coarse: the price at t can be up to an hour old.
- Only 2026 games are included (start-time coverage); the benchmark spans one MLB season and partial NBA/NFL seasons.
- NBA and NFL samples are small, so pooled results are dominated by MLB.
- The home-side price and the away-side price can differ by a cent or two (spread); using one side is a small choice
  that could matter for very close comparisons.
- Kalshi prices include traders' own models, so "beating the market" is a high bar by design.

## Change log
- 2026-11-13: spec frozen (no holdout results seen).
- 2026-11-17: rule 7 clarified: split at a date boundary so same-day games stay together (no holdout results seen).
- 2026-11-17: added the t = start - 24h variant (data/benchmark_24h.parquet, built with `--lead-hours 24`; 2,921 games with the SAME dev/holdout membership as the 1h benchmark, which is unchanged). At t - 24h, model_win_prob is excluded (its inputs run to game-day midnight, after t). News sources and their timestamp limits: Desktop audit 2026-11-17 (NBA official injury PDFs, 15-min; MLB Stats API feed snapshots, t - 1h only; nflverse injuries, week-level with a conservative rule). No holdout results seen.
- 2026-11-18: rule 2: NBA preseason games excluded (67 games, all dev). Found when the injury tool had no reports for them; the NBA 24h-vs-1h price gap was concentrated in preseason. Dev/holdout membership of every remaining game is unchanged (frozen split file). No holdout results seen.
- 2026-11-18: **holdout scoring pre-registered** (written and committed before any holdout result is computed; run once with `evals/score_holdout.py`, which refuses a second run). Systems: (1) market mid at t = start - 60 min on all holdout games; (2) my model alone and (3) a blend, logistic regression on logit(market) and logit(model) fit on ALL dev games with both, on holdout games with a model prediction; (4) agent v1 = agent/graph.py as of commit d5394fb, claude-opus-5-5 at effort medium, tools get_game, market_price, model_win_prob, team_recent_form, on 100 holdout games with a model prediction, stratified by sport (seed 0). Primary metric Brier, plus log loss; paired bootstrap vs the market (10,000 resamples); an edge is claimed only if the 95% CI excludes zero. Results are reported as they come out, including nulls. The t - 24h variant and the injury tool were tested on dev only and are not part of this holdout run.
- 2026-11-18: commit IDs above changed when git history was rewritten to remove private file paths before publishing (code unchanged): agent v1 pin d5394fb is now 878556f.
