# Execution-latency attribution: preregistration (2026-11-20)

Written **before** any run in this study. Results go in `latency_2026-11-20_results.md` / `.json`.
Code: `latency_attribution.py` (tested in `tests/test_latency_attribution.py`), engine `backtest_core.py`.
Any change made after results are seen is logged as a new trial.

## Question
The same-close backtest (+37.0%, 30 trades) beat the next-close backtest (+13.8%, 25 trades) by 23 points.
Where does that gap come from, and is any of it plausibly capturable by a signal computed before the close
(3:40 PM) and filled in the closing auction?

### Framing
Row t's features use day t's completed bar (`TrainingData/processor.py:307-429`), so the signal exists only
after 4:00 PM.
- **A, `same_close`:** information through bar t, filled at close t. Not implementable.
- **C, `next_close`:** the same signal filled at close t+1. Shifted by one day, this is a market-on-close order at
  close t using information through bar t-1. It is implementable today.
- **D, a 3:40 PM model:** information through bar t-1 plus the partial day t, filled at close t. It lies between C and A.

So the A-C gap is the value of day t's information when trading at close t. A 3:40 model can capture only the part
that is already known by 3:40.

## Hypotheses
- **H1:** the A-C gap is mostly an entry-price effect on the same signals, not a change in which trades are selected.
- **H2:** the overnight part of the gap (close t -> open t+1) is positive at the signal level.
- **Prior:** a large part of the gap may be one-day reversal or bid-ask bounce in closing prints of small, A-B names.
  That would not be a timing edge.

## Data (frozen)
- **Legacy set:** `forecasts/*_forecast.csv`, panel from `split_info.json` `oos_start` (2021-06-10). This window is
  burned for selection; it is used for attribution only.
- **Walk-forward set (primary for the gate):** the seed ensemble of `wf/forecasts/Y*_s*_test.csv.gz`, 2016-2023,
  built with `wf_evaluate.ensemble` / `to_dict`.
- **Opens:** `TrainingData/indicators_data/raw/stocksData/{T}_daily.csv` (Alpha Vantage, unadjusted), joined on date
  to each forecast frame. Ticker `X_daily_processed` maps to `X_daily.csv`. Rows where the raw close differs from the
  forecast Close by more than 0.5% are reported, and their Open is set to NaN.
- Nothing on or after 2025-01-02 is read.

## Part 1a: portfolio runs
`StrategyParams()` frozen defaults (20 bps, $5 minimum, causal spike filter, 5 slots, threshold 0.25,
std_factor 1.0). Only `fill` changes: A = `same_close`, B = `next_open`, C = `next_close`.
- **Metrics:** total return, CAGR, Sharpe, max drawdown, trades, costs, time in market, and a 20-day block-bootstrap
  Sharpe 95% CI (2,000 resamples, seed 20261120).
- **Reproduction checks (must hold before anything else is read):**
  - legacy A = +37.0% with 30 trades and C = +13.8% with 25 trades;
  - walk-forward C = 142 trades and CAGR +13.1%.
- **Trial accounting:** B is a new executed configuration, so N goes from 30 to 31. A and C are re-runs and add none.

## Part 1b: same-trade decomposition (on A's trades)
For each A trade (signal and buy at close t, exit price X on A's sell date), compute gross log returns:
- `on` = log(open t+1 / close t)
- `s1` = log(close t+1 / open t+1)
- `rest` = log(X / close t+1)

Then:
- `on + s1 + rest` = the trade's gross log return.
- The C-entry replay of the same trade earns `rest`; the B-entry replay earns `s1 + rest`.
- **Selection effect** = (C portfolio) - (A trades replayed with C entry), compared on sums of trade log returns.
- **Trade matching:** A, B and C trades are matched on (Ticker, SignalDate); report matched and unmatched counts.

## Part 1c: signal-level event study (gate statistic)
- **Candidates:** every (ticker j, day t <= T-2) that passes the engine's eligibility mask on day t (`bc._eligible_mask`)
  with best adjusted probability > 0.25, using the engine's own horizon and std-penalty logic. Open positions are
  ignored, so this is the signal, not the portfolio.
- **Universe on day t:** eligible tickers with finite close t and close t+1.
- **Per event:**
  - `gap` = log(close t+1 / close t), which is the A-over-C entry advantage
  - `on`, `s1` as above
  - `d0` = log(close t / open t)
  - `post` = log(close t+1+h / close t+1), where h is the event's best horizon in sessions
- **Excess:** each component minus the same-day equal-weight universe mean of that component.
- **Split flag:** an event is flagged if |on| > 0.4 or |s1| > 0.4. Primary statistics exclude flagged events;
  results with them included are also reported.
- **Inference:** average the excess component across events on each date, then take the mean of that daily series
  with a Newey-West t-stat (10 lags, `bc.newey_west_t`). Event-weighted means are also reported.
- **Diagnostics:**
  - split by the sign of day t's close-to-close return;
  - split by year.

## Gate G1 (unlocks Stage 2, the LSEG intraday probe, and Stage 3, the pre-close model)
On the walk-forward set, primary statistic (flagged events excluded), mean daily excess `gap`:
1. at least 0.0020 (20 bps). Costs are the same for either entry, so this is a materiality floor of about one side's
   cost, not a cost deduction;
2. Newey-West t >= 2.0;
3. positive in at least 5 of the 8 test years.

All three must hold. If G1 fails, the conclusion is that the same-close advantage is not a timing edge worth an intraday
build, and the work moves to the PIT next-open study instead.

## Gate G2 (only if G1 passes)
An LSEG probe of about 10 RICs must show:
- at least 3 years of 5-minute bars ending before 2025-01-02 for the sampled members, and
- a usable official closing price.

Otherwise the 3:40 PM model can only be evaluated by forward recording (live 15:40 snapshots from now on, evaluated after
a sample size preregistered separately).

## Known limitations
- Prices are unadjusted (splits handled only by the flag in 1c; portfolio runs keep the audit's treatment so they stay
  comparable).
- The A-B universe has survivorship bias. The gap statistic is relative to the same universe, which limits the damage.
- The opens come from a different vendor field than the closes; a mismatch check is reported.
