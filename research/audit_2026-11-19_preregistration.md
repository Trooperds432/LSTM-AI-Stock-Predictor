# Audit preregistration: LSTM probability strategy (2026-11-19)

Written **before** running any of the audit tests below. Nothing in this file is changed after results are seen; any later change is logged as a new trial in the results file.

## Object under test
- Model: Conv1D+LSTM with MC dropout, retrained 2026-11-19 (TF 2.21, early stopping on val AUC, best epoch 10, val AUC 0.605). Forecasts in `forecasts/` (227 tickers), test window 2021-06-11 to 2023-06-30.
- Strategy (frozen defaults, chosen before any results were seen): adj_prob = p − 1.0·σ; enter if adj_prob > 0.25; best horizon per ticker; up to 5 equal-weight slots; 20 bps per side; skip price < $5; exclude warrants/units/rights/preferreds.

## Hypotheses
- **H0 (null):** the model's probabilities carry no information for ranking stocks; any outperformance comes from luck, from look-ahead in the backtest, or from the universe and rules themselves.
- **H1:** ranking by the model's uncertainty-adjusted probability produces risk-adjusted returns above (a) SPY and (b) the same rules applied to randomly reassigned probabilities.
- **Economic rationale (weak, stated honestly):** short-horizon technical patterns, such as reversal and volatility effects, have documented but small predictive power (e.g. Jiang–Kelly–Xiu 2023; Fischer–Krauss 2018). There's no specific causal story for why this model's top picks should outperform, so the prior is low.

## Execution fixes applied to the main test (decided before the run)
1. **Next-bar fill:** a signal computed from day t's close fills at day t+1's close (the forecast files have no open price). The exit is the close `horizon` trading days after entry.
2. **Causal spike filter:** a stock is ineligible on day t if any daily move greater than 80% has occurred in its history **up to and including t**. This replaces the old full-period ticker removal, which looked ahead.

## Tests and preregistered pass/fail criteria (main config only)
| # | Test | Fails if |
|---|------|----------|
| 1 | Main run (next-bar, causal filter, 20 bps) | total return ≤ SPY over the same window |
| 2 | Negative control: shuffle probabilities across tickers within each day, 500 permutations, same rules | one-sided p-value (share of shuffles with total return ≥ main) > 0.05 |
| 3 | Block bootstrap (20-day blocks, 5,000 draws) of daily strategy returns | 95% CI of annualized Sharpe includes 0 |
| 4 | Deflated Sharpe Ratio (Bailey & López de Prado 2014), N = all trials so far (≥ 22 earlier + audit runs), trial Sharpe variance from the logged sweeps | DSR < 0.95 |
| 5 | Remove the top 3 trades by dollar P&L | remaining return ≤ SPY |
| 6 | Cost stress at 40 and 60 bps per side | 40 bps run total return ≤ SPY |
| 7 | Subperiods: 2021-06-11 → 2022-06-10 and 2022-06-11 → 2023-06-30 | strategy trails SPY in both subperiods |

Also reported for information only, not as pass/fail: Newey–West t-stat of mean daily return (10 lags), Sortino, profit factor, time in market, and turnover. Attribution runs (same-bar fill and old full-period filter) quantify how much each bias contributed.

**Verdict rule:** the edge counts as *supported* only if tests 1–4 all pass. If any of 1–4 fails, the edge is *not supported* on current evidence. Tests 5–7 are labeled robustness *warnings*.

## Known limitations that this audit does not fix
- Single train/test split and a single random seed; no walk-forward.
- Universe is an alphabetical slice (A–B tickers) with no delisted names, so survivorship is unknown.
- No validation-period forecasts exist, so parameters can't be tuned legitimately; the defaults stay frozen.
- Fills at the close with a flat cost assumption; no spread or volume data, so capacity is unknown.
