# Audit results: LSTM probability strategy (2026-11-19)

Preregistration: `audit_2026-11-19_preregistration.md` (written before these runs). Raw output: `audit_2026-11-19_results.json`. Main-run trades: `audit_2026-11-19_main_trades.csv`. Code: `backtest_core.py` (10 unit tests in `tests/`), `audit_backtest.py`. Seed 20261119. Python 3.13, TF 2.21 env at `C:\Users\Jonat\venvs\lstm-predictor`.

## Verdict: edge NOT supported on current evidence
Three of the four core tests fail. The earlier +37% headline was mostly a backtest artifact; what remains is +13.8% over two years on 25 trades, which is statistically indistinguishable from random picks.

## Attribution: where the +37% came from
Same model, same forecasts, same frozen parameters (threshold 0.25, σ-penalty 1.0, 5 slots, 20 bps per side, $5 min), 212 tickers, window 2021-06-11 to 2023-06-30.

| Run | Fill | Spike filter | Return | Sharpe | Max DD | Trades |
|---|---|---|---|---|---|---|
| A (old, reproduces prior +37%) | same close as signal | full-period (look-ahead) | +37.0% | 0.77 | −23.7% | 30 |
| B | next close | full-period | +13.8% | 0.38 | −25.9% | 25 |
| C | same close | causal | +37.0% | 0.77 | −23.7% | 30 |
| **MAIN** | **next close** | **causal** | **+13.8%** | **0.38** | **−25.9%** | **25** |
| SPY (price only, no dividends) | | | +4.5% | 0.21 | −25.4% | |

- **Same-close fill accounted for 23 of the 37 points.** The signal used day t's close, and the old backtest bought at that same close. One day of delay removes most of the gain. That pattern is typical of a short-term reversal or noise effect that disappears once you can't trade at the exact price that triggered the signal.
- **The spike filter made no difference.** The look-ahead version and the causal version picked the same trades.

## Preregistered tests (MAIN)
| # | Test | Result | Pass? |
|---|---|---|---|
| 1 | Return vs SPY | +13.8% vs +4.5% | PASS |
| 2 | Shuffle control (500 within-day permutations of probabilities) | p = 0.168; null median −12.4%, 95th percentile +30.2% | **FAIL** |
| 3 | Block bootstrap (20-day blocks, 5,000 draws) | Sharpe 0.38, 95% CI [−0.99, 1.70]; IR vs SPY 0.30, CI [−1.20, 1.85]; total return CI [−44%, +126%] | **FAIL** |
| 4 | Deflated Sharpe (N = 28 trials) | DSR = 0.30 (benchmark SR₀ ≈ 0.74 annualized); PSR vs 0 = 0.71 | **FAIL** |
| 5 | Remove the top 3 trades | −16.9% (the top 3 were 223% of net P&L) | warning |
| 6 | Cost stress | 40 bps: +11.6%; 60 bps: +9.5% | PASS |
| 7 | Subperiods | P1 (Jun 21–Jun 22): +18.2% vs SPY −8.1%; P2 (Jun 22–Jun 23): −3.7% vs SPY +13.7% | PASS (beats in 1 of 2) |

Information only: Newey–West t-stat of mean daily return = 0.56; Sortino 0.57; win rate 60%; average win +14.9% vs average loss −14.7%; profit factor 1.39; time in market 99%; annual turnover about 4.8×.

## Interpretation
- **Same-close fill:** the earlier result relied on trading at the exact close that produced the signal. That's not achievable in practice.
- **Shuffle control:** after the fix, randomly reassigning the model's probabilities among the same stocks beats the strategy in about 17% of shuffles. The ranking is not shown to add value beyond the universe and rules.
- **Bootstrap and t-stat:** the Sharpe CI spans roughly −1 to +1.7 and the t-stat is 0.56. Two years of daily data cannot distinguish this from zero.
- **Deflated Sharpe:** with 28 configurations tried on the same test window, the best-of-N benchmark Sharpe is about 0.74. The observed 0.38 is well below it.
- **Concentration:** profit depends on a handful of trades. Without the top 3 the strategy loses 17%.
- **Regime:** all the outperformance came in the 2021–22 decline. In the 2022–23 recovery it lost while SPY gained 14%.

## Known limitations (not fixed by this audit)
- Single train/test split and a single training seed; no walk-forward.
- Alphabetical A–B universe with no delisted names, so survivorship is unknown.
- No validation-period forecasts, so legitimate tuning is impossible.
- Close-to-close fills with flat costs; no spread or volume data, so capacity is unknown. SPY benchmark excludes dividends.

## What would change the verdict (next experiments, each to be preregistered)
1. **Walk-forward retraining:** e.g. yearly refits over 2015–2023 with purge and embargo of 126 days (the longest label horizon) to get many out-of-sample years instead of one window.
2. **Several training seeds:** at least 5, reporting the spread of results.
3. **Validation-period forecasts:** save them so thresholds can be set without touching the test period.
4. **A wider universe:** large and liquid, including delisted names, to raise the trade count enough for any test to have power.
5. **A simpler baseline:** check whether a plain short-term reversal or momentum rank under the same rules matches the model. If it does, the LSTM adds nothing.

## Trial log
- 22 trials before the audit (11 per model): `research/trials_2026-11-19/`.
- 6 audit runs (A, B, C, MAIN, cost40, cost60).
- Total N = 28 for the deflated Sharpe. Shuffles and bootstraps are controls, not trials.
