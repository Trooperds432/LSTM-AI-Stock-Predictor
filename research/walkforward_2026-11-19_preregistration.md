# Walk-forward preregistration: LSTM probability strategy (2026-11-19)

Written **before** any walk-forward model is trained or evaluated. Results go in `walkforward_2026-11-19_results.md`. Any change after results are seen is logged as a new trial.

## Question
Does the Conv1D+LSTM probability ranking produce risk-adjusted returns above SPY, above an equal-weight hold of the same universe, and above shuffled signals, when the model is **re-trained every year using only data available before that year**?

- **H0:** no information in the ranking. Outperformance, if any, is due to luck, the universe, or the rules.
- **H1:** the ranking adds value out of sample across many years.
- **Prior:** low. The single-window audit (`audit_2026-11-19_results.md`) found no supported edge.

## Design (frozen)
- **Test years** Y = 2016 … 2023 (8 folds). Earlier years lack enough training history; only 93 tickers have data from 2011.
- **Sample date d, label end e(d) = d + 126 trading days** (the longest horizon). For fold Y:
  - **Test:** d ∈ [Y-01-01, Y-12-31]. Daily forecasts.
  - **Validation:** d ∈ [V0, V1), with V1 = first trading day of Y minus 126 trading days and V0 = V1 minus 252 trading days. Labels therefore end before the test year starts.
  - **Train:** d < V0 minus 126 trading days (purge and embargo, so training labels end before validation samples begin).
- **Fitted on training rows only:** per-stock label thresholds (mean + 1σ of the forward log return, the same rule as the notebook), feature scaler, and the drop-list of constant features. Price-level features are made relative to close; `close` and `OBV` are excluded, as in the current notebook.
- **Model:** identical architecture to the notebook (Conv1D 32 → BN → MCDropout 0.3 → LSTM 64 → MCDropout 0.3 → Dense 32 → Dense 4 sigmoid). Adam, BCE, batch 128, at most 100 epochs, early stopping on val AUC (patience 8, warm-up 3, restore best), ReduceLROnPlateau (factor 0.5, patience 3). Window 60+1, training step 21 days, validation step 21, inference step 1. MC dropout with 25 samples.
- **Seeds:** 3 per fold (seed = 1000·Y + k, k = 0, 1, 2), with TF op determinism on.
  - **Primary signal = seed ensemble:** p = mean of the seed means; σ = sqrt(mean within-seed MC variance + variance across seed means).
- **Universe:** every processed ticker with data, excluding warrants, units, rights and preferreds. Known survivorship bias: 234 of 236 tickers are still listed in 2023.

## Strategy (frozen, same as the audit)
- adj = p − 1.0·σ; enter if adj > 0.25.
- Best horizon per ticker; up to 5 equal-weight slots.
- 20 bps per side; skip prices below $5; causal 80% spike filter.
- **Next-day close fill.** One continuous backtest over 2016-01-04 → 2023-12-29, each day using forecasts from the model trained before that year. Starting capital $5,000. Engine: `backtest_core.py` (unit-tested).

## Comparators (same window, same costs where they trade)
1. SPY buy and hold (price only).
2. Equal-weight hold of the eligible universe (daily rebalanced, no costs, which flatters the comparator).
3. **Short-term reversal baseline**, same engine and rules: score_1w = cross-sectional percentile rank of −(5-day return), other horizons blank, σ = 0, threshold 0.0, 5 slots, 20 bps, next-day fill.
4. Shuffle control: 500 within-day permutations of (p, σ) across tickers, same rules.

## Primary tests (ensemble signal, frozen strategy)
| # | Test | Fails if |
|---|---|---|
| 1 | CAGR vs SPY | strategy CAGR ≤ SPY CAGR |
| 2 | Shuffle control, 500 permutations | one-sided p (total return) > 0.05 |
| 3 | Block bootstrap (20-day blocks, 5,000 draws) | 95% CI of annualized Sharpe includes 0 |
| 4 | Deflated Sharpe, N = 28 prior trials + walk-forward candidate configs (primary + secondary = 2) → N = 30 | DSR < 0.95 |

**Verdict:** the edge counts as *supported* only if tests 1–4 all pass.

## Robustness warnings (reported, not part of the verdict)
| # | Check | Warning if |
|---|---|---|
| 5 | Remove the top 3 trades by dollar P&L | CAGR ≤ SPY |
| 6 | Year by year | beats SPY in fewer than 5 of 8 years |
| 7 | Seed stability: backtest each single seed | any single-seed Sharpe ≤ 0 |
| 8 | Beats the reversal baseline | Sharpe ≤ reversal baseline Sharpe |
| 9 | Beats the equal-weight universe | CAGR ≤ EW CAGR |
| 10 | Cost stress at 40 bps | CAGR ≤ SPY |

## Secondary analysis (preregistered, nested selection)
For each fold, choose `min_accepted` from {0.20, 0.25, 0.30, 0.35} by the highest Sharpe on that fold's **validation** forecasts (same engine and rules, fills inside the validation window). Then apply it to that fold's test year. Report the stitched result with the same tests 1–3. It's informational and doesn't override the primary verdict.

## Also reported
Per-fold val AUC, test AUC per horizon, best epoch, number of training samples, and tickers per fold. Newey–West t-stat, Sortino, win rate, profit factor, turnover, and time in market.

## Known limitations
- Survivorship bias in the universe.
- Close-to-close fills with flat costs, and no capacity analysis.
- The SPY benchmark excludes dividends.
- The label measures from day d's close, but the trade enters at d+1's close (a mismatch kept on purpose, because it is how the model is trained).
- With only 3 seeds, seed variance is roughly estimated.
