# Walk-forward results: LSTM probability strategy (evaluated 2026-11-20)

- **Preregistration:** `walkforward_2026-11-19_preregistration.md`.
- **Raw output:** `walkforward_2026-11-19_results.json`. Trades: `walkforward_2026-11-19_trades.csv`.
- **Code:** `walk_forward.py`, `wf_evaluate.py`, `backtest_core.py` (close-only path).
- **Run:** 8 folds (test years 2016–2023) × 3 seeds = 24 models, all completed.

## Verdict: edge NOT supported on current evidence
Tests 1–3 pass, but test 4, the deflated Sharpe, fails. This is the strongest result so far, and it is still not enough.

## Data caveats (read first)
These forecasts come from the **old** processed data:
- **Universe:** an alphabetical A–B slice of 216 tickers in which almost every name survives to 2023 (survivorship bias).
- **Leaks:** the congressional-trade and insider features had the look-ahead leaks fixed in commit `849e665`. Only 19–28 tickers carry those features.

The result is a baseline record only. It must be repeated on the point-in-time S&P 500 data layer (`TrainingData/pit/`) before any claim can be made.

## Primary (seed ensemble, frozen strategy, next-close fills, 20 bps)

| | Strategy | SPY (price only) | Equal-weight universe | Reversal baseline |
|---|---|---|---|---|
| CAGR | +13.1% | +11.5% | +6.2% | Sharpe 0.06 |
| Sharpe | 0.72 | 0.69 | 0.39 | |
| Max drawdown | −28.9% | −34.1% | −44.1% | |

Trades 142; win rate 61%; Newey–West t-stat 2.04.

| # | Test | Result | Pass |
|---|---|---|---|
| 1 | CAGR vs SPY | 13.1% vs 11.5% | PASS |
| 2 | Shuffle control (500 within-day permutations) | p = 0.032 | PASS |
| 3 | Block-bootstrap Sharpe 95% CI | [0.07, 1.41] | PASS |
| 4 | Deflated Sharpe (N = 30 trials) | DSR = 0.39 | **FAIL** |
| 5 | Remove top 3 trades | CAGR ≤ SPY | warning |
| 6 | Years beating SPY | 3 of 8 | warning |
| 7 | Seed stability (single-seed Sharpe) | 0.99 / 1.21 / 1.23 | pass |
| 8 | vs reversal baseline | 0.72 vs 0.06 | pass |
| 9 | vs equal-weight universe | 13.1% vs 6.2% | pass |
| 10 | Cost stress, 40 bps | CAGR ≤ SPY | warning |

**Year by year, strategy vs SPY:**
- 2016: +20.0% vs +11.2%
- 2017: +22.6% vs +19.4%
- 2018: −15.6% vs −6.3%
- 2019: +5.7% vs +28.8%
- 2020: +80.6% vs +16.2%
- 2021: +13.1% vs +27.0%
- 2022: −6.8% vs −19.5%
- 2023: +6.0% vs +24.3%

**Model diagnostics:** mean best validation AUC 0.598. Mean test AUC by horizon: 1d 0.564, 1w 0.573, 1m 0.585, 6m 0.591.

## Secondary (nested threshold chosen on validation; informational only)
CAGR +27.8%, Sharpe 1.35. Thresholds chosen per year:
- 2016: 0.30
- 2017: 0.30
- 2018: 0.20
- 2019: 0.25
- 2020: 0.25
- 2021: 0.20
- 2022: 0.25
- 2023: 0.20

The thresholds were picked only on each fold's validation window, so this is a legitimate out-of-sample procedure. It is still a second candidate and is counted in N. It does not override the primary verdict.

## Interpretation
- **Rankings carry information.** For the first time the shuffle control passes: the model's ranking beats random picks from the same universe under the same rules. Test AUC is a modest 0.56–0.59 but consistent across horizons and seeds.
- **Not enough after correcting for the search.** With 30 configurations tried, the deflated Sharpe benchmark is far above the observed Sharpe. Performance is also concentrated: 2020 alone returned +81%, and the strategy beat SPY in only 3 of 8 years. It does not survive 40 bps costs.
- **The survivor-biased universe flatters equal-weight comparisons**, but it should not flatter the shuffle control, which uses the same universe.

## Next
Repeat the identical design on the point-in-time S&P 500 universe:
- membership mask
- next-open fills
- spread-based costs
- settlement of delisted names

Preregister it first, with N = 30 carried forward.
