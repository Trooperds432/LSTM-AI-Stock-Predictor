# Point-in-time S&P 500 data layer: acceptance tests (preregistered 2026-11-19)

Written before any `pit/` artifact is built. Thresholds are fixed. Any change after results are seen is logged here with its date and reason; nothing is silently loosened.

## Scope
- **Securities:** S&P 500 members by ISIN (`security_id`), XNYS sessions 2011-01-03 → 2026-11-18.
- **Source data:** the LSEG snapshot `TrainingData/indicators_data/lseg/20261119T132813190380Z` and its sibling runs, the hand-verified override CSVs in `TrainingData/pit/overrides/`, and SEC filings.
- **Reserved confirmation period:** 2025-01-02 → 2026-11-18.
  - The data layer may *build* these sessions so the acceptance tests can cover them.
  - No feature, model, threshold or strategy choice may *read* them. Loaders enforce this with a guard that raises unless `confirm=True`.

## Acceptance tests (all must pass before any model or strategy work)

| ID | Area | Test | Pass condition |
|---|---|---|---|
| M1 | Membership | Replay vs the 4 LSEG checkpoints (2015-12-31, 2019-12-31, 2023-12-29, 2026-11-18) | 0 extra, 0 missing at every checkpoint |
| M2 | Membership | Replay vs the dated boundary probes in `boundary_probes/` | 0 unexplained differences; each explained difference cites a source in `overrides/` |
| M3 | Membership | Opening roster 2011-01-03 vs SPY's SEC holdings as of 2010-12-31 | Exact match after documented, sourced corrections |
| M4 | Membership | Internal consistency | No forward or reverse event conflicts; 495–510 members per session |
| M5 | Membership | Regression | The new pipeline reproduces `membership_dated_candidate.csv.gz` exactly, or every row difference is a documented correction |
| I1 | Identity | `security_master` | No security has two primary RICs on one session; no RIC maps to two securities on one session |
| P1 | Prices | Executable bar coverage | ≥ 99.9% of member-sessions have a valid bar (L>0, H≥max(O,C), L≤min(O,C), V>0) from the security's primary RIC for that session |
| P2 | Prices | Alias adjudication (BKR, CPB, HST, KMB) | For each pair, the chosen primary series' volume is within ±10% of an independent consolidated feed on ≥ 95% of sampled sessions; evidence is stored |
| P3 | Prices | Exceptions | Every member-session without a valid bar is in the exception ledger with a reason category and source |
| C1 | Splits | All capital-change split/consolidation events for members | Each has a trading-effective date confirmed by the price ratio (±10%), or a documented manual check |
| C2 | Splits | Continuity | No unexplained |overnight move| > 50% in split-adjusted open/close for member-sessions |
| C3 | Dividends | Ex-date cash dividends | Duplicates removed; non-USD or unknown-currency rows flagged; SPY distributions present for the TR benchmark |
| S1 | Settlement | Every member that leaves the panel without remaining listed | Exactly one terminal outcome (cash / stock / mixed / elective / contingent / OTC / liquidation) with a source URL; 0 `unknown` |
| S2 | Settlement | Validation | Cash deals: the settlement value is within ±2% of the last close. Stock and mixed deals: within ±5% of the last close, using the successor's first available open. Outliers are documented |
| K1 | Costs | Spread estimate coverage | ≥ 95% of member-sessions have a Corwin–Schultz or Abdi–Ranaldo 21-day spread estimate |
| K2 | Costs | Estimator sanity | Where LSEG closing bid/ask exist, the median absolute error of the estimate is reported. It is not a pass/fail threshold, because there's no prior basis for one |
| R1 | Reproducibility | Rebuild from scratch | Identical sha256 for every `pit/` artifact on a second run |

## Verdict rule
- **Data layer ACCEPTED** only if every test except K2 passes.
- A failure blocks Phase 7, the momentum and liquidity research round. The failure is fixed, or the affected securities and sessions are excluded with a documented reason.
- An exclusion of more than 0.5% of member-sessions counts as a failure.

## Amendment log
- **2026-11-20 — P2 (alias adjudication).** The preregistered volume criterion assumed LSEG `ACVOL_UNS` is consolidated volume. Measured against Alpha Vantage, it is about 25–38% of consolidated volume for *every* stock checked, including the controls AAPL, MSFT, JPM and XOM. It is venue volume.
  - **P2 as written therefore FAILS** for all series. It is kept as a failure, not loosened.
  - **New test P2a**, added after seeing data and labelled post hoc: the chosen primary series' close equals the consolidated close (within 1 bp) on ≥ 95% of sessions.
  - **Consequence:** LSEG volume cannot be used as consolidated volume for ADV, capacity or liquidity features until a consolidated-volume source or a validated scaling exists. This blocks K1's ADV inputs and the liquidity features of Phase 7.
