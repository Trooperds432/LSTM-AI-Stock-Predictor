# LSEG acquisition status

The desktop connection was verified on 19 November 2026 using the existing
Windows user-level `LSEG_APP_KEY`. The key inherited by the Codex process was
stale and failed authentication. No credential value is stored in this project.

Run with Workspace signed in, using Python with `lseg-data` installed:

```powershell
python TrainingData/lseg_download.py --start 2011-01-01 --end 2026-11-18
```

Explicit RICs can be supplied with `--rics`. Each run creates a new directory
under `TrainingData/indicators_data/lseg/`. Its manifest records requests,
missing values, row counts, dates, failures, SDK version and CSV SHA-256 hashes.
Raw observations retain missing values. Existing model inputs are not changed.

## Verified downloads

- `20261119T130902389810Z`: AAPL.O, MSFT.O and SPY.P unadjusted daily OHLCV,
  each 3,994 rows; dollar-index history, 4,127 rows; current reference snapshot;
  quarterly revenue, net income and total assets, each 128 rows across Apple
  and Microsoft; S&P 500 membership query dated 1 January 2011, 503 rows.
- `20261119T130956918725Z`: two-year Treasury BID yield history, 3,973 rows,
  and ten-year Treasury BID yield history, 3,972 rows. The initial request used
  an unsupported trade-price field; BID succeeded.
- Daily histories end on 18 November 2026. Their start is 3 January 2011.
- VIX and S&P 500 index prices returned explicit permission errors. SPY data
  is available, but is not an equivalent substitute for the missing VIX series.

## Required before training

This is a verified acquisition pilot, not a complete research dataset.

1. Resolve and validate permanent security identities and historical ticker/RIC
   changes for the intended universe. `TR.PermID` was requested but did not
   appear in the reference response; ISIN/RIC alone do not complete this step.
2. Retrieve historical membership across the research period. One constituent
   snapshot does not establish a survivorship-free universe; verify the API's
   effective-date interpretation, particularly holidays.
3. Acquire and reconcile corporate actions, adjusted/total returns and delisting
   outcomes against the saved unadjusted prices.
4. Obtain as-reported fundamentals and availability timestamps. The quarterly
   fiscal dates returned here are not proof of public availability; do not join
   them directly to fiscal period ends for predictive training.
5. Acquire historical spreads/liquidity for execution costs, exchange sessions,
   and publication timestamps for insider/political disclosures and news.
6. Resolve VIX entitlements and add credit spreads and release-vintage macro
   observations. Preserve the distinction between quotes and executed prices.
7. Expand beyond the pilot securities only after universe/identity validation.

No performance confirmation, model training or parameter selection was run.
Newly acquired later history must remain unexamined for performance until a
recipe, trials, costs, baselines and confirmation criteria are preregistered.
