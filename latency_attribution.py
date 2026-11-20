"""Execution-latency attribution (research/latency_2026-11-20_preregistration.md).

Where does the same-close (A) vs next-close (C) gap come from, and how much of it happens overnight?
  1a  portfolio runs A = same_close, B = next_open, C = next_close (frozen StrategyParams otherwise)
  1b  decomposition of A's trades into overnight / next session / rest, replayed with B and C entries
  1c  signal-level event study of every qualifying signal (the gate statistic, G1)

Usage:  python latency_attribution.py [--boot 2000]
Writes research/latency_2026-11-20_results.json and research/latency_2026-11-20_*.csv.gz.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as bc

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "TrainingData" / "indicators_data" / "raw" / "stocksData"
OUT = ROOT / "research"
TAG = "latency_2026-11-20"
SEED = 20261120
SPLIT_FLAG = 0.4          # |overnight| or |next-session| log move treated as a split / bad print
CLOSE_TOL = 0.005         # raw close must match the forecast Close within 0.5% for its Open to be used
G1_MIN_GAP = 0.0020
G1_MIN_T = 2.0
G1_MIN_YEARS = 5
FILLS = {"A_same_close": "same_close", "B_next_open": "next_open", "C_next_close": "next_close"}


# ----------------------------------------------------------------------------- data

def raw_name(ticker: str) -> str:
    """'AAL_daily_processed' -> 'AAL_daily.csv'."""
    base = ticker[: -len("_daily_processed")] if ticker.endswith("_daily_processed") else ticker
    return f"{base}_daily.csv"


def attach_open(forecasts: dict[str, pd.DataFrame], raw_dir: Path = RAW_DIR) -> tuple[dict[str, pd.DataFrame], dict]:
    """Add an Open column from the raw daily files. Open is NaN where the raw close disagrees with the
    forecast Close by more than CLOSE_TOL (different bar), or where no raw row exists."""
    out, missing_files, rows, mismatched, no_raw = {}, [], 0, 0, 0
    for t, df in forecasts.items():
        df = df.copy()
        path = Path(raw_dir) / raw_name(t)
        if not path.exists():
            missing_files.append(t)
            df["Open"] = np.nan
            out[t] = df
            continue
        raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        raw = raw[~raw.index.duplicated(keep="last")].reindex(df.index)
        ok_close = (raw["close"] / df["Close"] - 1.0).abs() <= CLOSE_TOL
        rows += len(df)
        no_raw += int(raw["close"].isna().sum())
        mismatched += int((raw["close"].notna() & ~ok_close).sum())
        df["Open"] = raw["open"].where(ok_close & (raw["open"] > 0))
        out[t] = df
    return out, {"tickers": len(forecasts), "missing_raw_files": missing_files, "rows": rows,
                 "rows_without_raw_bar": no_raw, "rows_close_mismatch": mismatched}


# ----------------------------------------------------------------------------- signals

def candidate_signals(panel: bc.Panel, p: bc.StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """Every (t, j) the engine would accept as a signal if it had a free slot: eligible on day t and
    best adjusted probability > min_accepted. Returns (mask [T,N], best horizon in sessions [T,N])."""
    T, N = panel.close.shape
    cand = np.zeros((T, N), bool)
    days = np.zeros((T, N), int)
    for i in range(T - 1):                       # no signals on the final day (engine rule)
        elig = bc._eligible_mask(panel, i, p)
        best_adj = np.full(N, -np.inf)
        best_d = np.zeros(N, int)
        for h, d in bc.PERIODS.items():
            s = panel.std[h][i]
            s = np.where(np.isfinite(s), s, 0.0)
            adj = panel.prob[h][i] - p.std_factor * np.maximum(s, 1e-6)
            better = np.isfinite(adj) & (adj > best_adj)
            best_adj[better] = adj[better]
            best_d[better] = d
        cand[i] = elig & (best_adj > p.min_accepted)
        days[i] = best_d
    return cand, days


def component_arrays(panel: bc.Panel) -> dict[str, np.ndarray]:
    """Log-return components dated at the signal day t (row t uses bars t and t+1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(panel.close)
        lo = np.log(panel.open) if panel.open is not None else np.full_like(lc, np.nan)
    nxt = lambda a: np.vstack([a[1:], np.full((1, a.shape[1]), np.nan)])  # noqa: E731
    return {"gap": nxt(lc) - lc, "on": nxt(lo) - lc, "s1": nxt(lc) - nxt(lo), "d0": lc - lo, "lc": lc}


def event_table(panel: bc.Panel, p: bc.StrategyParams, exclude_flagged: bool = True) -> pd.DataFrame:
    """One row per candidate signal with raw and excess components. Excess = component minus the
    equal-weight mean of the same component over the day's universe (eligible, finite close t and t+1)."""
    T, N = panel.close.shape
    cand, days = candidate_signals(panel, p)
    comp = component_arrays(panel)
    lc = comp.pop("lc")
    flagged = (np.abs(np.nan_to_num(comp["on"])) > SPLIT_FLAG) | (np.abs(np.nan_to_num(comp["s1"])) > SPLIT_FLAG)
    rows = []
    horizons = sorted(set(bc.PERIODS.values()))
    post = {}
    for d in horizons:
        a = np.full((T, N), np.nan)
        if T > d + 1:
            a[: T - d - 1] = lc[d + 1:] - lc[1: T - d]
        post[d] = a
    for i in range(T - 1):
        elig = bc._eligible_mask(panel, i, p)
        uni = elig & np.isfinite(comp["gap"][i])
        if exclude_flagged:
            uni &= ~flagged[i]
        ev = np.flatnonzero(cand[i] & uni)
        if ev.size == 0:
            continue
        means = {k: (np.nanmean(v[i][uni]) if np.isfinite(v[i][uni]).any() else np.nan) for k, v in comp.items()}
        pmeans = {d: (np.nanmean(post[d][i][uni]) if np.isfinite(post[d][i][uni]).any() else np.nan) for d in horizons}
        prev = lc[i] - lc[i - 1] if i > 0 else np.full(N, np.nan)
        for j in ev:
            r = {"Date": panel.dates[i], "Ticker": panel.tickers[j], "Days": int(days[i, j]),
                 "Flagged": bool(flagged[i, j]), "UniverseN": int(uni.sum()), "PrevRet": float(prev[j])}
            for k, v in comp.items():
                r[k] = float(v[i, j])
                r[f"x_{k}"] = float(v[i, j] - means[k])
            r["post"] = float(post[days[i, j]][i, j])
            r["x_post"] = float(post[days[i, j]][i, j] - pmeans[days[i, j]])
            rows.append(r)
    return pd.DataFrame(rows)


def daily_mean_stat(events: pd.DataFrame, col: str, lags: int = 10) -> dict:
    """Average `col` across events per date, then mean and Newey-West t of that daily series."""
    e = events.dropna(subset=[col])
    if e.empty:
        return {"n_events": 0, "n_days": 0, "mean": float("nan"), "nw_t": float("nan"), "event_mean": float("nan")}
    daily = e.groupby("Date")[col].mean().sort_index()
    return {"n_events": int(len(e)), "n_days": int(len(daily)), "mean": float(daily.mean()),
            "nw_t": bc.newey_west_t(daily.to_numpy(), lags=lags), "event_mean": float(e[col].mean())}


def by_year(events: pd.DataFrame, col: str) -> dict[int, float]:
    e = events.dropna(subset=[col])
    daily = e.groupby("Date")[col].mean()
    return {int(y): float(v) for y, v in daily.groupby(daily.index.year).mean().items()}


def gate_g1(stat: dict, years: dict[int, float]) -> dict:
    pos = sum(v > 0 for v in years.values())
    checks = {"mean_ge_20bps": stat["mean"] >= G1_MIN_GAP, "nw_t_ge_2": stat["nw_t"] >= G1_MIN_T,
              f"positive_years_ge_{G1_MIN_YEARS}": pos >= G1_MIN_YEARS}
    return {"checks": checks, "positive_years": pos, "n_years": len(years), "pass": all(checks.values())}


# ----------------------------------------------------------------------------- trades

def decompose_trades(trades: pd.DataFrame, panel: bc.Panel) -> pd.DataFrame:
    """Split each same-close trade (bought at close t) into on / s1 / rest gross log returns."""
    di = {d: k for k, d in enumerate(panel.dates)}
    tj = {t: k for k, t in enumerate(panel.tickers)}
    comp = component_arrays(panel)
    rows = []
    for tr in trades.itertuples(index=False):
        i, j = di[tr.SignalDate], tj[tr.Ticker]
        c_next = panel.close[i + 1, j] if i + 1 < len(panel.dates) else np.nan
        total = float(np.log(tr.ExitPrice / tr.EntryPrice))
        rest = float(np.log(tr.ExitPrice / c_next)) if np.isfinite(c_next) else np.nan
        rows.append({"Ticker": tr.Ticker, "SignalDate": tr.SignalDate, "SellDate": tr.SellDate, "Horizon": tr.Horizon,
                     "total": total, "on": float(comp["on"][i, j]), "s1": float(comp["s1"][i, j]),
                     "gap": float(comp["gap"][i, j]), "rest": rest})
    return pd.DataFrame(rows)


def match_trades(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ka = set(zip(a["Ticker"], a["SignalDate"])) if len(a) else set()
    kb = set(zip(b["Ticker"], b["SignalDate"])) if len(b) else set()
    return {"matched": len(ka & kb), "only_first": len(ka - kb), "only_second": len(kb - ka)}


def gross_log_sum(trades: pd.DataFrame) -> float:
    return float(np.log(trades["ExitPrice"] / trades["EntryPrice"]).sum()) if len(trades) else 0.0


# ----------------------------------------------------------------------------- driver

def bootstrap_sharpe_ci(equity: np.ndarray, n_boot: int, rng: np.random.Generator, block: int = 20) -> list[float]:
    from audit_backtest import block_bootstrap
    r = bc.daily_returns(equity)
    paths = block_bootstrap(r, n_boot, block, rng)
    sh = np.sqrt(252) * paths.mean(1) / paths.std(1, ddof=1)
    return np.nanpercentile(sh, [2.5, 97.5]).tolist()


def analyse(name: str, forecasts: dict[str, pd.DataFrame], start, n_boot: int, rng) -> dict:
    fc, open_report = attach_open(forecasts)
    panel = bc.build_panel(fc, start=start)
    p0 = bc.StrategyParams()
    v0 = p0.initial_value
    out: dict = {"window": [str(panel.dates[0].date()), str(panel.dates[-1].date())],
                 "universe": len(panel.tickers), "open_join": {**open_report,
                                                              "missing_raw_files": len(open_report["missing_raw_files"])}}
    runs = {}
    out["portfolio"] = {}
    for label, fill in FILLS.items():
        res = bc.run_strategy(panel, replace(p0, fill=fill))
        runs[label] = res
        s = bc.summarize(res, v0)
        s["sharpe_ci95"] = bootstrap_sharpe_ci(res.equity, n_boot, rng)
        s["gross_log_sum"] = gross_log_sum(res.trades)
        out["portfolio"][label] = s
        res.trades.to_csv(OUT / f"{TAG}_{name}_{label}_trades.csv.gz", index=False)

    a_tr = runs["A_same_close"].trades
    dec = decompose_trades(a_tr, panel)
    dec.to_csv(OUT / f"{TAG}_{name}_A_decomposition.csv.gz", index=False)
    sums = {k: float(dec[k].sum()) for k in ["total", "on", "s1", "rest"]}
    out["decomposition_A_trades"] = {
        "n": int(len(dec)), "sum_log": sums, "mean_log": {k: float(dec[k].mean()) for k in ["total", "on", "s1", "rest"]},
        "replay_C_entry_sum": sums["rest"], "replay_B_entry_sum": float((dec["s1"] + dec["rest"]).sum()),
        "selection_effect_C": out["portfolio"]["C_next_close"]["gross_log_sum"] - sums["rest"],
        "selection_effect_B": out["portfolio"]["B_next_open"]["gross_log_sum"] - float((dec["s1"] + dec["rest"]).sum()),
        "matching": {"A_vs_C": match_trades(a_tr, runs["C_next_close"].trades),
                     "A_vs_B": match_trades(a_tr, runs["B_next_open"].trades)},
    }

    ev = event_table(panel, p0, exclude_flagged=True)
    ev_all = event_table(panel, p0, exclude_flagged=False)
    ev.to_csv(OUT / f"{TAG}_{name}_events.csv.gz", index=False)
    stats = {c: daily_mean_stat(ev, c) for c in ["x_gap", "x_on", "x_s1", "x_d0", "x_post", "gap", "on", "s1"]}
    out["events"] = {
        "primary_excluding_flagged": stats,
        "including_flagged": {c: daily_mean_stat(ev_all, c) for c in ["x_gap", "x_on", "x_s1"]},
        "flagged_events": int(ev_all["Flagged"].sum()),
        "by_year_x_gap": by_year(ev, "x_gap"),
        "by_year_x_on": by_year(ev, "x_on"),
        "by_prev_day_sign": {sgn: {c: daily_mean_stat(ev[cond], c) for c in ["x_gap", "x_on", "x_s1"]}
                             for sgn, cond in [("prev_down", ev["PrevRet"] < 0), ("prev_up", ev["PrevRet"] >= 0)]},
    }
    out["events"]["g1"] = gate_g1(stats["x_gap"], out["events"]["by_year_x_gap"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    from wf_evaluate import ensemble, load_split, to_dict

    split = json.loads((ROOT / "forecasts/split_info.json").read_text())
    legacy, _ = bc.load_forecasts(ROOT / "forecasts")
    wf = to_dict(ensemble(load_split(ROOT / "wf", "test")))

    results = {"seed": SEED, "preregistration": f"research/{TAG}_preregistration.md",
               "trials_after_this_study": 31,
               "legacy": analyse("legacy", legacy, split["oos_start"], args.boot, rng),
               "walkforward": analyse("wf", wf, None, args.boot, rng)}
    results["G1_gate_walkforward"] = results["walkforward"]["events"]["g1"]
    (OUT / f"{TAG}_results.json").write_text(json.dumps(results, indent=2, default=str))
    for k in ["legacy", "walkforward"]:
        print(f"== {k} {results[k]['window']}")
        for lbl, s in results[k]["portfolio"].items():
            print(f"  {lbl:14s} ret {s['total_return']:+.1%} cagr {s['cagr']:+.1%} sharpe {s['sharpe']:.2f} "
                  f"trades {s['trades']}")
        st = results[k]["events"]["primary_excluding_flagged"]
        for c in ["x_gap", "x_on", "x_s1", "x_d0", "x_post"]:
            print(f"  {c:7s} mean {st[c]['mean']*1e4:+7.1f} bp  NW t {st[c]['nw_t']:+.2f}  events {st[c]['n_events']}")
    print("G1:", results["G1_gate_walkforward"])


if __name__ == "__main__":
    main()
