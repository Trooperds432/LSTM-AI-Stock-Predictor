"""Preregistered audit of the LSTM probability strategy.

Criteria are fixed in research/audit_2026-11-19_preregistration.md (written before this ran).
Usage:  python audit_backtest.py [--shuffles 500] [--boot 5000]
Outputs: research/audit_2026-11-19_results.json and printed summary.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import backtest_core as bc

ROOT = Path(__file__).resolve().parent
SEED = 20261119
SPLIT_DATE = pd.Timestamp("2022-06-10")  # subperiod boundary, preregistered
N = NormalDist()


def spy_equity(dates: list[pd.Timestamp], v0: float) -> np.ndarray:
    spy = pd.read_csv(ROOT / "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv",
                      parse_dates=["date"]).set_index("date")["close"].sort_index()
    s = spy.reindex(pd.DatetimeIndex(dates)).ffill().bfill().to_numpy(dtype=float)
    return v0 * s / s[0]


def block_bootstrap(r: np.ndarray, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: returns array [n_boot, len(r)] of resampled return paths."""
    n = r.size
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return r[idx]


def deflated_sharpe(sr_daily: float, sr_trials_daily: np.ndarray, n_obs: int, skew: float, kurt: float) -> dict:
    """Bailey & Lopez de Prado (2014). Sharpe values are per-period (daily), kurt is non-excess."""
    n_trials = len(sr_trials_daily)
    var_sr = float(np.var(sr_trials_daily, ddof=1))
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * N.inv_cdf(1 - 1 / n_trials) + gamma * N.inv_cdf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2)
    z = (sr_daily - sr0) * np.sqrt(n_obs - 1) / denom
    psr0 = (sr_daily - 0) * np.sqrt(n_obs - 1) / denom
    return {"n_trials": n_trials, "var_trial_sr_daily": var_sr, "sr0_daily": float(sr0),
            "sr0_annualized": float(sr0 * np.sqrt(252)), "dsr": float(N.cdf(z)),
            "psr_vs_zero": float(N.cdf(psr0))}


def segment(equity: np.ndarray, dates: list[pd.Timestamp], lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> dict:
    d = pd.DatetimeIndex(dates)
    m = np.ones(len(d), bool)
    if lo is not None:
        m &= d > lo
    if hi is not None:
        m &= d <= hi
    idx = np.flatnonzero(m)
    # include the prior day as the base so returns chain exactly
    start = max(idx[0] - 1, 0)
    e = equity[start: idx[-1] + 1]
    return {"total_return": float(e[-1] / e[0] - 1), "sharpe": bc.sharpe(bc.daily_returns(e)),
            "max_drawdown": bc.max_drawdown(e), "from": str(d[idx[0]].date()), "to": str(d[idx[-1]].date())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=500)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    split = json.loads((ROOT / "forecasts/split_info.json").read_text())
    forecasts, excluded = bc.load_forecasts(ROOT / "forecasts")
    panel = bc.build_panel(forecasts, start=split["oos_start"])
    main_p = bc.StrategyParams()  # frozen defaults
    v0 = main_p.initial_value
    spy = spy_equity(panel.dates, v0)
    spy_ret = float(spy[-1] / spy[0] - 1)

    out: dict = {"seed": SEED, "params_main": asdict(main_p), "universe": len(panel.tickers),
                 "excluded_non_common": excluded, "window": [str(panel.dates[0].date()), str(panel.dates[-1].date())],
                 "spy": {"total_return": spy_ret, "sharpe": bc.sharpe(bc.daily_returns(spy)),
                         "max_drawdown": bc.max_drawdown(spy), "note": "price only, no dividends"}}

    # ---- Attribution: how much did each bias contribute?
    variants = {
        "A_old_sameclose_fullfilter": replace(main_p, next_bar_fill=False, spike_mode="full"),
        "B_nextbar_fullfilter": replace(main_p, next_bar_fill=True, spike_mode="full"),
        "C_sameclose_causalfilter": replace(main_p, next_bar_fill=False, spike_mode="causal"),
        "MAIN_nextbar_causal": main_p,
        "cost40": replace(main_p, cost_bps=40.0),
        "cost60": replace(main_p, cost_bps=60.0),
    }
    runs: dict[str, bc.Result] = {}
    out["runs"] = {}
    for name, p in variants.items():
        res = bc.run_strategy(panel, p)
        runs[name] = res
        out["runs"][name] = bc.summarize(res, v0)
    main_res = runs["MAIN_nextbar_causal"]
    main_sum = out["runs"]["MAIN_nextbar_causal"]
    main_res.trades.to_csv(ROOT / "research/audit_2026-11-19_main_trades.csv", index=False)
    r = bc.daily_returns(main_res.equity)
    r_spy = bc.daily_returns(spy)

    tests: dict = {}
    # ---- Test 1: beats SPY
    tests["1_main_vs_spy"] = {"strategy": main_sum["total_return"], "spy": spy_ret,
                              "pass": main_sum["total_return"] > spy_ret}

    # ---- Test 2: shuffled-signal negative control
    null_rets = np.empty(args.shuffles)
    null_sharpes = np.empty(args.shuffles)
    for k in range(args.shuffles):
        res = bc.run_strategy(panel, main_p, rng=rng, shuffle_signals=True)
        null_rets[k] = res.equity[-1] / v0 - 1
        null_sharpes[k] = bc.sharpe(bc.daily_returns(res.equity))
    p_shuffle = float((1 + np.sum(null_rets >= main_sum["total_return"])) / (1 + args.shuffles))
    tests["2_shuffle_control"] = {
        "n": args.shuffles, "p_value": p_shuffle,
        "null_return_mean": float(null_rets.mean()), "null_return_p5_p50_p95": np.percentile(null_rets, [5, 50, 95]).tolist(),
        "null_sharpe_p50_p95": np.nanpercentile(null_sharpes, [50, 95]).tolist(),
        "pass": p_shuffle <= 0.05}

    # ---- Test 3: block bootstrap CI of Sharpe (and of excess over SPY)
    boots = block_bootstrap(r, args.boot, args.block, rng)
    boot_sr = np.sqrt(252) * boots.mean(1) / boots.std(1, ddof=1)
    excess = r - r_spy[: r.size]
    boots_x = block_bootstrap(excess, args.boot, args.block, rng)
    boot_ir = np.sqrt(252) * boots_x.mean(1) / boots_x.std(1, ddof=1)
    boot_tot = np.prod(1 + boots, axis=1) - 1
    ci_sr = np.percentile(boot_sr, [2.5, 97.5]).tolist()
    tests["3_bootstrap"] = {"block": args.block, "n": args.boot, "sharpe": main_sum["sharpe"], "sharpe_ci95": ci_sr,
                            "info_ratio_vs_spy": bc.sharpe(excess), "info_ratio_ci95": np.percentile(boot_ir, [2.5, 97.5]).tolist(),
                            "total_return_ci95": np.percentile(boot_tot, [2.5, 97.5]).tolist(),
                            "pass": ci_sr[0] > 0}

    # ---- Test 4: Deflated Sharpe with every trial run so far
    trial_sr = []
    for f in glob.glob(str(ROOT / "research/trials_2026-11-19/*/*.json")):
        s = json.load(open(f))["strategy"].get("sharpe")
        if s is not None:
            trial_sr.append(s)
    trial_sr += [v["sharpe"] for v in out["runs"].values() if np.isfinite(v["sharpe"])]
    sr_d = float(r.mean() / r.std(ddof=1))
    rs = pd.Series(r)
    dsr = deflated_sharpe(sr_d, np.array(trial_sr) / np.sqrt(252), r.size, float(rs.skew()), float(rs.kurt() + 3))
    tests["4_deflated_sharpe"] = {**dsr, "pass": dsr["dsr"] >= 0.95}

    # ---- Test 5: remove top-3 trades by dollar P&L
    pnl = main_res.trades["PnL$"].sort_values(ascending=False)
    ret_wo = float((main_res.equity[-1] - pnl.head(3).sum()) / v0 - 1)
    tests["5_remove_top3"] = {"top3_pnl": pnl.head(3).round(2).tolist(), "top3_share_of_net_pnl": float(pnl.head(3).sum() / pnl.sum()) if pnl.sum() else None,
                              "return_without_top3": ret_wo, "spy": spy_ret, "pass": ret_wo > spy_ret}

    # ---- Test 6: cost stress
    tests["6_cost_stress"] = {"cost40_return": out["runs"]["cost40"]["total_return"],
                              "cost60_return": out["runs"]["cost60"]["total_return"], "spy": spy_ret,
                              "pass": out["runs"]["cost40"]["total_return"] > spy_ret}

    # ---- Test 7: subperiods
    sub = {}
    for label, lo, hi in [("P1", None, SPLIT_DATE), ("P2", SPLIT_DATE, None)]:
        sub[label] = {"strategy": segment(main_res.equity, panel.dates, lo, hi), "spy": segment(spy, panel.dates, lo, hi)}
    beats = [sub[k]["strategy"]["total_return"] > sub[k]["spy"]["total_return"] for k in sub]
    tests["7_subperiods"] = {**sub, "beats_spy": beats, "pass": any(beats)}

    out["tests"] = tests
    core = all(tests[k]["pass"] for k in ["1_main_vs_spy", "2_shuffle_control", "3_bootstrap", "4_deflated_sharpe"])
    out["verdict"] = "edge SUPPORTED" if core else "edge NOT SUPPORTED on current evidence"
    out["warnings"] = [k for k in ["5_remove_top3", "6_cost_stress", "7_subperiods"] if not tests[k]["pass"]]
    out["n_trials_total"] = dsr["n_trials"] + args.shuffles * 0  # shuffles are controls, not trials

    path = ROOT / "research/audit_2026-11-19_results.json"
    path.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"Universe {out['universe']} tickers, window {out['window']}, SPY {spy_ret:+.1%}")
    for k, v in out["runs"].items():
        print(f"  {k:28s} ret {v['total_return']:+7.1%}  sharpe {v['sharpe']:5.2f}  mdd {v['max_drawdown']:+6.1%}  trades {v['trades']:3d}")
    for k, v in tests.items():
        print(f"  TEST {k:22s} {'PASS' if v['pass'] else 'FAIL'}")
    print("VERDICT:", out["verdict"], "| warnings:", out["warnings"])
    print("Wrote", path)


if __name__ == "__main__":
    main()
