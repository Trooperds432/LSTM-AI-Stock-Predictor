"""Evaluate walk-forward forecasts against the preregistered tests
(research/walkforward_2026-11-19_preregistration.md).

Usage: python wf_evaluate.py [--dir wf] [--shuffles 500] [--boot 5000]
Writes research/walkforward_2026-11-19_results.json (or *_smoke.json for --dir wf_smoke).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as bc
from audit_backtest import block_bootstrap, deflated_sharpe, segment, spy_equity

ROOT = Path(__file__).resolve().parent
SEED = 20261120
HZ = list(bc.PERIODS)
GRID = [0.20, 0.25, 0.30, 0.35]


def load_split(d: Path, split: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(d / "forecasts" / f"Y*_s*_{split}.csv.gz"))):
        m = re.search(r"Y(\d+)_s(\d+)_", Path(f).name)
        df = pd.read_csv(f, parse_dates=["Date"])
        df["Year"], df["Seed"] = int(m.group(1)), int(m.group(2))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """p = mean of seed means; sigma = sqrt(mean within-seed var + var across seed means)."""
    df = df.copy()
    for h in HZ:
        df[f"_var_{h}"] = df[f"Pred_Prob_Std_{h}"] ** 2
    g = df.groupby(["Year", "Date", "Ticker"], sort=False)
    out = g[["Close"]].first()
    for h in HZ:
        out[f"Pred_Prob_{h}"] = g[f"Pred_Prob_{h}"].mean()
        within = g[f"_var_{h}"].mean()
        across = g[f"Pred_Prob_{h}"].var(ddof=0).fillna(0.0)
        out[f"Pred_Prob_Std_{h}"] = np.sqrt(within + across)
    return out.reset_index()


def to_dict(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["Close"] + [f"Pred_Prob_{h}" for h in HZ] + [f"Pred_Prob_Std_{h}" for h in HZ]
    return {t: g.set_index("Date").sort_index()[cols] for t, g in long.groupby("Ticker")}


def reversal_forecasts(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """score_1w = cross-sectional pct rank of -(5-day return); other horizons blank; sigma 0."""
    wide = prices.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    ret5 = pd.DataFrame({t: wide[t].dropna().pct_change(5) for t in wide.columns}).reindex(wide.index)
    score = (-ret5).rank(axis=1, pct=True)
    out = {}
    for t in wide.columns:
        df = pd.DataFrame({"Close": wide[t]}).dropna()
        for h in HZ:
            df[f"Pred_Prob_{h}"] = score[t].reindex(df.index) if h == "1w" else np.nan
            df[f"Pred_Prob_Std_{h}"] = 0.0
        out[t] = df
    return out


def ew_universe(panel: bc.Panel, v0: float) -> np.ndarray:
    c = panel.close
    r = c[1:] / c[:-1] - 1.0
    ok = np.isfinite(r) & (np.nan_to_num(c[:-1]) >= 5.0)
    daily = np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
    return v0 * np.concatenate([[1.0], np.cumprod(1 + daily)])


def run_summary(panel, p, v0):
    res = bc.run_strategy(panel, p)
    return res, bc.summarize(res, v0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="wf")
    ap.add_argument("--shuffles", type=int, default=500)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    d = ROOT / a.dir
    rng = np.random.default_rng(SEED)
    p0 = bc.StrategyParams()
    v0 = p0.initial_value

    test = load_split(d, "test")
    val = load_split(d, "val")
    ens = ensemble(test)
    years = sorted(ens["Year"].unique())
    panel = bc.build_panel(to_dict(ens))
    spy = spy_equity(panel.dates, v0)
    spy_sum = {"total_return": float(spy[-1] / v0 - 1), "sharpe": bc.sharpe(bc.daily_returns(spy)),
               "max_drawdown": bc.max_drawdown(spy),
               "cagr": float((spy[-1] / v0) ** (252 / (len(spy) - 1)) - 1)}

    out: dict = {"seed": SEED, "dir": a.dir, "years": years, "window": [str(panel.dates[0].date()), str(panel.dates[-1].date())],
                 "universe": len(panel.tickers), "params": asdict(p0), "spy": spy_sum}

    # Fold diagnostics
    metas = [json.load(open(f)) for f in sorted(glob.glob(str(d / "meta" / "*.json")))]
    out["folds"] = [{k: m[k] for k in ["year", "seed_index", "n_train", "n_val", "tickers_trained", "best_epoch",
                                        "best_val_auc", "test_auc", "minutes"]} for m in metas]

    # ---- Primary
    main_res, main_sum = run_summary(panel, p0, v0)
    main_res.trades.to_csv(ROOT / f"research/walkforward_2026-11-19_trades{'_smoke' if 'smoke' in a.dir else ''}.csv", index=False)
    r = bc.daily_returns(main_res.equity)
    out["primary"] = main_sum

    # ---- Comparators
    ew = ew_universe(panel, v0)
    rev_panel = bc.build_panel(reversal_forecasts(ens[["Date", "Ticker", "Close"]]), start=panel.dates[0])
    rev_res, rev_sum = run_summary(rev_panel, replace(p0, min_accepted=0.0, std_factor=0.0), v0)
    cost40_res, cost40_sum = run_summary(panel, replace(p0, cost_bps=40.0), v0)
    out["comparators"] = {
        "ew_universe": {"total_return": float(ew[-1] / v0 - 1), "sharpe": bc.sharpe(bc.daily_returns(ew)),
                        "max_drawdown": bc.max_drawdown(ew), "cagr": float((ew[-1] / v0) ** (252 / (len(ew) - 1)) - 1)},
        "reversal_baseline": rev_sum, "cost40": cost40_sum}

    tests: dict = {}
    tests["1_cagr_vs_spy"] = {"strategy": main_sum["cagr"], "spy": spy_sum["cagr"], "pass": main_sum["cagr"] > spy_sum["cagr"]}

    null = np.empty(a.shuffles)
    for k in range(a.shuffles):
        null[k] = bc.run_strategy(panel, p0, rng=rng, shuffle_signals=True).equity[-1] / v0 - 1
    p_sh = float((1 + np.sum(null >= main_sum["total_return"])) / (1 + a.shuffles))
    tests["2_shuffle_control"] = {"n": a.shuffles, "p_value": p_sh, "null_p5_p50_p95": np.percentile(null, [5, 50, 95]).tolist(),
                                  "pass": p_sh <= 0.05}

    boots = block_bootstrap(r, a.boot, 20, rng)
    bsr = np.sqrt(252) * boots.mean(1) / boots.std(1, ddof=1)
    ci = np.percentile(bsr, [2.5, 97.5]).tolist()
    exc = r - bc.daily_returns(spy)[: r.size]
    bx = block_bootstrap(exc, a.boot, 20, rng)
    tests["3_bootstrap"] = {"sharpe": main_sum["sharpe"], "sharpe_ci95": ci, "info_ratio": bc.sharpe(exc),
                            "info_ratio_ci95": np.percentile(np.sqrt(252) * bx.mean(1) / bx.std(1, ddof=1), [2.5, 97.5]).tolist(),
                            "pass": ci[0] > 0}

    # ---- Secondary: per-fold threshold chosen on validation forecasts, applied via an equivalent prob shift
    val_ens = ensemble(val)
    chosen = {}
    for y in years:
        vp = bc.build_panel(to_dict(val_ens[val_ens["Year"] == y]))
        scores = {}
        for g in GRID:
            s = bc.summarize(bc.run_strategy(vp, replace(p0, min_accepted=g)), v0)["sharpe"]
            scores[g] = s if np.isfinite(s) else -np.inf
        chosen[int(y)] = max(scores, key=scores.get)
    shifted = ens.copy()
    for y, g in chosen.items():
        m = shifted["Year"] == y
        for h in HZ:
            shifted.loc[m, f"Pred_Prob_{h}"] += (p0.min_accepted - g)  # adj > g  <=>  (p + 0.25 - g) - sigma > 0.25
    sec_res, sec_sum = run_summary(bc.build_panel(to_dict(shifted)), p0, v0)
    rs = bc.daily_returns(sec_res.equity)
    bs = block_bootstrap(rs, a.boot, 20, rng)
    out["secondary_nested_threshold"] = {"chosen_by_year": chosen, **sec_sum,
                                         "sharpe_ci95": np.percentile(np.sqrt(252) * bs.mean(1) / bs.std(1, ddof=1), [2.5, 97.5]).tolist(),
                                         "beats_spy_cagr": sec_sum["cagr"] > spy_sum["cagr"]}

    # ---- Deflated Sharpe: 28 earlier trials + primary + secondary
    trial_sr = [json.load(open(f))["strategy"]["sharpe"] for f in glob.glob(str(ROOT / "research/trials_2026-11-19/*/*.json"))]
    audit = json.load(open(ROOT / "research/audit_2026-11-19_results.json"))
    trial_sr += [v["sharpe"] for v in audit["runs"].values()]
    trial_sr += [main_sum["sharpe"], sec_sum["sharpe"]]
    trial_sr = np.array([s for s in trial_sr if s is not None and np.isfinite(s)])
    rr = pd.Series(r)
    dsr = deflated_sharpe(float(r.mean() / r.std(ddof=1)), trial_sr / np.sqrt(252), r.size, float(rr.skew()), float(rr.kurt() + 3))
    tests["4_deflated_sharpe"] = {**dsr, "pass": dsr["dsr"] >= 0.95}

    # ---- Robustness warnings
    pnl = main_res.trades["PnL$"].sort_values(ascending=False) if len(main_res.trades) else pd.Series(dtype=float)
    yrs_total = (len(main_res.equity) - 1) / 252
    v_wo = main_res.equity[-1] - pnl.head(3).sum()
    cagr_wo = float((v_wo / v0) ** (1 / yrs_total) - 1) if v_wo > 0 else -1.0
    tests["5_remove_top3"] = {"top3_pnl": pnl.head(3).round(2).tolist(), "cagr_without_top3": cagr_wo, "pass": cagr_wo > spy_sum["cagr"]}

    by_year = {}
    for y in years:
        lo, hi = pd.Timestamp(f"{y - 1}-12-31"), pd.Timestamp(f"{y}-12-31")
        by_year[int(y)] = {"strategy": segment(main_res.equity, panel.dates, lo, hi)["total_return"],
                           "spy": segment(spy, panel.dates, lo, hi)["total_return"]}
    wins = sum(v["strategy"] > v["spy"] for v in by_year.values())
    tests["6_year_by_year"] = {**by_year, "years_beating_spy": wins, "pass": wins >= 5}

    seed_sharpes = {}
    for k in sorted(test["Seed"].unique()):
        sp = bc.build_panel(to_dict(test[test["Seed"] == k].drop(columns=["Seed"])))
        seed_sharpes[int(k)] = bc.summarize(bc.run_strategy(sp, p0), v0)["sharpe"]
    tests["7_seed_stability"] = {"sharpe_by_seed": seed_sharpes, "pass": all(s > 0 for s in seed_sharpes.values())}
    tests["8_vs_reversal"] = {"strategy_sharpe": main_sum["sharpe"], "reversal_sharpe": rev_sum["sharpe"],
                              "pass": main_sum["sharpe"] > rev_sum["sharpe"]}
    tests["9_vs_ew_universe"] = {"strategy_cagr": main_sum["cagr"], "ew_cagr": out["comparators"]["ew_universe"]["cagr"],
                                 "pass": main_sum["cagr"] > out["comparators"]["ew_universe"]["cagr"]}
    tests["10_cost40"] = {"cagr": cost40_sum["cagr"], "spy": spy_sum["cagr"], "pass": cost40_sum["cagr"] > spy_sum["cagr"]}

    out["tests"] = tests
    core = all(tests[k]["pass"] for k in ["1_cagr_vs_spy", "2_shuffle_control", "3_bootstrap", "4_deflated_sharpe"])
    out["verdict"] = "edge SUPPORTED" if core else "edge NOT SUPPORTED on current evidence"
    out["warnings"] = [k for k in ["5_remove_top3", "6_year_by_year", "7_seed_stability", "8_vs_reversal",
                                   "9_vs_ew_universe", "10_cost40"] if not tests[k]["pass"]]

    name = "walkforward_2026-11-19_results" + ("_smoke" if "smoke" in a.dir else "") + ".json"
    (ROOT / "research" / name).write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    print(f"Window {out['window']}, universe {out['universe']}, SPY CAGR {spy_sum['cagr']:+.1%} Sharpe {spy_sum['sharpe']:.2f}")
    print(f"PRIMARY  CAGR {main_sum['cagr']:+.1%}  Sharpe {main_sum['sharpe']:.2f}  MDD {main_sum['max_drawdown']:+.1%}  trades {main_sum['trades']}")
    print(f"SECONDARY CAGR {sec_sum['cagr']:+.1%}  Sharpe {sec_sum['sharpe']:.2f}  thresholds {chosen}")
    print(f"EW univ  CAGR {out['comparators']['ew_universe']['cagr']:+.1%}  | reversal Sharpe {rev_sum['sharpe']:.2f}")
    for k, v in tests.items():
        print(f"  TEST {k:20s} {'PASS' if v['pass'] else 'FAIL'}")
    print("VERDICT:", out["verdict"], "| warnings:", out["warnings"])


if __name__ == "__main__":
    main()
