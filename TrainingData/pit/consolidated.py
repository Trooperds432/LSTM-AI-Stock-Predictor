"""Consolidated volume and closing bid/ask per security-session, plus acceptance re-test P2 and test K2.

Input: indicators_data/lseg/consolidated/<stamp>/ (pit/lseg_consolidated.py), mapped back to securities
through the primary RIC each consolidated RIC was derived from and the dated security master.
Output: indicators_data/pit/consolidated_pit.parquet
  session, isin, volume_consolidated, close_consolidated, bid, ask, quoted_spread ((ask-bid)/mid)

P2 (re-test, consolidated series): volume within +/-10% of Alpha Vantage consolidated volume on >= 95% of
  overlapping sessions, per security with a local Alpha Vantage file.
K2: estimated spread (pit.spreads 21-day median) vs 21-day median quoted closing spread; error reported.

Usage: python -m pit.consolidated   (from TrainingData/)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import LSEG_ROOT, OUTPUT, TRAINING_DATA
from .manifest import write_manifest


def load_raw() -> tuple[pd.DataFrame, list]:
    run = sorted((LSEG_ROOT / "consolidated").glob("*"))[-1]
    man = json.loads((run / "manifest.json").read_text())
    back = {c: p for p, c in man["pairs"]}
    frames, files = [], []
    for key, rec in sorted(man["batches"].items()):
        if rec["status"] != "ok":
            continue
        f = run / f"{key}.csv"
        files.append(f)
        raw = pd.read_csv(f, header=[0, 1], index_col=0)
        for cons in rec["rics"]:
            if cons not in raw.columns.get_level_values(0):
                continue
            b = raw[cons].apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if b.empty:
                continue
            frames.append(pd.DataFrame({"session": b.index.astype(str).str[:10], "ric": back[cons], "cons_ric": cons,
                                        "volume_consolidated": b.get("ACVOL_UNS"), "close_consolidated": b.get("TRDPRC_1"),
                                        "bid": b.get("BID"), "ask": b.get("ASK")}))
    return pd.concat(frames, ignore_index=True), files


def main() -> None:
    raw, files = load_raw()
    px = pd.read_parquet(OUTPUT / "prices_pit.parquet", columns=["session", "isin", "ric", "close", "valid_bar", "member"])
    m = px.merge(raw, on=["ric", "session"], how="left")
    mid = (m.bid + m.ask) / 2
    m["quoted_spread"] = np.where((m.bid > 0) & (m.ask >= m.bid), (m.ask - m.bid) / mid, np.nan)
    out = m[["session", "isin", "ric", "cons_ric", "volume_consolidated", "close_consolidated", "bid", "ask", "quoted_spread"]]
    out_path = OUTPUT / "consolidated_pit.parquet"
    out.to_parquet(out_path, index=False)

    mem = m[m.member & m.valid_bar]
    cov_vol = float((mem.volume_consolidated > 0).mean())
    cov_q = float(np.isfinite(mem.quoted_spread).mean())
    close_match = float(((mem.close_consolidated / mem.close - 1).abs() < 1e-4).mean())

    # P2 re-test against local Alpha Vantage files (consolidated volume), per security
    av_dir = TRAINING_DATA / "indicators_data" / "raw" / "stocksData"
    per = []
    for isin, d in mem.groupby("isin"):
        tk = d.ric.iloc[-1].split(".")[0].split("^")[0]
        f = av_dir / f"{tk}_daily.csv"
        if "^" in d.ric.iloc[-1] or not f.exists():
            continue  # expired RICs: AV ticker may belong to another company
        av = pd.read_csv(f, usecols=["date", "close", "volume"]).rename(columns={"date": "session"})
        j = d.merge(av, on="session", how="inner", suffixes=("", "_av"))
        j = j[(j.volume > 0) & (j.volume_consolidated > 0) & ((j.close / j.close_av - 1).abs() < 1e-3)]
        if len(j) < 60:
            continue
        r = j.volume_consolidated / j.volume
        per.append({"isin": isin, "ticker": tk, "sessions": len(j), "median_ratio": float(r.median()),
                    "within_10pct": float(((r - 1).abs() <= 0.10).mean())})
    per = pd.DataFrame(per)
    p2_share = float((per.within_10pct >= 0.95).mean()) if len(per) else float("nan")

    # K2: estimated vs quoted spread (21-session medians)
    sp = pd.read_parquet(OUTPUT / "spreads_pit.parquet")
    q = m[m.valid_bar].sort_values(["isin", "session"])[["session", "isin", "quoted_spread"]].copy()
    q["quoted_21d"] = q.groupby("isin").quoted_spread.transform(lambda s: s.rolling(21, min_periods=10).median())
    k = q.merge(sp, on=["session", "isin"]).merge(mem[["session", "isin"]], on=["session", "isin"])
    k = k[np.isfinite(k.quoted_21d) & np.isfinite(k.spread_est)]
    err = k.spread_est - k.quoted_21d
    audit = {
        "member_sessions": int(len(mem)), "consolidated_volume_coverage": round(cov_vol, 5),
        "quoted_spread_coverage": round(cov_q, 5), "consolidated_close_matches_primary_close": round(close_match, 5),
        "P2_retest": {"securities_compared": int(len(per)), "share_passing_95pct_within_10pct": round(p2_share, 4),
                      "median_ratio_p10_p50_p90": per.median_ratio.quantile([.1, .5, .9]).round(4).tolist() if len(per) else [],
                      "pass": bool(p2_share >= 0.95), "worst": per.nsmallest(8, "within_10pct").round(4).to_dict("records")},
        "K2": {"sessions": int(len(k)), "quoted_21d_median_p10_p50_p90": k.quoted_21d.quantile([.1, .5, .9]).round(5).tolist(),
               "estimate_median": round(float(k.spread_est.median()), 5),
               "median_abs_error": round(float(err.abs().median()), 5), "median_error": round(float(err.median()), 5),
               "rank_corr_by_security": round(float(k.groupby("isin")[["spread_est", "quoted_21d"]].median().corr("spearman").iloc[0, 1]), 4),
               "note": "Reported, not pass/fail (preregistered)."}}
    (OUTPUT / "consolidated_pit_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    write_manifest(OUTPUT / "consolidated_pit_manifest.json", [out_path], files + [OUTPUT / "prices_pit.parquet"],
                   {"script": "pit.consolidated"})
    print(json.dumps({k2: v for k2, v in audit.items() if k2 != "P2_retest"}, default=str))
    print(json.dumps({k2: v for k2, v in audit["P2_retest"].items() if k2 != "worst"}))
    for w in audit["P2_retest"]["worst"]:
        print("  worst", w)


if __name__ == "__main__":
    main()
