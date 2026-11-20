"""Bid-ask spread estimates for every security-session (acceptance test K1).

Corwin-Schultz (2012) and Abdi-Ranaldo (2017) two-day estimators from pit.costs, computed on
split-adjusted prices (so splits are not mistaken for spreads), as 21-session trailing medians.
`spread_est` = Corwin-Schultz where available, otherwise Abdi-Ranaldo; both are kept.

Usage: python -m pit.spreads   (from TrainingData/) -> indicators_data/pit/spreads_pit.parquet + audit
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import OUTPUT
from .costs import abdi_ranaldo_spread, corwin_schultz_spread
from .manifest import write_manifest

MIN_PERIODS = 10


def estimate(px: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    m = px.merge(fac[["session", "isin", "split_factor"]], on=["session", "isin"], how="left")
    m["split_factor"] = m["split_factor"].fillna(1.0)
    out = []
    for isin, d in m.sort_values(["isin", "session"]).groupby("isin", sort=False):
        h, l, c = (d[k] * d.split_factor for k in ("high", "low", "close"))
        cs = corwin_schultz_spread(h, l, c, min_periods=MIN_PERIODS).clip(lower=0)
        ar = abdi_ranaldo_spread(h, l, c, min_periods=MIN_PERIODS).clip(lower=0)
        out.append(pd.DataFrame({"session": d.session.values, "isin": isin, "spread_cs": cs.values, "spread_ar": ar.values}))
    s = pd.concat(out, ignore_index=True)
    s["spread_est"] = s.spread_cs.where(np.isfinite(s.spread_cs), s.spread_ar)
    return s


def main() -> None:
    px = pd.read_parquet(OUTPUT / "prices_pit.parquet", columns=["session", "isin", "high", "low", "close", "valid_bar", "member"])
    fac = pd.read_parquet(OUTPUT / "adjustment_factors_pit.parquet")
    valid = px[px.valid_bar]
    s = estimate(valid, fac)
    mem = px[px.member][["session", "isin"]].merge(s, on=["session", "isin"], how="left")
    cov = float(np.isfinite(mem.spread_est).mean())
    q = mem.spread_est.quantile([0.1, 0.5, 0.9]).round(5).tolist()
    by_year = mem.assign(y=mem.session.str[:4]).groupby("y").spread_est.median().round(5).to_dict()
    audit = {"member_sessions": len(mem), "coverage": round(cov, 5), "K1_pass": cov >= 0.95,
             "member_spread_p10_p50_p90": q, "median_by_year": by_year,
             "corwin_schultz_share": round(float(np.isfinite(mem.spread_cs).mean()), 5),
             "note": "Estimates on split-adjusted prices; negative two-day estimates set to 0 before the median."}
    out = OUTPUT / "spreads_pit.parquet"
    s.to_parquet(out, index=False)
    (OUTPUT / "spreads_pit_audit.json").write_text(json.dumps(audit, indent=2))
    write_manifest(OUTPUT / "spreads_pit_manifest.json", [out],
                   [OUTPUT / "prices_pit.parquet", OUTPUT / "adjustment_factors_pit.parquet"], {"script": "pit.spreads"})
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
