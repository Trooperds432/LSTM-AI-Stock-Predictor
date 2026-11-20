"""Executable daily price panel by security (ISIN), using each security's primary RIC per session.

Outputs (indicators_data/pit/):
  prices_pit.parquet       session, isin, ric, open, high, low, close, volume_venue, valid_bar, member, source_file
  prices_pit_audit.json    P1 coverage, P3 exception ledger summary
research/pit_alias_adjudication.json   P2 / P2a evidence vs Alpha Vantage consolidated data

Notes
- LSEG ACVOL_UNS is venue (primary-listing) volume, ~25-38% of consolidated (see acceptance amendment
  2026-11-20). It is stored as `volume_venue`, never as consolidated volume.
- Prices are unadjusted; corporate-action adjustment happens in pit.corporate_actions.

Usage: python -m pit.prices   (run from TrainingData/)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import overrides as ov
from .config import OUTPUT, PROJECT_ROOT, SNAPSHOT, TRAINING_DATA
from .manifest import write_manifest

FIELDS = {"OPEN_PRC": "open", "HIGH_1": "high", "LOW_1": "low", "TRDPRC_1": "close", "ACVOL_UNS": "volume_venue"}
# Priority: earlier runs win when the same (ric, session) appears twice with a valid bar
PRICE_RUNS = ["main", "recovery", "bemis", "gap_recovery", "micro_focus", "successors", "tyco"]


def valid_mask(df: pd.DataFrame) -> pd.Series:
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume_venue"])
    return (l > 0) & (h >= o) & (h >= c) & (l <= o) & (l <= c) & (v > 0)


def load_all_bars() -> tuple[pd.DataFrame, list[Path]]:
    frames, files = [], []
    for prio, run in enumerate(PRICE_RUNS):
        folder = SNAPSHOT[run]
        man = json.loads((folder / "manifest.json").read_text())
        for rec in man["datasets"]:
            if rec["kind"] != "daily_prices" or rec["status"] != "retrieved":
                continue
            path = folder / rec["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
                raise ValueError(f"checksum mismatch: {path}")
            files.append(path)
            uni = rec["universe"]
            batch = isinstance(uni, list)
            raw = pd.read_csv(path, header=[0, 1] if batch else 0, index_col=0)
            for ric in (uni if batch else [uni]):
                if batch and ric not in raw:
                    continue
                b = (raw[ric] if batch else raw).reindex(columns=list(FIELDS)).apply(pd.to_numeric, errors="coerce")
                b = b.rename(columns=FIELDS).dropna(how="all")
                b["session"] = b.index.astype(str).str[:10]
                b["ric"], b["source_file"], b["_prio"] = ric, str(path.relative_to(TRAINING_DATA)), prio
                frames.append(b.reset_index(drop=True))
    bars = pd.concat(frames, ignore_index=True)
    bars["valid_bar"] = valid_mask(bars)
    # keep the best row per (ric, session): valid before invalid, then run priority
    bars = bars.sort_values(["ric", "session", "valid_bar", "_prio"], ascending=[True, True, False, True])
    bars = bars.drop_duplicates(["ric", "session"]).drop(columns="_prio")
    return bars, files


def map_to_securities(bars: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """Attach each bar to the security whose primary RIC it is on that session."""
    prim = sm[sm.role.eq("primary")]
    out = []
    for r in prim.itertuples(index=False):
        b = bars[(bars.ric == r.ric) & (bars.session >= r.valid_from) & (bars.session < r.valid_to)]
        if len(b):
            out.append(b.assign(isin=r.isin))
    px = pd.concat(out, ignore_index=True)
    dup = px.duplicated(["isin", "session"], keep=False)
    if dup.any():
        raise ValueError(f"two primary bars for one security-session: {px[dup].head().to_dict('records')}")
    return px


def adjudicate_aliases(bars: pd.DataFrame) -> dict:
    """P2 / P2a: compare primary and alias series with Alpha Vantage consolidated daily data."""
    av_dir = TRAINING_DATA / "indicators_data" / "raw" / "stocksData"
    result = {}
    for t in ov.load("listing_transfers").itertuples(index=False):
        tk = t.primary_ric.split(".")[0]
        av = pd.read_csv(av_dir / f"{tk}_daily.csv", dtype={"date": str}).set_index("date").add_prefix("av_")
        rows = {}
        for role, ric in (("primary", t.primary_ric), ("alias", t.alias_ric)):
            b = bars[bars.ric.eq(ric) & bars.valid_bar].set_index("session").join(av, how="inner")
            for period, m in (("pre_transfer", b.index < t.transfer_session), ("post_transfer", b.index >= t.transfer_session)):
                x = b[m]
                if len(x) == 0:
                    continue
                vr = x.volume_venue / x.av_volume
                rows[f"{role}:{ric}:{period}"] = {
                    "sessions": int(len(x)), "first": x.index.min(), "last": x.index.max(),
                    "close_match_1bp": round(float((x.close / x.av_close - 1).abs().lt(1e-4).mean()), 4),
                    "venue_to_consolidated_volume_median": round(float(vr.median()), 4),
                    "volume_within_10pct": round(float(((vr - 1).abs() <= 0.10).mean()), 4)}
        prim = [v for k, v in rows.items() if k.startswith("primary")]
        p2a = all(v["close_match_1bp"] >= 0.95 for v in prim)
        p2 = all(v["volume_within_10pct"] >= 0.95 for v in prim)
        result[t.isin] = {"primary_ric": t.primary_ric, "alias_ric": t.alias_ric, "transfer_session": t.transfer_session,
                          "series": rows, "P2a_close_pass": p2a, "P2_volume_pass": p2}
    return {"consolidated_source": "Alpha Vantage TIME_SERIES_DAILY (local raw/stocksData)", "pairs": result,
            "P2a_pass": all(v["P2a_close_pass"] for v in result.values()),
            "P2_pass": all(v["P2_volume_pass"] for v in result.values())}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sm = pd.read_csv(OUTPUT / "security_master.csv", dtype=str)
    members = pd.read_csv(OUTPUT / "membership_pit.csv", dtype=str)
    bars, files = load_all_bars()
    px = map_to_securities(bars, sm)

    mem_keys = members.assign(member=True)
    panel = px.merge(mem_keys, on=["session", "isin"], how="outer")
    panel["member"] = panel["member"].fillna(False).astype(bool)
    panel["valid_bar"] = panel["valid_bar"].fillna(False).astype(bool)
    panel = panel.sort_values(["isin", "session"]).reset_index(drop=True)
    cols = ["session", "isin", "ric", "open", "high", "low", "close", "volume_venue", "valid_bar", "member", "source_file"]
    panel = panel[cols]
    out = OUTPUT / "prices_pit.parquet"
    panel.to_parquet(out, index=False)

    mem = panel[panel.member]
    miss = mem[~mem.valid_bar]
    ledger = ov.load("member_session_exceptions")
    in_ledger = miss.merge(ledger[["session", "isin", "event_type"]], on=["session", "isin"], how="left")
    unexplained = in_ledger[in_ledger.event_type.isna()]
    adjud = adjudicate_aliases(bars)
    (PROJECT_ROOT / "research" / "pit_alias_adjudication.json").write_text(json.dumps(adjud, indent=2))
    cov = 1 - len(miss) / len(mem)
    audit = {"member_sessions": int(len(mem)), "member_sessions_with_valid_bar": int(mem.valid_bar.sum()),
             "coverage": round(cov, 6), "P1_pass": cov >= 0.999,
             "missing_member_sessions": int(len(miss)),
             "missing_in_exception_ledger": int(in_ledger.event_type.notna().sum()),
             "missing_not_in_ledger": int(len(unexplained)), "P3_pass": len(unexplained) == 0,
             "missing_by_reason": in_ledger.event_type.fillna("NOT_IN_LEDGER").value_counts().to_dict(),
             "unexplained_by_isin": unexplained.groupby("isin").session.agg(["count", "min", "max"]).reset_index().to_dict("records"),
             "P2a_pass": adjud["P2a_pass"], "P2_pass": adjud["P2_pass"],
             "price_files_checksum_verified": len(files), "securities_with_prices": int(px["isin"].nunique())}
    (OUTPUT / "prices_pit_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    write_manifest(OUTPUT / "prices_pit_manifest.json", [out], files + [OUTPUT / "security_master.csv", OUTPUT / "membership_pit.csv"],
                   {"script": "pit.prices"})
    print(json.dumps({k: v for k, v in audit.items() if k != "unexplained_by_isin"}, default=str))
    for r in audit["unexplained_by_isin"][:15]:
        print("  unexplained", r)


if __name__ == "__main__":
    main()
