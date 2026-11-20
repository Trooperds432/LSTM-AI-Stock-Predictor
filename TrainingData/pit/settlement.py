"""Terminal outcomes for index members whose trading stops when they leave the index (acceptance S1/S2).

Sources, in priority order:
  1. Hand-verified terms (TrainingData/lseg_verified_merger_terms.csv, elective and contingent term files).
  2. LSEG M&A deal records (pit/lseg_deals.py download): the completed deal where the security is the
     target and whose effective date is within 10 days of its last valid bar.
     - Cash-only deals: cash per share = LSEG "Share Price Paid by Acquiror for Target Shares"
       (matched all 22 hand-verified cash deals within 0.5%).
     - Stock/mixed deals: exchange ratio and cash parsed from the LSEG synopsis (recovered 28 of 29
       hand-verified ratios); kept as `terms_parsed_unvalidated` until checked against the successor price.
  3. Everything else -> `needs_manual` (no guessing).

Usage: python -m pit.settlement  (from TrainingData/) -> indicators_data/pit/settlement_pit.csv + audit json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import overrides as ov
from .config import LSEG_ROOT, OUTPUT, TRAINING_DATA

RATIO = re.compile(r"(\d+(?:\.\d+)?)\s+(?:(?:new|common|ordinary|class\s+\w|[A-Z][\w&.'-]*)\s+){0,5}(?:common\s+)?shares?\b", re.I)
CASH = re.compile(r"(?:US)?\$\s?(\d+(?:\.\d+)?)\s+(?:in\s+cash|cash)|USD\s?(\d+(?:\.\d+)?)\s+(?:in\s+)?cash", re.I)
MATCH_DAYS = 10
CASH_TOL = 0.02


def parse_terms(synopsis: str) -> tuple[list[float], list[float]]:
    """Candidate exchange ratios (0 < r < 20) and cash amounts mentioned in an LSEG deal synopsis."""
    t = str(synopsis)
    ratios = [float(x) for x in RATIO.findall(t) if 0 < float(x) < 20]
    cash = [float(a or b) for a, b in CASH.findall(t)]
    return ratios, cash


def latest_deals_dir() -> Path:
    runs = sorted((LSEG_ROOT / "deals").glob("*"))
    if not runs:
        raise FileNotFoundError("no LSEG deals download; run pit/lseg_deals.py")
    return runs[-1]


def select_target_deals(targets: pd.DataFrame, deals: pd.DataFrame, retire: pd.DataFrame) -> pd.DataFrame:
    """For each target security, the completed deal where it is the target closest to its last valid bar."""
    deals = deals.merge(retire[["Instrument", "Organization PermID"]], on="Instrument")
    deals = deals[(deals["Target PermID"] == deals["Organization PermID"]) & deals["Deal Status"].eq("Completed")].copy()
    deals["eff"] = pd.to_datetime(deals["Date Effective"], errors="coerce")
    out = []
    for r in targets.itertuples(index=False):
        ref = pd.Timestamp(r.last_bar if isinstance(r.last_bar, str) else r.last_member)
        c = deals[deals.Instrument == r.ric].assign(gap=lambda d: (d.eff - ref).dt.days.abs()).sort_values("gap")
        rec = {"isin": r.isin, "ric": r.ric, "last_member": r.last_member, "last_bar": r.last_bar}
        if len(c) and c.gap.iloc[0] <= MATCH_DAYS:
            b = c.iloc[0]
            rec.update(deal_id=b["M&A Deal Identifier"], acquiror=b["Acquiror Full Name"], effective=b["Date Effective"],
                       structure=b["Consideration Structure"],
                       price_paid=pd.to_numeric(b["Share Price Paid by Acquiror for Target Shares"], errors="coerce"),
                       synopsis=b["Deal Synopsis"])
        out.append(rec)
    return pd.DataFrame(out)


def manual_terms_by_isin(targets: pd.DataFrame) -> dict[str, dict]:
    """Map pit/overrides/manual_settlement_terms.csv rows (keyed by RIC + completion date) to target ISINs.
    A RIC that carried several securities (JCI.N) is matched to the target whose last bar is closest."""
    out = {}
    for m in ov.load("manual_settlement_terms").to_dict("records"):
        c = targets[targets.ric == m["ric"]]
        if c.empty:
            raise ValueError(f"manual settlement row matches no target: {m['ric']}")
        gap = (pd.to_datetime(c.last_bar) - pd.Timestamp(m["completion_date"])).abs()
        isin = c.loc[gap.idxmin(), "isin"]
        if isin in out:
            raise ValueError(f"two manual settlement rows for {isin}")
        out[isin] = m
    return out


def classify(sel: pd.DataFrame, last_close: pd.Series, verified: pd.DataFrame, special: set[str],
             manual: dict[str, dict] | None = None) -> pd.DataFrame:
    manual = manual or {}
    rows = []
    for r in sel.to_dict("records"):
        lc = last_close.get(r["isin"], np.nan)
        rec = {k: r.get(k) for k in ["isin", "ric", "last_member", "last_bar", "deal_id", "acquiror", "structure"]}
        rec.update(last_close=lc, cash_per_share=np.nan, successor_shares_per_share=np.nan, successor=None)
        if r["isin"] in set(verified.predecessor_isin):
            v = verified[verified.predecessor_isin == r["isin"]].iloc[0]
            rec.update(outcome="merger_verified_terms", cash_per_share=v.cash_usd_per_share,
                       successor_shares_per_share=v.successor_shares_per_share, successor=v.successor_name,
                       source=v.source, status="terms_verified_settlement_pending")
        elif r["isin"] in manual:
            m = manual[r["isin"]]
            flagged = any(k in str(m["election_note"]) for k in ("SECONDARY", "confirm"))
            rec.update(outcome=m["outcome"], cash_per_share=float(m["cash_per_share"]),
                       successor_shares_per_share=float(m["successor_shares_per_share"]), successor=m["successor_name"],
                       other_consideration=m["other_consideration"] or None, election_note=m["election_note"] or None,
                       source=m["source"], status="terms_sourced_needs_confirmation" if flagged else "terms_sourced_settlement_pending")
        elif r["isin"] in special:
            rec.update(outcome="elective_or_contingent", source="TrainingData/lseg_elective_terms.csv|lseg_contingent_terms.csv",
                       status="terms_verified_settlement_pending")
        elif pd.isna(r.get("deal_id")):
            rec.update(outcome="unknown", source=None, status="needs_manual")
        elif r["structure"] == "Cash Only" and pd.notna(r["price_paid"]):
            err = r["price_paid"] / lc - 1 if lc and np.isfinite(lc) else np.nan
            ok = np.isfinite(err) and abs(err) <= CASH_TOL
            rec.update(outcome="cash_merger", cash_per_share=r["price_paid"], successor=r["acquiror"],
                       source=f"lseg_deal:{r['deal_id']}", settle_vs_last_close=err,
                       status="validated" if ok else "needs_manual")
        else:
            ratios, cash = parse_terms(r["synopsis"])
            if len(set(ratios)) == 1:
                rec.update(outcome="stock_or_mixed_merger", successor_shares_per_share=ratios[0],
                           cash_per_share=cash[0] if len(set(cash)) == 1 else (0.0 if r["structure"] == "Stock Only" else np.nan),
                           successor=r["acquiror"], source=f"lseg_deal:{r['deal_id']}", status="terms_parsed_unvalidated")
            else:
                rec.update(outcome="stock_or_mixed_merger", successor=r["acquiror"], source=f"lseg_deal:{r['deal_id']}",
                           status="needs_manual", parse_note=f"ratios={ratios} cash={cash}")
        rows.append(rec)
    return pd.DataFrame(rows)


STOCK_TOL = 0.05


def successor_bars() -> pd.DataFrame:
    """All LSEG daily bars by RIC: every price run plus reference_extra successor downloads."""
    from .prices import load_all_bars
    bars, _ = load_all_bars()
    bars = bars[bars.valid_bar][["ric", "session", "open", "close"]]
    extra = []
    for f in sorted((LSEG_ROOT / "reference_extra").glob("*/bars_*.csv")):
        b = pd.read_csv(f)
        date_col = b.columns[0]
        ric = json.loads((f.parent / "manifest.json").read_text())["args"]
        name = f.stem[5:]
        ric = next((r for r in ric if r.replace(".", "_").replace("^", "_") == name), None)
        if ric is None:
            continue
        b = b.rename(columns={date_col: "session", "OPEN_PRC": "open", "TRDPRC_1": "close"})
        b["session"] = b["session"].astype(str).str[:10]
        b = b[(pd.to_numeric(b.open, errors="coerce") > 0)]
        extra.append(b.assign(ric=ric)[["ric", "session", "open", "close"]])
    return pd.concat([bars, *extra], ignore_index=True).drop_duplicates(["ric", "session"])


def validate(st: pd.DataFrame, bars: pd.DataFrame, succ: pd.DataFrame, special_terms: dict) -> pd.DataFrame:
    """S2: settlement value = cash + ratio x successor open on its first session after the target's last bar,
    compared with the target's last valid close. Cash-only tolerance 2%, any stock component 5%."""
    succ_map = dict(zip(succ.target_isin, succ.successor_ric))
    by_ric = {r: g.sort_values("session") for r, g in bars.groupby("ric")}
    out = []
    for r in st.to_dict("records"):
        cash = r.get("cash_per_share")
        ratio = r.get("successor_shares_per_share")
        if r["isin"] in special_terms:
            cash, ratio = special_terms[r["isin"]]
        cash = 0.0 if pd.isna(cash) else float(cash)
        ratio = 0.0 if pd.isna(ratio) else float(ratio)
        ric = succ_map.get(r["isin"], "") or ""
        if r["isin"] == "US9314271084":  # Walgreens take-private: LSEG-parsed ratio is wrong; cash terms only
            ratio = 0.0
        open_px, open_date, same_close = np.nan, None, np.nan
        if ratio > 0 and ric in by_ric:
            g = by_ric[ric]
            limit = (pd.Timestamp(r["last_bar"]) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            after = g[(g.session > str(r["last_bar"])) & (g.session <= limit)].head(1)
            if len(after):
                open_px, open_date = float(after.open.iloc[0]), after.session.iloc[0]
            same = g[g.session == str(r["last_bar"])]
            if len(same):
                same_close = float(same.close.iloc[0])  # diagnostic: arbitrage value on the target's last day
        if ratio > 0 and not np.isfinite(open_px):
            res, value, err = "no_successor_price", np.nan, np.nan
        else:
            value = cash + (ratio * open_px if ratio > 0 else 0.0)
            err = value / r["last_close"] - 1 if r["last_close"] else np.nan
            tol = STOCK_TOL if ratio > 0 else CASH_TOL
            res = "pass" if np.isfinite(err) and abs(err) <= tol else "outside_tolerance"
        same_err = (cash + ratio * same_close) / r["last_close"] - 1 if ratio > 0 and np.isfinite(same_close) else np.nan
        out.append({**r, "successor_ric": ric, "successor_open_date": open_date, "successor_open": open_px,
                    "settlement_value": value, "settle_vs_last_close": err, "S2": res,
                    "sameday_value_vs_last_close": same_err})
    return pd.DataFrame(out)


def main() -> None:
    d = latest_deals_dir()
    deals = pd.read_csv(d / "deals.csv", dtype=str)
    retire = pd.read_csv(d / "retire.csv", dtype=str)
    targets = pd.read_csv(OUTPUT / "settlement_targets.csv", dtype=str)
    verified = pd.read_csv(TRAINING_DATA / "lseg_verified_merger_terms.csv")
    special = set(pd.read_csv(TRAINING_DATA / "lseg_elective_terms.csv").predecessor_isin) | \
        set(pd.read_csv(TRAINING_DATA / "lseg_contingent_terms.csv").predecessor_isin)
    px = pd.read_parquet(OUTPUT / "prices_pit.parquet", columns=["session", "isin", "close", "valid_bar"])
    last_close = px[px.valid_bar].sort_values("session").groupby("isin").close.last()
    sel = select_target_deals(targets, deals, retire)
    st = classify(sel, last_close, verified, special, manual_terms_by_isin(targets))
    # Elective (Hudson City: prorated stock fraction) and contingent (Hologic: cash + CVR) term files
    el = pd.read_csv(TRAINING_DATA / "lseg_elective_terms.csv")
    ct = pd.read_csv(TRAINING_DATA / "lseg_contingent_terms.csv")
    special_terms = {r.predecessor_isin: ((1 - r.stock_election_fraction_prorated) * r.default_cash_usd_per_share,
                                          r.stock_election_fraction_prorated * r.stock_exchange_ratio) for r in el.itertuples()}
    special_terms.update({r.predecessor_isin: (r.cash_usd_per_share, 0.0) for r in ct.itertuples()})
    succ = pd.read_csv(ov.OVERRIDES / "settlement_successor_rics.csv", dtype=str, keep_default_na=False)
    st = validate(st, successor_bars(), succ, special_terms)
    exc = pd.read_csv(ov.OVERRIDES / "settlement_s2_exceptions.csv")
    st = st.merge(exc[["ric", "category", "fallback"]].rename(columns={"category": "S2_exception"}), on="ric", how="left")
    fb = {"settlement_value": st.settlement_value, "last_close": st.last_close,
          "zero": pd.Series(0.0, index=st.index), "cash_only": st.cash_per_share.fillna(0.0)}
    st["settlement_value_used"] = st.settlement_value
    for k, v in fb.items():
        m = st.fallback.eq(k)
        st.loc[m, "settlement_value_used"] = v[m]
    st.to_csv(OUTPUT / "settlement_pit.csv", index=False, lineterminator="\n")
    review = st[st.status.eq("needs_manual")]
    review.to_csv(OUTPUT / "settlement_needs_manual.csv", index=False, lineterminator="\n")
    audit = {"deals_download": str(d.name), "targets": len(st), "by_status": st.status.value_counts().to_dict(),
             "by_outcome": st.outcome.value_counts().to_dict(),
             "S1_pass": bool((st.outcome != "unknown").all() and (st.status != "needs_manual").all()),
             "needs_manual": review[["ric", "last_bar", "outcome"]].values.tolist(),
             "S2_counts": st.S2.value_counts().to_dict(),
             "S2_undocumented": st[(st.S2 != "pass") & st.S2_exception.isna()].ric.tolist(),
             "S2_pass_with_documented_exceptions": bool(((st.S2 == "pass") | st.S2_exception.notna()).all()),
             "S2_exceptions_by_category": st.S2_exception.value_counts().to_dict(),
             "S2_exceptions": st[st.S2 != "pass"][["ric", "outcome", "successor_ric", "settlement_value", "last_close",
                                                    "settle_vs_last_close", "S2"]].round(4).values.tolist()}
    (OUTPUT / "settlement_pit_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps({k: audit[k] for k in ["targets", "by_status", "by_outcome", "S1_pass"]}))


if __name__ == "__main__":
    main()
