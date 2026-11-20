"""Corporate actions for the PIT panel: trading-effective splits/consolidations/spin-offs, cash dividends,
and cumulative adjustment factors (acceptance tests C1-C3).

Rules
- Price-ratio events (Share Split, Share Consolidation, Demerger, Rights Issue, Share Conversion):
  LSEG's "Capital Change Effective Date" is often the day before trading reflects the change. The trading
  date is the first of {effective date, next session} whose overnight ratio open_t / close_{t-1} is within
  10% of the LSEG adjustment factor. Otherwise the event is `unverified` and listed for review.
- Dividends: de-duplicated cash amounts on the ex-date session (LSEG did not return currency; assumed USD
  for these US listings and reported as an assumption).
- Factors are cumulative and backward-looking from the latest session: adjusted = raw * factor, so the most
  recent prices are unadjusted. `split_factor` uses price-ratio events only; `tr_factor` adds dividends.

Usage: python -m pit.corporate_actions   (from TrainingData/)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import END, OUTPUT, OVERRIDES, SNAPSHOT, START
from .manifest import write_manifest

RATIO_TYPES = ["Share Split", "Share Consolidation", "Demerger", "Rights Issue", "Share Conversion"]
TOL = 0.10
CONTINUITY = 0.50


def load_actions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    a = SNAPSHOT["actions"]
    cc = pd.concat([pd.read_csv(f) for f in sorted(a.glob("capital_changes_*.csv"))], ignore_index=True)
    dv = pd.concat([pd.read_csv(f) for f in sorted(a.glob("dividends_*.csv"))], ignore_index=True)
    stats = {"capital_changes_raw": len(cc), "capital_change_duplicates": int(cc.duplicated().sum()),
             "dividends_raw": len(dv), "dividend_duplicates": int(dv.duplicated().sum()),
             "dividends_without_ex_date": int(dv["Dividend Ex Date"].isna().sum())}
    return cc.drop_duplicates(), dv.drop_duplicates().dropna(subset=["Dividend Ex Date"]), stats


def to_isin(events: pd.DataFrame, sm: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Attach the security that the RIC carried on the event date (dated security master, primary rows)."""
    prim = sm[sm.role.eq("primary")]
    out = []
    for r in prim.itertuples(index=False):
        e = events[(events.Instrument == r.ric) & (events[date_col] >= r.valid_from) & (events[date_col] < r.valid_to)]
        if len(e):
            out.append(e.assign(isin=r.isin))
    return pd.concat(out, ignore_index=True) if out else events.iloc[0:0].assign(isin=None)


def verify_ratio_events(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Find the trading session at which each price-ratio event shows up in open_t / close_{t-1}."""
    g = {i: d.reset_index(drop=True) for i, d in px.groupby("isin")}
    rows = []
    for e in ev.itertuples(index=False):
        rec = {"isin": e.isin, "ric": e.Instrument, "type": e._1, "lseg_effective": e._2, "factor": e._3,
               "trading_session": None, "observed_ratio": np.nan, "status": "unverified"}
        d = g.get(e.isin)
        if d is None or not (START <= e._2 <= END) or not np.isfinite(e._3) or e._3 <= 0:
            rec["status"] = "outside_window_or_no_prices"
            rows.append(rec)
            continue
        idx = d.index[d.session >= e._2]
        for k in idx[:2]:  # effective date (or first session on/after it) and the next session
            if k == 0:
                continue
            ratio = d.open[k] / d.close[k - 1]
            if np.isfinite(ratio) and abs(ratio / e._3 - 1) <= TOL:
                rec.update(trading_session=d.session[k], observed_ratio=ratio, status="verified")
                break
        if rec["status"] != "verified" and len(idx) and idx[0] > 0:
            k = idx[0]
            rec["observed_ratio"] = d.open[k] / d.close[k - 1]
            if abs(e._3 - 1) < 0.02:  # factor ~1: no price effect expected
                rec["status"] = "no_price_effect"
        rows.append(rec)
    out = pd.DataFrame(rows)
    return verify_combined(out, g)


def verify_combined(out: pd.DataFrame, g: dict) -> pd.DataFrame:
    """Second pass: events on the same security within 3 days (e.g. a reverse split plus spin-offs on one
    session) are tested together: the product of their factors must match the observed ratio."""
    out = out.copy()
    for isin, grp in out[out.status.isin(["unverified", "verified"])].groupby("isin"):
        if not grp.status.eq("unverified").any():
            continue
        d = g[isin]
        dates = pd.to_datetime(grp.lseg_effective)
        for i in grp.index[grp.status.eq("unverified")]:
            near = grp[(dates - pd.Timestamp(out.at[i, "lseg_effective"])).abs() <= pd.Timedelta(days=3)]
            if len(near) < 2:
                continue
            combo = float(np.prod(near.factor))
            first = min(near.lseg_effective)
            idx = d.index[d.session >= first]
            for k in idx[:3]:
                if k == 0:
                    continue
                ratio = d.open[k] / d.close[k - 1]
                hit = abs(ratio / combo - 1) <= TOL
                if hit or abs(combo - 1) < 0.02:
                    status = "verified_combined" if hit and abs(combo - 1) >= 0.02 else "no_price_effect_combined"
                    out.loc[near.index, ["trading_session", "observed_ratio", "status"]] = [d.session[k], ratio, status]
                    break
    return out


def build_factors(px: pd.DataFrame, events: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """Backward cumulative factors per security. Event at session t multiplies all sessions < t by f."""
    out = []
    ev = events[events.status.isin(["verified", "verified_combined", "verified_reviewed", "added_reviewed"])]
    for isin, d in px.groupby("isin"):
        d = d.sort_values("session").reset_index(drop=True)[["session", "isin", "close"]]
        step_split = pd.Series(1.0, index=d.session)
        step_tr = pd.Series(1.0, index=d.session)
        for e in ev[ev["isin"] == isin].itertuples(index=False):
            if e.trading_session in step_split.index:
                step_split[e.trading_session] *= e.factor
                step_tr[e.trading_session] *= e.factor
        for v in divs[divs["isin"] == isin].itertuples(index=False):
            ses = d.session[d.session >= v.ex_date]
            if not len(ses):
                continue
            t = ses.iloc[0]
            k = d.index[d.session == t][0]
            if k == 0:
                continue
            prev_close = d.close[k - 1]
            if np.isfinite(prev_close) and prev_close > v.amount > 0:
                step_tr[t] *= 1 - v.amount / prev_close
        # factor for session s = product of steps strictly after s
        sf = step_split.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)
        tf = step_tr.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)
        out.append(pd.DataFrame({"session": d.session, "isin": isin, "split_factor": sf.values, "tr_factor": tf.values}))
    return pd.concat(out, ignore_index=True)


def continuity(px: pd.DataFrame, fac: pd.DataFrame, events: pd.DataFrame) -> dict:
    """C2: member-session overnight moves > 50% in split-adjusted prices that no verified event explains."""
    m = px.merge(fac, on=["session", "isin"])
    m = m.sort_values(["isin", "session"])
    m["adj_open"] = m.open * m.split_factor
    m["adj_prev_close"] = (m.close * m.split_factor).groupby(m["isin"]).shift(1)
    m["move"] = m.adj_open / m.adj_prev_close - 1
    big = m[m.member & m.valid_bar & (m.move.abs() > CONTINUITY)]
    return {"member_sessions_checked": int((m.member & m.valid_bar).sum()), "moves_over_50pct": int(len(big)),
            "examples": big[["session", "isin", "move"]].round(3).head(25).values.tolist(), "pass": len(big) == 0}


def spy_total_return(dv_raw: pd.DataFrame) -> pd.DataFrame:
    """SPY.P price and total-return index (dividends reinvested at the ex-date close), for benchmarks."""
    spy = pd.read_csv(SNAPSHOT["main"] / "SPY_P_prices.csv", index_col=0)
    spy = spy.apply(pd.to_numeric, errors="coerce").dropna(subset=["TRDPRC_1"])
    spy.index = spy.index.astype(str).str[:10]
    d = dv_raw[dv_raw.Instrument.eq("SPY.P")]
    div = d.groupby("Dividend Ex Date")["Gross Dividend Amount"].sum()
    close = spy["TRDPRC_1"]
    cash = pd.Series(0.0, index=close.index)
    for ex, amt in div.items():
        on = close.index[close.index >= ex]
        if len(on):
            cash[on[0]] += amt
    ret = (close + cash) / close.shift(1) - 1
    tr = (1 + ret.fillna(0)).cumprod()
    return pd.DataFrame({"session": close.index, "open": spy["OPEN_PRC"].values, "close": close.values,
                         "dividend": cash.values, "tr_index": tr.values})


def main() -> None:
    sm = pd.read_csv(OUTPUT / "security_master.csv", dtype=str)
    px = pd.read_parquet(OUTPUT / "prices_pit.parquet")
    px = px[px.valid_bar | px.member][["session", "isin", "open", "close", "valid_bar", "member"]]
    cc, dv, stats = load_actions()
    spy = spy_total_return(dv)
    spy.to_csv(OUTPUT / "spy_pit.csv", index=False, lineterminator="\n")
    cc = cc[cc["Corporate Change Event Type"].isin(RATIO_TYPES)]
    ev = to_isin(cc, sm, "Capital Change Effective Date")
    ev = ev[["Instrument", "Corporate Change Event Type", "Capital Change Effective Date", "Adjustment Factor", "isin"]]
    valid_px = px[px.valid_bar].sort_values(["isin", "session"])
    events = verify_ratio_events(ev, valid_px)
    # Hand-reviewed events (combined actions with cash, or price moves on the event day)
    rev = pd.read_csv(OVERRIDES / "corporate_action_reviews.csv")
    review_cash = []
    prim = sm[sm.role.eq("primary")]
    for r in rev.itertuples(index=False):
        if r.action == "add":  # action missing from LSEG entirely: add it with the reviewed factor
            own = prim[(prim.ric == r.ric) & (prim.valid_from <= r.trading_session) & (r.trading_session < prim.valid_to)]
            if own.empty:
                raise ValueError(f"added corporate action has no security: {r.ric} {r.trading_session}")
            events = pd.concat([events, pd.DataFrame([{
                "isin": own["isin"].iloc[0], "ric": r.ric, "type": "Added (missing from LSEG)", "lseg_effective": r.lseg_effective,
                "factor": float(r.factor_override), "trading_session": r.trading_session, "observed_ratio": np.nan,
                "status": "added_reviewed"}])], ignore_index=True)
            continue
        m = events.ric.eq(r.ric) & (pd.to_datetime(events.lseg_effective) - pd.Timestamp(r.lseg_effective)).abs().le(pd.Timedelta(days=3))
        if not m.any():
            raise ValueError(f"corporate action review matches no event: {r.ric} {r.lseg_effective}")
        events.loc[m, ["trading_session", "status"]] = [r.trading_session, "verified_reviewed"]
        if r.extra_cash_per_old_share > 0:
            review_cash.append({"isin": events.loc[m, "isin"].iloc[0], "ric": r.ric, "ex_date": r.trading_session,
                                "amount": float(r.extra_cash_per_old_share), "type": "Special (reviewed)"})
    dvi = to_isin(dv, sm, "Dividend Ex Date")
    divs = pd.DataFrame({"isin": dvi["isin"], "ric": dvi.Instrument, "ex_date": dvi["Dividend Ex Date"],
                         "amount": pd.to_numeric(dvi["Gross Dividend Amount"], errors="coerce"),
                         "type": dvi["Dividend Type"]}).dropna(subset=["amount"])
    divs = pd.concat([divs, pd.DataFrame(review_cash)], ignore_index=True)
    divs = divs[(divs.ex_date >= START) & (divs.ex_date <= END)].drop_duplicates(["isin", "ex_date", "amount", "type"])
    fac = build_factors(valid_px, events, divs)
    cont = continuity(px[px.valid_bar], fac, events)
    expl = pd.read_csv(OVERRIDES / "overnight_move_explanations.csv")
    isin_by_ric = {r.ric: r.isin for r in sm[sm.role.eq("primary")].itertuples(index=False)}
    explained = {(isin_by_ric.get(r.ric), r.session) for r in expl.itertuples(index=False)}
    cont["unexplained"] = [x for x in cont["examples"] if (x[1], x[0]) not in explained]
    cont["explained"] = len(cont["examples"]) - len(cont["unexplained"])
    cont["pass"] = len(cont["unexplained"]) == 0

    in_window = events[~events.status.eq("outside_window_or_no_prices")]
    split_like = in_window[in_window.type.isin(["Share Split", "Share Consolidation"])]
    audit = {**stats, "price_ratio_events_mapped": len(events), "by_type_status": events.groupby(["type", "status"]).size().to_dict(),
             "C1_split_consolidation_events_in_window": len(split_like),
             "C1_unverified": split_like[split_like.status.eq("unverified")][["ric", "lseg_effective", "factor", "observed_ratio"]].round(4).values.tolist(),
             "C1_pass": bool(split_like.status.isin(["verified", "verified_combined", "verified_reviewed", "no_price_effect", "no_price_effect_combined"]).all()),
             "C2": cont,
             "C3_dividends_used": len(divs), "C3_currency_assumption": "LSEG returned no currency; USD assumed for US listings",
             "C3_by_type": divs.type.value_counts().to_dict(),
             "C3_spy_dividends": int((spy.dividend > 0).sum()),
             "C3_spy_price_vs_total_return_2011_2024": [round(float(spy.loc[spy.session <= "2024-12-31", "close"].iloc[-1] / spy.close.iloc[0] - 1), 4),
                                                        round(float(spy.loc[spy.session <= "2024-12-31", "tr_index"].iloc[-1] - 1), 4)],
             "C3_pass": bool(len(divs) > 0 and (spy.dividend > 0).sum() >= 60)}
    audit["by_type_status"] = {f"{k[0]}|{k[1]}": int(v) for k, v in audit["by_type_status"].items()}
    events.to_csv(OUTPUT / "corporate_actions_pit.csv", index=False, lineterminator="\n")
    divs.to_csv(OUTPUT / "dividends_pit.csv", index=False, lineterminator="\n")
    fac.to_parquet(OUTPUT / "adjustment_factors_pit.parquet", index=False)
    (OUTPUT / "corporate_actions_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    write_manifest(OUTPUT / "corporate_actions_manifest.json",
                   [OUTPUT / "corporate_actions_pit.csv", OUTPUT / "dividends_pit.csv", OUTPUT / "adjustment_factors_pit.parquet"],
                   [*SNAPSHOT["actions"].glob("*.csv"), OUTPUT / "prices_pit.parquet", OUTPUT / "security_master.csv"],
                   {"script": "pit.corporate_actions"})
    print(json.dumps({k: audit[k] for k in ["by_type_status", "C1_split_consolidation_events_in_window", "C1_pass", "C3_dividends_used"]}, default=str))
    print("C1 unverified:", audit["C1_unverified"])
    print("C2:", {k: cont[k] for k in ["member_sessions_checked", "moves_over_50pct", "pass"]})
    for ex in cont["examples"][:25]:
        print("  ", ex)


if __name__ == "__main__":
    main()
