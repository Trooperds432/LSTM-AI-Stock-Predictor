"""Effective-dated security master: which RIC carries which security (ISIN) on which sessions.

Rows: isin, ric, valid_from (inclusive), valid_to (exclusive), role (primary|alias), source.
Built from the LSEG reference/identifier files plus pit/overrides/{ric_isin_dated,listing_transfers}.csv.

Usage: python -m pit.identity   (run from TrainingData/)
"""
from __future__ import annotations

import json

import pandas as pd

from . import overrides as ov
from .config import OUTPUT, SNAPSHOT
from .manifest import write_manifest

OPEN_FROM, OPEN_TO = "2011-01-03", "2099-12-31"


def reference_rows() -> pd.DataFrame:
    """Current LSEG reference: one row per instrument RIC with an ISIN."""
    files = sorted(SNAPSHOT["main"].glob("identifiers_*.csv"))
    files += [SNAPSHOT[k] / "reference_snapshot.csv" for k in ("micro_focus", "bemis", "successors", "tyco")]
    ref = pd.concat([pd.read_csv(f).assign(src_file=f.name) for f in files], ignore_index=True)
    ref = ref.dropna(subset=["ISIN"]).drop_duplicates(["Instrument", "ISIN"])
    if ref.Instrument.duplicated().any():
        raise ValueError(f"instrument with two ISINs: {ref[ref.Instrument.duplicated(keep=False)].to_dict('records')}")
    return ref


def build_security_master() -> pd.DataFrame:
    ref = reference_rows()
    dated = ov.load("ric_isin_dated")
    transfers = ov.load("listing_transfers")
    alias_rics = set(transfers.alias_ric)
    rows = []
    for r in ref.itertuples(index=False):
        if r.Instrument in set(dated.ric):
            continue  # replaced by dated rows below
        role = "alias" if r.Instrument in alias_rics else "primary"
        rows.append((r.ISIN, r.Instrument, OPEN_FROM, OPEN_TO, role, f"lseg_reference:{r.src_file}"))
    for d in dated.itertuples(index=False):
        rows.append((d.isin, d.ric, d.valid_from, d.valid_to, "primary", d.source))
    sm = pd.DataFrame(rows, columns=["isin", "ric", "valid_from", "valid_to", "role", "source"])
    for t in transfers.itertuples(index=False):
        on_day = sm.ric.isin([t.primary_ric, t.alias_ric]) & (sm.valid_from <= t.transfer_session) & (t.transfer_session < sm.valid_to)
        got = set(sm.loc[on_day, "isin"])
        if got != {t.isin}:
            raise ValueError(f"listing transfer {t.isin}: RICs map to {got}")
    return sm.sort_values(["isin", "valid_from", "role", "ric"]).reset_index(drop=True)


def validate(sm: pd.DataFrame) -> dict:
    """I1: no security has overlapping primary RICs; no RIC maps to two securities at once."""
    def overlaps(g: pd.DataFrame) -> list:
        g = g.sort_values("valid_from")
        return [(a.ric, b.ric, b.valid_from) for a, b in zip(g.iloc[:-1].itertuples(), g.iloc[1:].itertuples())
                if b.valid_from < a.valid_to]
    primary_overlaps = {isin: o for isin, g in sm[sm.role.eq("primary")].groupby("isin") if (o := overlaps(g))}
    ric_overlaps = {}
    for ric, g in sm.groupby("ric"):
        g = g.sort_values("valid_from")
        bad = [(a.isin, b.isin) for a, b in zip(g.iloc[:-1].itertuples(), g.iloc[1:].itertuples()) if b.valid_from < a.valid_to]
        if bad:
            ric_overlaps[ric] = bad
    return {"rows": len(sm), "securities": int(sm["isin"].nunique()), "rics": int(sm.ric.nunique()),
            "aliases": sm[sm.role.eq("alias")][["isin", "ric"]].values.tolist(),
            "primary_overlaps": primary_overlaps, "ric_overlaps": ric_overlaps,
            "pass": not primary_overlaps and not ric_overlaps}


def primary_ric_on(sm: pd.DataFrame, isin: str, session: str) -> str | None:
    m = sm[(sm["isin"] == isin) & sm.role.eq("primary") & (sm.valid_from <= session) & (session < sm.valid_to)]
    return m.ric.iloc[0] if len(m) else None


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sm = build_security_master()
    audit = validate(sm)
    out = OUTPUT / "security_master.csv"
    sm.to_csv(out, index=False, lineterminator="\n")
    (OUTPUT / "security_master_audit.json").write_text(json.dumps(audit, indent=2))
    write_manifest(OUTPUT / "security_master_manifest.json", [out],
                   [*SNAPSHOT["main"].glob("identifiers_*.csv"), *ov.OVERRIDES.glob("*.csv")], {"script": "pit.identity"})
    print(json.dumps({k: audit[k] for k in ["rows", "securities", "rics", "aliases", "pass"]}))


if __name__ == "__main__":
    main()
