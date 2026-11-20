"""Daily point-in-time S&P 500 membership by ISIN, replayed from LSEG checkpoints and change events.

Port of TrainingData/lseg_membership_replay.py with every hard-coded rule moved into
pit/overrides/*.csv. The algorithm is unchanged: reverse-replay events from the earliest
non-empty checkpoint to the opening session, then forward-replay over the XNYS calendar.

Usage: python -m pit.membership   (run from TrainingData/)
"""
from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as xc
import pandas as pd

from . import overrides as ov
from .config import END, OUTPUT, SNAPSHOT, START
from .manifest import write_manifest


def reference_identities(snapshot: Path) -> dict[str, str]:
    ref = pd.concat([pd.read_csv(f) for f in sorted(snapshot.glob("identifiers_*.csv"))])
    return ref.set_index("Instrument").ISIN.to_dict()


def _resolve(isin: str, ids: dict[str, str]) -> str:
    return ids[isin[4:]] if isin.startswith("ric:") else isin


def map_roster(rows: pd.DataFrame, date: str, ids: dict[str, str], names: pd.DataFrame,
               subs: pd.DataFrame, dedup: pd.DataFrame) -> set[str]:
    """Map one dated LSEG constituent response (Constituent RIC, Constituent Name) to ISINs."""
    rows = rows.dropna(subset=["Constituent RIC"])
    for d in dedup.itertuples(index=False):
        if date < d.before_date and {d.keep_ric, d.drop_ric} <= set(rows["Constituent RIC"]):
            if ids[d.keep_ric] != ids[d.drop_ric]:
                raise ValueError(f"alias dedup {d.keep_ric}/{d.drop_ric} are different securities")
            rows = rows.loc[rows["Constituent RIC"].ne(d.drop_ric)]
    name_map = {(r.constituent_ric, r.constituent_name): r.instrument for r in names.itertuples(index=False)}
    members = [ids[name_map.get((ric, name), ric)] for ric, name in zip(rows["Constituent RIC"], rows["Constituent Name"])]
    for s in subs[subs.applies_to.eq("checkpoints")].itertuples(index=False):
        if s.valid_from <= date < s.valid_to:
            members = [s.to_isin if m == s.from_isin else m for m in members]
    if len(set(members)) != len(members):
        raise ValueError(f"unresolved identity collision on {date}")
    return set(members)


def load_events(snapshot: Path, ids: dict[str, str]) -> dict[str, tuple[set, set]]:
    changes = pd.read_csv(snapshot / "membership_changes.csv")
    changes["ISIN"] = changes["Constituent RIC"].map(ids)
    for o in ov.load("membership_change_isin_overrides").itertuples(index=False):
        m = changes["Constituent RIC"].eq(o.constituent_ric) & changes.Date.eq(o.date)
        if not m.any():
            raise ValueError(f"change override matches no row: {o}")
        changes.loc[m, "ISIN"] = o.isin
    extra = ov.load("extra_membership_events")
    extra = pd.DataFrame({"Date": extra["date"], "ISIN": [_resolve(i, ids) for i in extra["isin"]], "Change": extra["change"]})
    allc = pd.concat([changes[["Date", "ISIN", "Change"]], extra], ignore_index=True)
    if allc.ISIN.isna().any():
        raise ValueError(f"change events without ISIN: {allc[allc.ISIN.isna()].head().to_dict('records')}")
    events = {}
    for date, g in allc.groupby("Date"):
        joins, leaves = set(g.loc[g.Change.eq("Joiner"), "ISIN"]), set(g.loc[g.Change.eq("Leaver"), "ISIN"])
        events[date] = (joins - leaves, leaves - joins)
    return events


def check_probe_assertions(snapshot: Path) -> list[dict]:
    out = []
    for a in ov.load("membership_probe_assertions").itertuples(index=False):
        probe = pd.read_csv(snapshot / "boundary_probes" / f"{a.date}.csv")
        present = bool(probe["Constituent RIC"].eq(a.ric).any())
        ok = len(probe) >= int(a.min_rows) and present == (a.expected_member == "True")
        out.append({"date": a.date, "ric": a.ric, "expected": a.expected_member, "present": present, "ok": ok})
        if not ok:
            raise ValueError(f"probe assertion failed: {out[-1]}")
    return out


def replay(snapshot: Path = SNAPSHOT["main"]) -> tuple[pd.DataFrame, dict]:
    ids = reference_identities(snapshot)
    names, subs, dedup = ov.load("membership_name_overrides"), ov.load("membership_isin_substitutions"), ov.load("membership_alias_dedup")
    checkpoints = {}
    for path in sorted(snapshot.glob("members_*.csv")):
        rows = pd.read_csv(path)
        if rows.dropna(subset=["Constituent RIC"]).empty:
            continue
        date = path.stem[8:]
        checkpoints[date] = map_roster(rows, date, ids, names, subs, dedup)
    events = load_events(snapshot, ids)
    probes = check_probe_assertions(snapshot)

    anchor = min(checkpoints)
    members = set(checkpoints[anchor])
    reverse_conflicts = []
    for date, (joins, leaves) in reversed(list(events.items())):
        if date <= anchor:
            if joins - members or leaves & members:
                reverse_conflicts.append(date)
            members = (members - joins) | leaves
    opening = set(members)

    sessions = xc.get_calendar("XNYS", start=START, end=END).sessions.strftime("%Y-%m-%d")
    unknown = sorted(set(events) - set(sessions))
    if unknown:
        raise ValueError(f"events on non-sessions: {unknown[:5]}")
    rows, conflicts, residuals, counts = [], [], {}, []
    for date in sessions:
        joins, leaves = events.get(date, (set(), set()))
        if leaves - members or joins & members:
            conflicts.append({"date": date, "missing_leavers": sorted(leaves - members), "existing_joiners": sorted(joins & members)})
        members = (members - leaves) | joins
        if date in checkpoints:
            residuals[date] = {"extra": sorted(members - checkpoints[date]), "missing": sorted(checkpoints[date] - members)}
        counts.append(len(members))
        rows.extend((date, isin) for isin in sorted(members))
    panel = pd.DataFrame(rows, columns=["session", "isin"])
    audit = {"anchor_checkpoint": anchor, "opening_count": len(opening), "opening_independently_verified": False,
             "rows": len(panel), "sessions": len(sessions), "min_members": min(counts), "max_members": max(counts),
             "checkpoints": residuals, "forward_conflicts": conflicts, "reverse_conflicts": reverse_conflicts,
             "probe_assertions": probes, "pending_sources": ov.pending_sources()}
    return panel, audit


def compare_boundary_probes(panel: pd.DataFrame, snapshot: Path = SNAPSHOT["main"]) -> dict:
    """M2: compare the replay with every non-empty dated boundary probe."""
    ids = reference_identities(snapshot)
    # Expired instruments recovered in later runs (Micro Focus, Bemis) carry their own reference rows
    for extra in (SNAPSHOT["micro_focus"], SNAPSHOT["bemis"]):
        ref = pd.read_csv(extra / "reference_snapshot.csv")
        ids.update({k: v for k, v in ref.set_index("Instrument").ISIN.items() if k not in ids})
    names, subs, dedup = ov.load("membership_name_overrides"), ov.load("membership_isin_substitutions"), ov.load("membership_alias_dedup")
    by_session = panel.groupby("session")["isin"].apply(set)
    expl = ov.load("membership_probe_explanations")
    out = {}
    for path in sorted((snapshot / "boundary_probes").glob("*.csv")):
        date = path.stem
        rows = pd.read_csv(path)
        if "Constituent RIC" not in rows or rows.dropna(subset=["Constituent RIC"]).empty or date not in by_session:
            continue
        try:
            probe = map_roster(rows, date, ids, names, subs, dedup)
        except (KeyError, ValueError) as e:
            out[date] = {"error": repr(e)}
            continue
        rep = by_session[date]
        extra, missing = sorted(rep - probe), sorted(probe - rep)
        explained = [(e.replay_isin, e.probe_isin) for e in expl.itertuples(index=False) if e.date == date]
        ex_r, ex_p = {a for a, _ in explained}, {b for _, b in explained}
        out[date] = {"probe_size": len(probe), "extra_in_replay": extra, "missing_from_replay": missing,
                     "unexplained_extra": [i for i in extra if i not in ex_r],
                     "unexplained_missing": [i for i in missing if i not in ex_p]}
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, audit = replay()
    audit["boundary_probes"] = compare_boundary_probes(panel)
    out = OUTPUT / "membership_pit.csv"
    panel.to_csv(out, index=False, lineterminator="\n")
    (OUTPUT / "membership_pit_audit.json").write_text(json.dumps(audit, indent=2))
    inputs = [*SNAPSHOT["main"].glob("identifiers_*.csv"), *SNAPSHOT["main"].glob("members_*.csv"),
              SNAPSHOT["main"] / "membership_changes.csv", *Path(ov.OVERRIDES).glob("*.csv")]
    write_manifest(OUTPUT / "membership_pit_manifest.json", [out], inputs, {"script": "pit.membership"})
    print(json.dumps({k: audit[k] for k in ["anchor_checkpoint", "opening_count", "rows", "min_members", "max_members",
                                            "reverse_conflicts"]}))
    print("checkpoints:", {d: (len(v["extra"]), len(v["missing"])) for d, v in audit["checkpoints"].items()})
    print("forward conflicts:", len(audit["forward_conflicts"]), "| pending sources:", len(audit["pending_sources"]))
    print("probe diffs:", {d: (len(v.get("extra_in_replay", [])), len(v.get("missing_from_replay", []))) if "error" not in v else v["error"][:60]
                           for d, v in audit["boundary_probes"].items()})


if __name__ == "__main__":
    main()
