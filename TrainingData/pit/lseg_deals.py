"""Download LSEG M&A deal records and retirement dates for securities that stopped trading.

Run with the Python that has `lseg.data` (system Python) while LSEG Workspace is signed in:
    python TrainingData/pit/lseg_deals.py [--targets indicators_data/pit/settlement_targets.csv]

Writes indicators_data/lseg/deals/<UTC stamp>/{deals.csv, retire.csv, manifest.json}.
Raw data only; selection and validation happen in pit.settlement.
The app key is read from HKCU\\Environment or the process environment and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TRAINING_DATA = Path(__file__).resolve().parents[1]
DEAL_FIELDS = ["TR.MnADealId", "TR.MnATarget", "TR.MnATargetPermId", "TR.MnAAcquiror", "TR.MnAAnnDate",
               "TR.MnAEffectiveDate", "TR.MnAStatus", "TR.MnAConsiderationStructure", "TR.MnAOfferPricePerShare",
               "TR.MnADealValue", "TR.MnASynopsis"]
RETIRE_FIELDS = ["TR.RetireDate", "TR.InstrumentIsActive", "TR.OrganizationID", "TR.CommonName"]


def open_session(ld):
    keys = []
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
                keys.append(winreg.QueryValueEx(reg, "LSEG_APP_KEY")[0])
        except OSError:
            pass
    keys.append(os.environ.get("LSEG_APP_KEY"))
    for key in dict.fromkeys(k.strip() for k in keys if k and k.strip()):
        s = ld.session.desktop.Definition(app_key=key).get_session()
        if s.open().name == "Opened":
            ld.session.set_default(s)
            return s
        s.close()
    sys.exit("Workspace session could not be opened (is LSEG Workspace running and signed in?)")


def fetch(ld, rics: list[str], fields: list[str], batch: int) -> tuple[pd.DataFrame, list[dict]]:
    frames, errors = [], []
    for i in range(0, len(rics), batch):
        chunk = rics[i:i + batch]
        try:
            frames.append(ld.get_data(chunk, fields))
        except Exception as e:  # retry one by one so a single bad RIC does not lose the batch
            for ric in chunk:
                try:
                    frames.append(ld.get_data(ric, fields))
                except Exception as e2:
                    errors.append({"ric": ric, "error": str(e2)[:200]})
        print(f"  {min(i + batch, len(rics))}/{len(rics)}", flush=True)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=TRAINING_DATA / "indicators_data/pit/settlement_targets.csv")
    ap.add_argument("--batch", type=int, default=10)
    a = ap.parse_args()
    import lseg.data as ld
    logging.disable(logging.CRITICAL)
    rics = pd.read_csv(a.targets).ric.dropna().unique().tolist()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = TRAINING_DATA / "indicators_data/lseg/deals" / stamp
    out.mkdir(parents=True)
    s = open_session(ld)
    try:
        print("retire dates"); retire, e1 = fetch(ld, rics, RETIRE_FIELDS, 50)
        print("deals"); deals, e2 = fetch(ld, rics, DEAL_FIELDS, a.batch)
    finally:
        s.close()
    retire.to_csv(out / "retire.csv", index=False, lineterminator="\n")
    deals.to_csv(out / "deals.csv", index=False, lineterminator="\n")
    man = {"retrieved_at_utc": stamp, "provider": "LSEG Workspace desktop", "sdk_version": ld.__version__,
           "targets_file": str(a.targets), "rics": len(rics), "deal_fields": DEAL_FIELDS, "retire_fields": RETIRE_FIELDS,
           "errors": e1 + e2, "files": {f: {"rows": int(len(df)), "sha256": hashlib.sha256((out / f).read_bytes()).hexdigest()}
                                         for f, df in (("deals.csv", deals), ("retire.csv", retire))}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({"folder": str(out), "rics": len(rics), "deal_rows": len(deals), "retire_rows": len(retire),
                      "errors": len(man["errors"])}))


if __name__ == "__main__":
    main()
