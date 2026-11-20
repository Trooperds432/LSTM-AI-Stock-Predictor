"""Small LSEG reference/price pulls for settlement validation (run with the Python that has lseg.data).

  python TrainingData/pit/lseg_reference_extra.py names            -> company names for all security-master RICs
  python TrainingData/pit/lseg_reference_extra.py bars RIC [RIC..]  -> daily OPEN/CLOSE bars 2011-01-01..2026-11-18

Writes indicators_data/lseg/reference_extra/<UTC stamp>/{names.csv|bars_*.csv, manifest.json}.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lseg_deals import open_session  # noqa: E402

TRAINING_DATA = Path(__file__).resolve().parents[1]


def main() -> None:
    mode, args = sys.argv[1], sys.argv[2:]
    import lseg.data as ld
    logging.disable(logging.CRITICAL)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = TRAINING_DATA / "indicators_data/lseg/reference_extra" / stamp
    out.mkdir(parents=True)
    s = open_session(ld)
    files, errors = [], []
    try:
        if mode == "names":
            rics = pd.read_csv(TRAINING_DATA / "indicators_data/pit/security_master.csv").ric.unique().tolist()
            frames = []
            for i in range(0, len(rics), 100):
                try:
                    frames.append(ld.get_data(rics[i:i + 100], ["TR.CommonName", "TR.ISIN", "TR.OrganizationID"]))
                except Exception as e:
                    errors.append({"batch": i, "error": str(e)[:200]})
            pd.concat(frames).to_csv(out / "names.csv", index=False, lineterminator="\n")
            files.append("names.csv")
        elif mode == "bars":
            for ric in args:
                try:
                    b = ld.get_history(ric, fields=["OPEN_PRC", "TRDPRC_1", "ACVOL_UNS"], start="2011-01-01",
                                       end="2026-11-18", interval="daily", adjustments="unadjusted")
                    name = f"bars_{ric.replace('.', '_').replace('^', '_')}.csv"
                    b.to_csv(out / name, lineterminator="\n")
                    files.append(name)
                except Exception as e:
                    errors.append({"ric": ric, "error": str(e)[:200]})
        else:
            sys.exit("mode must be names or bars")
    finally:
        s.close()
    man = {"retrieved_at_utc": stamp, "mode": mode, "args": args, "errors": errors,
           "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({"folder": str(out), "files": len(files), "errors": errors}))


if __name__ == "__main__":
    main()
