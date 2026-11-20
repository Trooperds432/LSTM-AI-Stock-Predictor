"""sha256 manifests so every derived artifact records exactly which inputs produced it."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_path: Path, outputs: list[Path], inputs: list[Path], meta: dict | None = None) -> dict:
    """Write <out_path> (json) listing output and input files with sizes and hashes.
    The timestamp is kept out of the hashed outputs so rebuilds stay byte-identical."""
    def rec(p: Path) -> dict:
        p = Path(p)
        return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}

    m = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "outputs": [rec(p) for p in outputs], "inputs": [rec(p) for p in sorted(map(Path, inputs))],
         "meta": meta or {}}
    Path(out_path).write_text(json.dumps(m, indent=2))
    return m
