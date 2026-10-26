"""Export stored structures as CSV tables and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _claims(el: dict) -> str:
    return ";".join(map(str, el["provenance"]["claims"]))


def export_structure(rep: dict, patent_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{patent_id}.structure.json").write_text(json.dumps(rep, indent=2))
    _write(out / f"{patent_id}.components.csv", [
        {"id": c["id"], "name": c["name"], "parent": c.get("parent") or "",
         "claims": _claims(c), "status": c["provenance"]["status"]} for c in rep["components"]])
    _write(out / f"{patent_id}.functionalities.csv", [
        {"id": f["id"], "component": f["component"], "text": f["text"], "target": f.get("target") or "",
         "claims": _claims(f), "status": f["provenance"]["status"]} for f in rep["functionalities"]])
