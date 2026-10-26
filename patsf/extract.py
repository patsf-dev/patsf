"""extract: transforms a patent's textual claims into the structured representation.

Input to the LLM: a patent's first independent claim and all other claims that are dependent on it.
Results are stored under a tag naming the model.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

from . import db, prompts
from .llm import LLM
from .schema import REPRESENTATION, check_representation


def format_claims(claims: list[dict]) -> str:
    lines = []
    for c in claims:
        dep = "independent" if c["independent"] else "depends on " + ", ".join(map(str, c["parents"]))
        lines.append(f"[{c['number']}] ({dep}; {c['claim_type']})\n{c['text']}")
    return "\n\n".join(lines)


def merge_duplicates(rep: dict) -> dict:
    """Merge functionalities with the same component, text (ignoring case and spacing) and
    target into one entry, keeping the first entry's id and quote and combining the claims."""
    def merged(items, key):
        kept, seen = [], {}
        for x in items:
            k = key(x)
            if k in seen:
                first = seen[k]
                first["provenance"]["claims"] = sorted(set(first["provenance"]["claims"]) | set(x["provenance"]["claims"]))
                continue
            seen[k] = x
            kept.append(x)
        return kept
    return {**rep, "functionalities": merged(
        rep.get("functionalities", []),
        lambda f: (f["component"], " ".join(f["text"].lower().split()), f.get("target")))}


def resolve_components(rep: dict) -> dict:
    """References that name a component ("controller") instead of giving its id ("C3") are
    pointed at the component with that name (ignoring case): a functionality's component and
    target, and a component's parent."""
    ids = {c["id"] for c in rep.get("components", [])}
    by_name = {}
    for c in rep.get("components", []):
        by_name[" ".join(c.get("name", "").lower().split())] = c["id"]
    fix = lambda ref: ref if ref is None or ref in ids else by_name.get(" ".join(str(ref).lower().split()), ref)
    for f in rep.get("functionalities", []):
        f["component"] = fix(f["component"])
        if "target" in f:
            f["target"] = fix(f["target"])
    for c in rep.get("components", []):
        if "parent" in c:
            c["parent"] = fix(c["parent"])
    return rep


def _store(conn: sqlite3.Connection, patent_id: str, tag: str, model: str, prompt_version: str,
           rep: dict, warnings: list[str]) -> int:
    version = (conn.execute("SELECT MAX(version) FROM structures WHERE patent_id=? AND tag=?",
                            (patent_id, tag)).fetchone()[0] or 0) + 1
    with conn:
        conn.execute(
            "INSERT INTO structures VALUES (?,?,?,?,?,?,?,?)",
            (patent_id, tag, version, model, prompt_version,
             dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             json.dumps(rep), json.dumps(warnings)),
        )
    return version


def _require_claims(conn: sqlite3.Connection, patent_id: str) -> tuple:
    doc = db.document(conn, patent_id)
    claims = db.claims(conn, patent_id)
    if doc is None or doc["status"] != "ok" or not claims:
        raise LookupError(f"{patent_id}: no stored claims; run fetch first")
    return doc, claims


def extract(conn: sqlite3.Connection, patent_id: str, llm: LLM, tag: str, force: bool = False,
            max_repairs: int = 1) -> tuple[int, list[str]]:
    """Extract and store a new structure version under `tag`. Returns (version, warnings)."""
    # A stored structure is reused, unless it is empty.
    if not force and (have := db.latest_structure(conn, patent_id, tag)) and have[1].get("components"):
        row = conn.execute("SELECT warnings FROM structures WHERE patent_id=? AND tag=? AND version=?",
                           (patent_id, tag, have[0])).fetchone()
        return have[0], json.loads(row["warnings"])
    doc, claims = _require_claims(conn, patent_id)

    if hasattr(llm, "represent"):          # rule-based baseline: no prompt, no repair round
        rep = llm.represent(doc, claims)
        warnings = check_representation(rep, claims)
        return _store(conn, patent_id, tag, llm.name, llm.version, rep, warnings), warnings

    user = prompts.EXTRACT_USER.format(patent_id=patent_id, title=doc["title"] or "",
                                       claims=format_claims(claims))
    # An empty structure is a failed run (seen with gpt-oss on 2 of 51 patents): ask again.
    for _ in range(3):
        rep = merge_duplicates(resolve_components(llm.json(prompts.EXTRACT_SYSTEM, user, REPRESENTATION, "submit_representation")))
        if rep.get("components"):
            break
    else:
        raise RuntimeError("the model returned an empty structure three times; nothing stored")
    warnings = check_representation(rep, claims)

    # One repair round for integrity problems (schema problems are retried inside the client).
    for _ in range(max_repairs):
        if not warnings:
            break
        repair = (user + "\n\nYour previous representation:\n" + json.dumps(rep)
                  + "\n\nIt has these problems. Return a corrected, complete representation:\n- "
                  + "\n- ".join(warnings[:40]))
        rep2 = merge_duplicates(resolve_components(llm.json(prompts.EXTRACT_SYSTEM, repair, REPRESENTATION, "submit_representation")))
        w2 = check_representation(rep2, claims)
        if rep2.get("components") and len(w2) <= len(warnings):
            rep, warnings = rep2, w2

    return _store(conn, patent_id, tag, llm.name, prompts.EXTRACT_VERSION, rep, warnings), warnings
