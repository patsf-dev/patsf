"""Agreement between codings of the same patents, based on F1 scores.
For every pair of tags (e.g., from two models, or a model and the hand coding): do they decompose a patent the same way? 
"""

from __future__ import annotations

import itertools
import re
import sqlite3

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import db
from .compare import links

STOP = set("a an the of to and or in on for by with at from as is are be being been that this "
           "which wherein whereby said least one more each any via".split())

# ---------------------------------------------------------------- structural F1

def _tokens(*texts: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", " ".join(t for t in texts if t).lower())
    return {w for w in words if w not in STOP}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _text(kind: str, el: dict) -> set[str]:
    if kind == "components":
        return _tokens(el["name"])
    return _tokens(el["text"])


def match(kind: str, a: list[dict], b: list[dict], threshold: float = 0.35) -> list[tuple[str, str, float]]:
    """One-to-one matching on claim overlap and wording that maximizes the total score
    (scipy's linear_sum_assignment). Pairs below `threshold` are not matched.
    TO DO: examine sensitivity to threshold and weight parameters. """
    if not a or not b:
        return []
    name = lambda el: " ".join(el["name"].lower().split()) if kind == "components" else None
    A = [(set(x["provenance"]["claims"]), _text(kind, x), name(x)) for x in a]
    B = [(set(y["provenance"]["claims"]), _text(kind, y), name(y)) for y in b]
    score = np.zeros((len(a), len(b)))
    for i, (cx, tx, nx) in enumerate(A):
        for j, (cy, ty, ny) in enumerate(B):
            same_name = nx is not None and nx == ny
            if cx & cy or same_name:
                s_ = 0.4 * _jaccard(cx, cy) + 0.6 * _jaccard(tx, ty) + (0.3 if same_name else 0.0)
                score[i, j] = s_ if s_ >= threshold else 0.0
    rows, cols = linear_sum_assignment(score, maximize=True)
    return [(a[i]["id"], b[j]["id"], round(float(score[i, j]), 3)) for i, j in zip(rows, cols) if score[i, j] > 0]


KINDS = ["components", "links", "functionalities"]


def _link_id(pair) -> str:
    return "-".join(sorted(pair))


def match_links(rep_a: dict, rep_b: dict, component_pairs=None) -> tuple[list[tuple[str, str, float]], int, int]:
    """Links among the components both codings found. Returns (matched pairs, eligible links in
    A, eligible links in B): a link is eligible when both of its ends are paired components (see
    `match`), and matches when the other coding joins the same pair. `component_pairs` is the
    component matching, if already computed."""
    if component_pairs is None:
        component_pairs = match("components", rep_a["components"], rep_b["components"])
    pairs = {ia: ib for ia, ib, _ in component_pairs}
    paired_b = set(pairs.values())
    eligible_a = [p for p in links(rep_a) if all(x in pairs for x in p)]
    eligible_b = [p for p in links(rep_b) if all(x in paired_b for x in p)]
    theirs = set(eligible_b)
    out = []
    for pair in eligible_a:
        mapped = frozenset(pairs[x] for x in pair)
        if mapped in theirs:
            theirs.discard(mapped)
            out.append((_link_id(pair), _link_id(mapped), 1.0))
    return out, len(eligible_a), len(eligible_b)


def structural_f1(rep_a: dict, rep_b: dict) -> dict:
    scores, comp_pairs = {}, None
    for kind in KINDS:
        if kind == "links":
            pairs, na, nb = match_links(rep_a, rep_b, comp_pairs)
        else:
            na, nb = len(rep_a.get(kind, [])), len(rep_b.get(kind, []))
            pairs = match(kind, rep_a[kind], rep_b[kind])
            if kind == "components":
                comp_pairs = pairs
        if na + nb:
            scores[kind] = round(2 * len(pairs) / (na + nb), 3)
    scores["macro"] = round(sum(scores.values()) / len(scores), 3) if scores else None
    return scores


def structure_agreement(conn: sqlite3.Connection, patent_ids: list[str], tag_list: list[str]) -> dict:
    """Pairwise structural F1, averaged over patents coded under both tags."""
    out = {}
    for a, b in itertools.combinations(tag_list, 2):
        per_patent = {}
        for pid in patent_ids:
            ra, rb = db.latest_structure(conn, pid, a), db.latest_structure(conn, pid, b)
            if ra and rb:
                per_patent[pid] = structural_f1(ra[1], rb[1])
        if not per_patent:
            continue
        mean = {}
        for k in KINDS + ["macro"]:
            vals = [p[k] for p in per_patent.values() if p.get(k) is not None]
            mean[k] = round(sum(vals) / len(vals), 3) if vals else None
        out[f"{a} vs {b}"] = {"mean": mean, "patents": per_patent}
    return out


def agreement(conn: sqlite3.Connection, citing_ids: list[str], tag_list: list[str],
              include_cited: bool = True) -> dict:
    """Structural F1 for every pair of tags, on the patents and (by default) their cited patents."""
    patents = list(citing_ids)
    if include_cited:
        for cid in citing_ids:
            patents += [c for c in db.cited_ids(conn, cid) if c not in patents]
    return {"tags": tag_list, "structural_f1": structure_agreement(conn, patents, tag_list)}


def format_report(rep: dict) -> str:
    lines = [f"Models: {', '.join(rep['tags'])}", "", "Structural F1 (mean over patents)"]
    for pair, v in rep["structural_f1"].items():
        m = v["mean"]
        lines.append(f"  {pair}: macro {m['macro']}  (" + ", ".join(f"{k} {m[k]}" for k in KINDS)
                     + f"; {len(v['patents'])} patent(s))")
    if not rep["structural_f1"]:
        lines.append("  no patent is coded under two of these profiles")
    return "\n".join(lines)
