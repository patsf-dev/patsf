"""Compare a patent with the patents it cites.

The structure of a focal invention is compared against the structures of the cited patents. 
The cited patents are first pooled, so that an element (component or functionality) exists if any of the cited patents has it.

  COMPONENT      component       NEW | EXISTING: EXISTING if a cited patent has a matching component
  FUNCTIONALITY  functionality   NEW | EXISTING: EXISTING if a cited patent states a matching
                                 functionality, whichever part performs it (function is compared
                                 apart from form)
  LINK           pair of components joined by a functionality's subject and target:
                 EXISTING  one cited patent joins components matching both subject and target, by any action
                 RECOMBINED  both components are EXISTING components, but no cited patent joins them
                 NEW  at least one component is a NEW component

Details:
- C0, the invention as a whole, is not a component: it is left out of the list of components, the links and the
component pairs. Only its functionalities (if any) are compared.
- Functionalities include both static ("surround a body") and dynamic ("rotate") actions. 
- A component is matched by its name, with its parent's name as context when the parent is not the
invention as a whole ("LiDAR emitter, part of sensor").

Matching uses a standard retrieve-then-rerank design. 
(1) A sentence embeddings from an open model (PatentSBERTa by default) first `shortlist` most similar
prior-art texts with cosine similarity of at least `shortlist_floor`. 
(2) A cross-encoder, an open model then reads the two texts together, scores each shortlisted pair.

A pair matches when that score reaches the threshold (recorded in [tool.patsf.compare] pyproject.toml). 
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from . import db

DEFAULTS = {
    "model": "AI-Growth-Lab/PatentSBERTa",
    "cross_encoder": "cross-encoder/stsb-roberta-base",
    "shortlist": 5,
    "shortlist_floor": 0.5,
    # Cross-encoder scores (0-1) at which a shortlisted pair matches. Provisional; calibrate on
    # hand-coded patents.
    "thresholds": {"component": 0.5, "functionality": 0.5},
}


# ---------------------------------------------------------------- settings

def settings_from(cfg: dict | None) -> dict:
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict):
            s.setdefault(k, {}).update(v)
        else:
            s[k] = v
    return s


def load_settings(path: str | Path = "pyproject.toml") -> dict:
    """[tool.patsf.compare] from pyproject.toml, over the defaults."""
    import tomllib
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))["tool"]["patsf"].get("compare")
    return settings_from(cfg)


# ---------------------------------------------------------------- similarity

_MODELS: dict = {}


def _load(kind: str, name: str):
    if (kind, name) not in _MODELS:
        import sentence_transformers as st
        _MODELS[(kind, name)] = st.SentenceTransformer(name) if kind == "bi" else st.CrossEncoder(name)
    return _MODELS[(kind, name)]


class Similarity:
    """Two-step match: embeddings shortlist, a cross-encoder scores the shortlisted pairs.
    `matrix` returns the cross-encoder score for shortlisted pairs and 0 elsewhere.
    Models are loaded once per process."""

    def __init__(self, settings: dict, encoder=None, cross=None):
        self.encoder = encoder or _load("bi", settings["model"])
        self.cross = cross or _load("cross", settings["cross_encoder"])
        self.k, self.floor = settings["shortlist"], settings["shortlist_floor"]
        self.name = f"{settings['model']} + {settings['cross_encoder']}"
        self._vec: dict[str, np.ndarray] = {}
        self._pair: dict[tuple[str, str], float] = {}

    def _vecs(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._vec]
        if new:
            for t, v in zip(new, self.encoder.encode(new, normalize_embeddings=True)):
                self._vec[t] = np.asarray(v, dtype=float)
        return np.vstack([self._vec[t] for t in texts])

    def matrix(self, a: list[str], b: list[str]) -> list[list[float]]:
        if not a or not b:
            return [[] for _ in a]
        cos = self._vecs(a) @ self._vecs(b).T
        wanted = []
        for i in range(len(a)):
            top = np.argsort(-cos[i])[: self.k]
            wanted += [(i, int(j)) for j in top if cos[i, j] >= self.floor]
        new = list(dict.fromkeys((a[i], b[j]) for i, j in wanted if (a[i], b[j]) not in self._pair))
        if new:
            for pair, score in zip(new, self.cross.predict(new)):
                self._pair[pair] = float(score)
        out = np.zeros((len(a), len(b)))
        for i, j in wanted:
            out[i, j] = self._pair[(a[i], b[j])]
        return out.tolist()


def similarity(settings: dict, encoder=None, cross=None) -> Similarity:
    return Similarity(settings, encoder, cross)


# ---------------------------------------------------------------- texts of elements

def _comp_text(c: dict, rep: dict | None = None) -> str:
    """A component's name, and its parent's name when the parent is not the invention as a whole."""
    if rep is not None and c.get("parent"):
        parent = next((p for p in rep["components"] if p["id"] == c["parent"]), None)
        if parent is not None and parent.get("parent"):
            return f"{c['name']}, part of {parent['name']}"
    return c["name"]


ROOT = "C0"   # the invention as a whole


def links(rep: dict) -> dict[frozenset, set[int]]:
    """Pairs of parts joined by a functionality's subject and target, with their claims. C0 is
    not a part, so pairs involving it are left out."""
    out: dict[frozenset, set[int]] = {}
    for f in rep.get("functionalities", []):
        t = f.get("target")
        if t and t != f["component"] and ROOT not in (t, f["component"]):
            out.setdefault(frozenset((f["component"], t)), set()).update(f["provenance"]["claims"])
    return out


def _best(row: list[float]) -> tuple[int, float]:
    """The best-scoring candidate, or (-1, 0.0) if nothing was shortlisted."""
    if not row:
        return -1, 0.0
    j = max(range(len(row)), key=row.__getitem__)
    return (j, row[j]) if row[j] > 0 else (-1, 0.0)


# ---------------------------------------------------------------- the comparison

def assess(citing: dict, cited: dict[str, dict], sim, settings: dict) -> dict:
    """citing: structure; cited: {patent_id: structure}."""
    th = settings["thresholds"]

    # ---- COMPONENT (the parts: every component but C0, here and in the cited patents)
    parts = [c for c in citing["components"] if c["id"] != ROOT]
    pool_c = [(pid, c, _comp_text(c, rep)) for pid, rep in cited.items() for c in rep["components"]
              if c["id"] != ROOT]
    cm = sim.matrix([_comp_text(c, citing) for c in parts], [t for _, _, t in pool_c])
    comp_rows, comp_hits, comp_matches = [], {}, {}
    for c, row in zip(parts, cm):
        matches = {(pool_c[j][0], pool_c[j][1]["id"]) for j, s in enumerate(row) if s >= th["component"]}
        hits = {pid for pid, _ in matches}
        comp_hits[c["id"]], comp_matches[c["id"]] = hits, matches
        j, s = _best(row)
        comp_rows.append({"component": c["id"], "name": c["name"],
                          "COMPONENT": "EXISTING" if hits else "NEW",
                          "closest": f"{pool_c[j][0]}:{pool_c[j][1]['id']} {pool_c[j][2]}" if j >= 0 else "",
                          "score": round(s, 3), "matched_patents": ";".join(sorted(hits))})

    # ---- FUNCTIONALITY: matched against every cited functionality, whichever part performs it.
    comps = {c["id"]: c for c in citing["components"]}
    pool_f = [(pid, f) for pid, rep in cited.items() for f in rep["functionalities"]]
    fm = sim.matrix([f["text"] for f in citing["functionalities"]], [f["text"] for _, f in pool_f])
    func_rows = []
    for f, row in zip(citing["functionalities"], fm):
        hits = {pool_f[j][0] for j, s in enumerate(row) if s >= th["functionality"]}
        j, score = _best(row)
        subject = comps.get(f["component"])
        func_rows.append({"functionality": f["id"],
                          "component": subject["name"] if subject else f["component"],
                          "text": f["text"],
                          "target": comps[f["target"]]["name"] if f.get("target") in comps else "",
                          "FUNCTIONALITY": "EXISTING" if hits else "NEW",
                          "closest": f"{pool_f[j][0]}:{pool_f[j][1]['id']} {pool_f[j][1]['text']}" if j >= 0 else "",
                          "score": round(score, 3), "matched_patents": ";".join(sorted(hits)),
                          "claims": ";".join(map(str, f["provenance"]["claims"]))})

    patent = {}

    # ---- per claim: how many of its functionalities are NEW and EXISTING (diagnostic)
    claim_rows = []
    verdict_of = {r["functionality"]: r["FUNCTIONALITY"] for r in func_rows}
    for k in sorted({k for f in citing["functionalities"] for k in f["provenance"]["claims"]}):
        fids = [f["id"] for f in citing["functionalities"] if k in f["provenance"]["claims"]]
        claim_rows.append({"claim": k, "functionalities": len(fids),
                           "NEW": sum(verdict_of[f] == "NEW" for f in fids),
                           "EXISTING": sum(verdict_of[f] == "EXISTING" for f in fids)})

    # ---- component recombination: pairs of EXISTING components that no single cited patent
    # holds together.
    existing = [c["id"] for c in parts if comp_hits.get(c["id"])]
    pairs = [(x, y) for i, x in enumerate(existing) for y in existing[i + 1:]]
    together = sum(bool(comp_hits[x] & comp_hits[y]) for x, y in pairs)
    patent.update({"component_pairs": len(pairs), "component_pairs_recombined": len(pairs) - together})

    # ---- LINK: EXISTING if one cited patent joins parts matching both ends; RECOMBINED if both
    # ends exist but no single cited patent joins them; NEW if an end is a NEW component.
    name = lambda x: comps[x]["name"] if x in comps else x
    linked = {pid: set(links(rep)) for pid, rep in cited.items()}
    link_rows = []
    for pair, claims in sorted(links(citing).items(), key=lambda kv: sorted(kv[0])):
        a, b = sorted(pair)
        ma, mb = comp_matches.get(a, set()), comp_matches.get(b, set())
        sources = sorted({pa for pa, xa in ma for pb, xb in mb
                          if pa == pb and xa != xb and frozenset((xa, xb)) in linked.get(pa, ())})
        verdict = "EXISTING" if sources else "RECOMBINED" if comp_hits.get(a) and comp_hits.get(b) else "NEW"
        link_rows.append({"link": f"{a}-{b}", "a": name(a), "b": name(b),
                          "LINK": verdict, "matched_patents": ";".join(sources),
                          "claims": ";".join(map(str, sorted(claims)))})

    out = {"patent": [patent], "components": comp_rows, "links": link_rows, "functionalities": func_rows,
           "claims": claim_rows}
    patent.update(constructs(out))
    return out


# Patent-level constructs: what the analysis uses, and what `agree` scores.
CONSTRUCTS = [
    "new_component_share",             # COMPONENT = NEW, share of the citing components
    "new_functionality_share",         # FUNCTIONALITY = NEW, share of the citing functionalities
    "link_new_component_share",        # LINK = NEW, share of the citing links
    "link_recombination_share",        # LINK = RECOMBINED, share of the citing links
    "component_recombination_share",   # pairs of existing parts never held by one cited patent
]

# Counts exported with the constructs. The totals (components, functionalities, links) are
# architecture measures and are not repeated here.
COUNTS = ["components_new", "functionalities_new", "links_new_component", "links_recombined",
          "component_pairs", "component_pairs_recombined"]


def constructs(out: dict) -> dict:
    """Counts behind the constructs, then the constructs (shares) computed from them. A share is
    None (undefined, not zero) when its denominator is 0, e.g. a patent with no links."""
    p = out["patent"][0]
    links_ = out.get("links", [])
    n = {
        "components_new": sum(r["COMPONENT"] == "NEW" for r in out["components"]),
        "functionalities_new": sum(r["FUNCTIONALITY"] == "NEW" for r in out["functionalities"]),
        "links_new_component": sum(r["LINK"] == "NEW" for r in links_),
        "links_recombined": sum(r["LINK"] == "RECOMBINED" for r in links_),
        "component_pairs": p.get("component_pairs", 0),
        "component_pairs_recombined": p.get("component_pairs_recombined", 0),
    }
    total = {"components": len(out["components"]), "functionalities": len(out["functionalities"]),
             "links": len(links_), "component_pairs": n["component_pairs"]}
    ratio = lambda num, den: round(n[num] / total[den], 3) if total[den] else None
    return {**n,
            "new_component_share": ratio("components_new", "components"),
            "new_functionality_share": ratio("functionalities_new", "functionalities"),
            "link_new_component_share": ratio("links_new_component", "links"),
            "link_recombination_share": ratio("links_recombined", "links"),
            "component_recombination_share": ratio("component_pairs_recombined", "component_pairs")}


def compare(conn, citing_id: str, tag: str, settings: dict, encoder=None, cross=None) -> dict | None:
    """Compare `citing_id` with the structures of its cited patents stored under `tag`."""
    have = db.latest_structure(conn, citing_id, tag)
    if have is None:
        return None
    cited = {c: s[1] for c in db.cited_ids(conn, citing_id) if (s := db.latest_structure(conn, c, tag))}
    sim = similarity(settings, encoder, cross)
    out = assess(have[1], cited, sim, settings)
    # How many cited patents the comparison covers: non-US and unreadable cited patents have no
    # structure, so their elements cannot count as EXISTING. Use as a control.
    out["patent"][0]["cited_compared"] = len(cited)
    out["settings"] = {"similarity": sim.name, "thresholds": settings["thresholds"],
                       "shortlist": settings["shortlist"], "shortlist_floor": settings["shortlist_floor"],
                       "cited_patents": sorted(cited)}
    return out


def summary(out: dict) -> str:
    s, p = out["settings"], out["patent"][0]
    count = lambda rows, key: dict(Counter(r[key] for r in rows if r[key]))
    return (f"models {s['similarity']}; {len(s['cited_patents'])} cited patents pooled\n"
            f"  COMPONENT: {count(out['components'], 'COMPONENT')}\n"
            f"  FUNCTIONALITY: {count(out['functionalities'], 'FUNCTIONALITY')}\n"
            f"  LINK: {count(out['links'], 'LINK')}\n"
            "  counts: " + ", ".join(f"{k} {p[k]}" for k in COUNTS) + "\n"
            "  constructs: " + ", ".join(f"{k} {p[k]}" for k in CONSTRUCTS))


def export(out: dict, path: Path, citing_id: str) -> None:
    from .report import _write
    path.mkdir(parents=True, exist_ok=True)
    for kind in ("patent", "components", "links", "functionalities", "claims"):
        _write(path / f"{citing_id}.compare.{kind}.csv", out[kind])
