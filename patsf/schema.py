"""JSON Schema for the representation, and checks on extracted structures.
"""

from __future__ import annotations

import re

from jsonschema import Draft202012Validator

_str = {"type": "string"}

PROVENANCE = {
    "type": "object",
    "description": "Where the element comes from. `quote` is copied verbatim from the claim text.",
    "properties": {
        "claims": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "quote": _str,
        "status": {"enum": ["stated", "inferred"]},
    },
    "required": ["claims", "quote", "status"],
    "additionalProperties": False,
}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


REPRESENTATION = _obj({
    "components": {"type": "array", "items": _obj({
        "id": _str, "name": _str, "parent": {"type": ["string", "null"]}, "provenance": PROVENANCE,
    }, ["id", "name", "parent", "provenance"])},

    "functionalities": {"type": "array", "items": _obj({
        "id": _str, "component": _str, "text": _str, "target": {"type": ["string", "null"]},
        "provenance": PROVENANCE,
    }, ["id", "component", "text", "target", "provenance"])},
}, ["components", "functionalities"])


# ---------------------------------------------------------------- semantic checks

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def schema_errors(obj: dict, schema: dict) -> list[str]:
    """Violations of a JSON Schema, as 'path: message' strings (empty if valid)."""
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(obj)]


def check_representation(rep: dict, claims: list[dict]) -> list[str]:
    """Schema validation, references, coverage of independent claims and quote grounding.

    Returns a list of warnings; schema errors are returned with the prefix 'schema:'.
    """
    if errors := [f"schema: {e}" for e in schema_errors(rep, REPRESENTATION)]:
        return errors
    warn: list[str] = []
    seen: set[str] = set()
    for kind in ["components", "functionalities"]:
        for el in rep[kind]:
            if el["id"] in seen:
                warn.append(f"duplicate id {el['id']}")
            seen.add(el["id"])
    comps = {c["id"] for c in rep["components"]}
    for fn in rep["functionalities"]:
        if fn["component"] not in comps:
            warn.append(f"functionality {fn['id']}: unknown component {fn['component']}")
    for c in rep["components"]:
        if c["parent"] is not None and c["parent"] not in comps:
            warn.append(f"component {c['id']}: unknown parent {c['parent']}")
    for fn in rep["functionalities"]:
        if fn["target"] is not None and fn["target"] not in comps:
            warn.append(f"functionality {fn['id']}: unknown target {fn['target']}")
        elif fn["target"] is not None and fn["target"] == fn["component"]:
            warn.append(f"functionality {fn['id']}: targets its own component")

    # Quote grounding, and coverage: every independent claim must be cited by some element.
    claim_text = {c["number"]: _norm(c["text"]) for c in claims}
    all_text = " ".join(claim_text.values())
    cited_claims: set[int] = set()
    for kind in ["components", "functionalities"]:
        for el in rep[kind]:
            prov = el["provenance"]
            cited_claims.update(prov["claims"])
            bad = [n for n in prov["claims"] if n not in claim_text]
            if bad:
                warn.append(f"{el['id']}: cites unknown claims {bad}")
            q = _norm(prov["quote"])
            if prov["status"] == "stated" and q and q not in all_text:
                warn.append(f"{el['id']}: quote not found in claims: {prov['quote'][:80]!r}")
    independent = {c["number"] for c in claims if c["independent"]}
    missing = sorted(independent - cited_claims)
    if missing:
        warn.append(f"independent claims not covered by any element: {missing}")
    return warn



if __name__ == "__main__":      # python -m patsf.schema prints the schema, e.g. for an appendix
    import json
    print(json.dumps(REPRESENTATION, indent=1))
