"""USED ONLY FOR GENERATING A BASELINE
Rule-based baseline: subject-action-object (SAO) extraction, a standard rule-based way to
extract function from text (see Zheng et al. 2026). 
Each claim is parsed with spaCy. For every verb with a subject and a direct object:
  component      the subject noun and the object noun (compound nouns + head noun, lemmatised)
  functionality  "<verb lemma> a <object>", performed by the subject
"""

from __future__ import annotations

import re

RULES_VERSION = "sao-2026-10-19"

RELATIVE = {"that", "which", "who"}


def _phrase(tok) -> str:
    comps = [c.lemma_.lower() for c in tok.lefts if c.dep_ == "compound"]
    return " ".join(comps + [tok.lemma_.lower()])


class RuleExtractor:
    """Used in place of a model client: `extract` calls `represent` instead of prompting."""

    version = RULES_VERSION

    def __init__(self, spacy_model: str = "en_core_web_trf"):
        import spacy
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        self.name = f"sao:{spacy_model}-{self.nlp.meta['version']}"

    @staticmethod
    def _subject(verb, depth=0):
        subj = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subj is not None and subj.lower_ in RELATIVE and verb.dep_ == "relcl":
            return verb.head
        if subj is None and verb.dep_ in ("conj", "xcomp") and depth < 5:
            return RuleExtractor._subject(verb.head, depth + 1)
        if subj is None and verb.dep_ in ("acl", "relcl") and verb.head.pos_ in ("NOUN", "PROPN"):
            return verb.head            # "a sensor configured to emit a signal"
        return subj

    def represent(self, doc: dict, claims: list[dict]) -> dict:
        comps: dict[str, dict] = {}
        funcs: dict[tuple, dict] = {}

        def component(tok, claim):
            key = _phrase(tok)
            if key not in comps:
                comps[key] = {"id": f"C{len(comps)}", "name": key, "parent": None,
                              "provenance": {"claims": set(), "quote": tok.text, "status": "stated"}}
            comps[key]["provenance"]["claims"].add(claim)
            return comps[key]

        for c in claims:
            sdoc = self.nlp(re.sub(r"^\s*\d+\s*[.)]\s*", "", c["text"]))
            if not comps:     # C0: the first noun of the first claim
                first = next((t for t in sdoc if t.pos_ in ("NOUN", "PROPN") and t.dep_ != "compound"), None)
                if first is not None:
                    component(first, c["number"])
            for verb in sdoc:
                if verb.pos_ != "VERB":
                    continue
                obj = next((x for x in verb.children if x.dep_ in ("dobj", "obj")), None)
                subj = self._subject(verb)
                if obj is None or subj is None or subj.pos_ not in ("NOUN", "PROPN") \
                        or obj.pos_ not in ("NOUN", "PROPN"):
                    continue
                actor = component(subj, c["number"])
                target = component(obj, c["number"])
                o = _phrase(obj)
                text = f"{verb.lemma_.lower()} {'an' if o[:1] in 'aeiou' else 'a'} {o}"
                key = (actor["id"], text)
                if key not in funcs:
                    span = sdoc[verb.i:obj.i + 1].text if obj.i > verb.i else verb.text
                    funcs[key] = {"component": actor["id"], "text": text,
                                  "target": target["id"] if target is not actor else None,
                                  "provenance": {"claims": set(), "quote": " ".join(span.split()[:25]),
                                                 "status": "stated"}}
                funcs[key]["provenance"]["claims"].add(c["number"])
        if comps:   # the invention as a whole is claimed by every independent claim
            next(iter(comps.values()))["provenance"]["claims"].update(c["number"] for c in claims if c["independent"])

        def prov(p):
            return {**p, "claims": sorted(p["claims"])}
        return {"components": [{**x, "provenance": prov(x["provenance"])} for x in comps.values()],
                "functionalities": [{"id": f"FN{i + 1}", **f, "provenance": prov(f["provenance"])}
                                    for i, f in enumerate(funcs.values())]}
