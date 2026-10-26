"""fetch a patent and every patent it cites from Google Patents.

For each patent, stores its claims (with a claim tree), its citations and dates.
Raw HTML is cached so that parsing can be re-run without re-fetching.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from . import db

BASE = "https://patents.google.com/patent/{id}/en"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; patsf)"}


# ---------------------------------------------------------------- identifiers

def normalize_id(raw: str) -> str:
    """'11,059,478' -> 'US11059478'; 'US 2018/0284279 A1' -> 'US20180284279A1'."""
    s = re.sub(r"[\s,/\-]", "", raw.strip().upper())
    if s and s[0].isdigit():
        s = "US" + s
    return s


# ---------------------------------------------------------------- fetching

def fetch_html(patent_id: str, cache_dir: Path | None = None, retries: int = 4,
               pause: float = 1.0) -> str | None:
    """Return the page HTML, or None if Google Patents has no page for this id."""
    if cache_dir:
        cached = cache_dir / f"{patent_id}.html.gz"
        if cached.exists():
            return gzip.decompress(cached.read_bytes()).decode("utf-8")
    url = BASE.format(id=patent_id)
    for attempt in range(retries):
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(pause * 2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        time.sleep(pause)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{patent_id}.html.gz").write_bytes(gzip.compress(resp.content))
        return resp.text
    raise RuntimeError(f"giving up on {patent_id} after {retries} attempts")


# ---------------------------------------------------------------- parsing

def _text(el: Tag | None) -> str:
    if el is None:
        return ""
    return re.sub(r"[ \t]+", " ", el.get_text(" ", strip=False)).strip()


def _own_text(node: Tag) -> str:
    """Text of a claim-text div, excluding nested claim-text divs but keeping inline tags."""
    parts = []
    for child in node.children:
        if isinstance(child, Tag):
            if child.name == "div" and "claim-text" in (child.get("class") or []):
                continue
            parts.append(child.get_text())
        else:
            parts.append(str(child))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _claim_text(el: Tag) -> str:
    # Each claim-text div is one claim element (preamble, then each limitation); one per line.
    lines = [_own_text(n) for n in el.select("div.claim-text")]
    text = "\n".join(l for l in lines if l) or _text(el)
    return re.sub(r"[ \t]+([,;:.])", r"\1", text)


DEP_RE = re.compile(
    r"\b(?:of|in|according to|as (?:claimed|recited|defined) in|per)\s+"
    r"(?:any (?:one )?of\s+)?claims?\s+(\d+)(?:\s*(?:-|to|through|or|and)\s*(\d+))?",
    re.I,
)


def _parents(el: Tag, text: str) -> list[int]:
    refs = [r.get("idref", "") for r in el.select("claim-ref")]
    nums = sorted({int(m.group(1)) for r in refs if (m := re.search(r"(\d+)$", r))})
    if nums:
        return nums
    m = DEP_RE.search(text[:300])
    if not m:
        return []
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(lo, hi + 1)) if hi >= lo and hi - lo < 50 else [lo]


def _claim_type(text: str) -> str:
    head = text[:200].lower()
    if re.search(r"non-transitory|computer[- ]readable|machine[- ]readable|storage medi|program product", head):
        return "medium"
    if re.search(r"\b(a|an)\s+([\w-]+\s+){0,4}(method|process)\b", head):
        return "method"
    if re.search(r"\b(composition|compound|formulation|alloy|gel|solution|mixture)\b", head[:80]):
        return "composition"
    return "apparatus"


def parse_claims(soup: BeautifulSoup) -> list[dict]:
    section = soup.select_one('section[itemprop="claims"]')
    if section is None:
        return []
    out = []
    for el in section.select("div.claim[num]"):
        text = _claim_text(el)
        canceled = bool(re.match(r"\s*[\d\s\-–.]+\(?\s*(canceled|cancelled|deleted)\b", text, re.I))
        # "1." or a range such as "33-36. (canceled)"; fall back to the num attribute.
        m = re.match(r"\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*[.)]", text)
        if m:
            first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        else:
            nums = [int(x) for x in re.findall(r"\d+", el.get("num", ""))] or [len(out) + 1]
            first, last = nums[0], nums[-1]
        if canceled and last > first:
            out += [{"number": n, "text": f"{n}. (canceled)", "parents": [], "canceled": True}
                    for n in range(first, last + 1)]
            continue
        out.append({"number": first, "text": text, "parents": _parents(el, text),
                    "canceled": canceled})
    # Claim type: independent claims from their preamble, dependents inherit from their root.
    by_num = {c["number"]: c for c in out}
    for c in out:
        c["independent"] = not c["parents"]
    def root_type(c, seen=()):
        if c["independent"] or c["number"] in seen:
            return _claim_type(re.sub(r"^\s*\d+\s*[.)]\s*", "", c["text"]))
        parent = by_num.get(c["parents"][0])
        return root_type(parent, seen + (c["number"],)) if parent else "other"
    for c in out:
        c["claim_type"] = "other" if c["canceled"] else root_type(c)
    return out


def parse_citations(soup: BeautifulSoup) -> list[dict]:
    """The "Patent Citations" table: every patent cited by this one, by examiner or applicant."""
    out = []
    for tr in soup.select('tr[itemprop="backwardReferences"]'):
        num = tr.select_one('[itemprop="publicationNumber"]')
        if num is None:
            continue
        out.append({
            "cited_id": _text(num),
            "examiner_cited": tr.select_one('[itemprop="examinerCited"]') is not None,
            "priority_date": _text(tr.select_one('[itemprop="priorityDate"]')) or None,
            "publication_date": _text(tr.select_one('[itemprop="publicationDate"]')) or None,
            "assignee": _text(tr.select_one('[itemprop="assigneeOriginal"]')) or None,
            "title": _text(tr.select_one('[itemprop="title"]')) or None,
        })
    return out


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    meta = lambda name: (soup.find("meta", attrs={"name": name}) or {}).get("content", "").strip() or None
    time_ = lambda prop: _text(soup.select_one(f'time[itemprop="{prop}"]')) or None
    return {
        "patent_id": _text(soup.select_one('dd[itemprop="publicationNumber"]')) or None,
        "title": meta("DC.title"),
        "assignee": _text(soup.select_one('dd[itemprop="assigneeOriginal"]')) or None,
        "abstract": _text(soup.select_one('section[itemprop="abstract"] div.abstract'))
                    or _text(soup.select_one('section[itemprop="abstract"]')) or None,
        "priority_date": time_("priorityDate"),
        "filing_date": time_("filingDate"),
        "publication_date": time_("publicationDate"),
        "claims": parse_claims(soup),
        "citations": parse_citations(soup),
    }


# ---------------------------------------------------------------- storing

def first_claim_family(claims: list[dict]) -> list[dict]:
    """The first independent claim and the claims that depend on it, directly or through other
    claims. Other independent claims (a method, a system, a part claimed on its own) and their
    dependents are left out, as are canceled claims."""
    live = [c for c in claims if not c["canceled"]]
    first = next((c["number"] for c in live if c["independent"]), None)
    if first is None:
        return []
    by_num = {c["number"]: c for c in live}

    def root(c, seen=()):
        if c["independent"] or c["number"] in seen:
            return c["number"]
        parent = by_num.get(c["parents"][0])
        return root(parent, seen + (c["number"],)) if parent else None

    return [c for c in live if root(c) == first]


def store(conn: sqlite3.Connection, page: dict, source_url: str) -> str:
    """Store a parsed page. Only the first independent claim and its dependents are kept: every
    patent, citing or cited, is coded from the same kind of text, its first claim set."""
    pid = page["patent_id"]
    all_claims = sum(not c["canceled"] for c in page["claims"])     # kept as a control (claim count)
    page = {**page, "claims": first_claim_family(page["claims"])}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with conn:
        for table, key in [("claims", "patent_id"), ("citations", "citing_id")]:
            conn.execute(f"DELETE FROM {table} WHERE {key}=?", (pid,))
        conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pid, page["title"], page["assignee"], page["abstract"], page["priority_date"],
             page["filing_date"], page["publication_date"],
             all_claims, source_url, now, "ok"),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?)",
            [(pid, c["number"], c["text"], json.dumps(c["parents"]), int(c["independent"]),
              c["claim_type"], int(c["canceled"])) for c in page["claims"]],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?,?,?)",
            [(pid, c["cited_id"], int(c["examiner_cited"]), c["priority_date"],
              c["publication_date"], c["assignee"], c["title"]) for c in page["citations"]],
        )
    return pid


def _fetch_one(conn: sqlite3.Connection, raw_id: str, cache_dir: Path | None) -> str | None:
    """Fetch and store one patent. Returns the canonical publication number, or None."""
    requested = normalize_id(raw_id)
    if stored := db.find_document(conn, requested):
        return stored
    html = fetch_html(requested, cache_dir)
    if html is None:
        with conn:
            conn.execute("INSERT OR REPLACE INTO documents (patent_id, status, fetched_at) VALUES (?,?,?)",
                         (requested, "not_found", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
        return None
    page = parse_page(html)
    page["patent_id"] = page["patent_id"] or requested
    return store(conn, page, BASE.format(id=requested))


def fetch(conn: sqlite3.Connection, raw_id: str, cache_dir: Path | None = Path("html_cache"), log=print) -> str | None:
    """Fetch a patent and every US patent it cites. Returns the patent's publication number, or None."""
    pid = _fetch_one(conn, raw_id, cache_dir)
    if pid is None:
        log(f"{raw_id}: not found")
        return None
    log(f"{pid}: stored")
    cited = [r[0] for r in conn.execute("SELECT cited_id FROM citations WHERE citing_id=?", (pid,))]
    # Only US cited patents are fetched: claims of other offices are machine translations or missing.
    # They stay on the citation list, so `compare` can report how much prior art was left out.
    foreign = [c for c in cited if not normalize_id(c).startswith("US")]
    if foreign:
        log(f"  skipped {len(foreign)} non-US cited patents: {', '.join(foreign[:8])}{' ...' if len(foreign) > 8 else ''}")
    for cid in [c for c in cited if c not in foreign]:
        try:
            canon = _fetch_one(conn, cid, cache_dir)
        except Exception as exc:  # keep going; record the failure
            with conn:
                conn.execute("INSERT OR REPLACE INTO documents (patent_id, status) VALUES (?,?)",
                             (cid, f"error:{exc}"[:200]))
            log(f"  {cid}: error {exc}")
            continue
        log(f"  {cid}: {'stored' if canon else 'not found'}")
    return pid
