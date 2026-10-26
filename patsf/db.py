"""SQLite storage shared by the pipelines.

`fetch` writes documents, claims and citations.
`extract` adds `structures`, each carrying a `tag` naming the model profile that produced
it, so several models can be compared on the same patents. 
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    patent_id        TEXT PRIMARY KEY,   -- publication number incl. kind code, e.g. US11059478B2
    title            TEXT,
    assignee         TEXT,
    abstract         TEXT,
    priority_date    TEXT,
    filing_date      TEXT,
    publication_date TEXT,
    claims_count     INTEGER,
    source_url       TEXT,
    fetched_at       TEXT,
    status           TEXT                -- ok | not_found | error:<msg>
);

CREATE TABLE IF NOT EXISTS claims (
    patent_id   TEXT,
    number      INTEGER,
    text        TEXT,
    parents     TEXT,                    -- JSON list of claim numbers this claim depends on
    independent INTEGER,
    claim_type  TEXT,                    -- apparatus | method | medium | composition | other
    canceled    INTEGER,
    PRIMARY KEY (patent_id, number)
);

CREATE TABLE IF NOT EXISTS citations (    -- the citing patent's "Patent Citations" list
    citing_id        TEXT,
    cited_id         TEXT,
    examiner_cited   INTEGER,            -- 1 if cited by the examiner, 0 if by the applicant
    priority_date    TEXT,
    publication_date TEXT,
    assignee         TEXT,
    title            TEXT,
    PRIMARY KEY (citing_id, cited_id)
);

CREATE TABLE IF NOT EXISTS structures (
    patent_id      TEXT,
    tag            TEXT,                 -- model profile that coded it (e.g. gpt-oss-20b)
    version        INTEGER,
    model          TEXT,
    prompt_version TEXT,
    created_at     TEXT,
    structure      TEXT,                 -- JSON, validated against schema.REPRESENTATION
    warnings       TEXT,                 -- JSON list of validation warnings
    PRIMARY KEY (patent_id, tag, version)
);
"""


SCHEMA_VERSION = 2   # raise whenever SCHEMA changes in a way an existing database cannot absorb


def connect(path: str | Path = "patsf.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    found = conn.execute("PRAGMA user_version").fetchone()[0]
    has_tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    if has_tables and found != SCHEMA_VERSION:
        conn.close()
        raise SystemExit(f"{path} was made by an older version of patsf. Delete it and run fetch again; "
                         "cached pages in html_cache/ are reused.")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def document(conn: sqlite3.Connection, patent_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE patent_id=?", (patent_id,)).fetchone()


def claims(conn: sqlite3.Connection, patent_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE patent_id=? ORDER BY number", (patent_id,)
    ).fetchall()
    return [{**dict(r), "parents": json.loads(r["parents"])} for r in rows]


def find_document(conn: sqlite3.Connection, patent_id: str) -> str | None:
    """The stored publication number for an id given with or without its kind code
    (US11059478 finds US11059478B2), or None if the patent is not stored."""
    row = conn.execute("SELECT patent_id FROM documents WHERE status='ok' AND (patent_id=? OR patent_id LIKE ?)",
                       (patent_id, patent_id + "%")).fetchone()
    return row[0] if row else None


def cited_ids(conn: sqlite3.Connection, patent_id: str) -> list[str]:
    """Every US patent on this patent's citation list that was fetched successfully (non-US cited
    patents are left out of the analysis, including any fetched before that rule)."""
    rows = conn.execute(
        "SELECT DISTINCT d.patent_id FROM citations c JOIN documents d "
        "ON d.status='ok' AND (d.patent_id=c.cited_id OR d.patent_id LIKE c.cited_id || '%') "
        "WHERE c.citing_id=? ORDER BY d.patent_id", (patent_id,)).fetchall()
    return [r[0] for r in rows if r[0].startswith("US")]


def latest_structure(conn: sqlite3.Connection, patent_id: str, tag: str) -> tuple[int, dict] | None:
    row = conn.execute(
        "SELECT version, structure FROM structures WHERE patent_id=? AND tag=? ORDER BY version DESC LIMIT 1",
        (patent_id, tag),
    ).fetchone()
    return (row["version"], json.loads(row["structure"])) if row else None


def tags(conn: sqlite3.Connection) -> list[str]:
    """Every tag with at least one stored structure."""
    return [r[0] for r in conn.execute("SELECT DISTINCT tag FROM structures ORDER BY tag")]
