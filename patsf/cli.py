"""patsf: Patent Structure & Functionality.

  fetch        get claims and citations from Google Patents
  extract      extract the structure of each patent, coded from its claims by a model
  architecture architectural measures of the patent's own structure;
               writes results/<profile>/<patent>.architecture.csv
  compare      the patent's structure against its cited patents' structures;
               writes the codebook and the structures as CSV to results/<profile>/
  agree        F1 between codings (models, hand coding) of the same patents
  handcode     generates an Excel workbook for coding the patent by hand
  import       store a filled-in workbook's structure under the tag "human"

We can define which model to use for extract. Models are profiles in pyproject.toml ([tool.patsf]); 
--use picks one. For architecture and compare, --use also accepts the profile tags (e.g., "human" or "gpt-oss-20b"). 
Comparison settings are in [tool.patsf.compare].
Data is kept in patsf.db in the current folder.

Examples
  patsf --use qwen35-9b extract US11059478
  patsf --use qwen35-9b compare US11059478
  patsf agree US11059478 --tags gpt-oss-20b human
  patsf --use qwen35-9b architecture US11059478
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db


def _llm(args):
    from .llm import from_profile, load_profile
    name, prof = load_profile(args.use)
    return from_profile(prof), name


def _resolve(conn, raw):
    from .fetch import normalize_id
    rid = normalize_id(raw)
    return db.find_document(conn, rid) or rid


def _tag(conn, args):
    """The tag to read structures from: a model profile, or any tag in the database (e.g. human)."""
    from .llm import load_profile
    if args.use and args.use in db.tags(conn):
        return args.use
    return load_profile(args.use)[0]


def _family(conn, pid):
    """The patent and every cited patent that was fetched."""
    return [pid] + db.cited_ids(conn, pid)


def cmd_fetch(conn, args):
    from .fetch import fetch
    fetch(conn, args.patent)


def cmd_extract(conn, args, llm=None, tag=None):
    from .extract import extract
    if llm is None:
        llm, tag = _llm(args)
    pid = _resolve(conn, args.patent)
    if db.document(conn, pid) is None:
        sys.exit(f"{args.patent}: not fetched yet; run patsf fetch first")
    for t in _family(conn, pid):
        try:
            version, warnings = extract(conn, t, llm, tag, force=args.force)
        except Exception as exc:
            print(f"{t}: extraction failed: {exc}")
            continue
        print(f"{t} [{tag}]: structure v{version}" + (f", {len(warnings)} warnings" if warnings else ""))
        for w in warnings[:10]:
            print(f"   - {w}")


def cmd_compare(conn, args, tag=None):
    from .compare import compare, export, load_settings, summary
    from .report import export_structure
    tag = tag or _tag(conn, args)
    pid = _resolve(conn, args.patent)
    out = compare(conn, pid, tag, load_settings())
    if out is None:
        sys.exit(f"{pid} [{tag}]: no structure; run extract first")
    path = Path("results") / tag
    export(out, path, pid)
    for t in _family(conn, pid):
        if have := db.latest_structure(conn, t, tag):
            export_structure(have[1], t, path / "structures")
    print(f"{pid} [{tag}]: {summary(out)}\nwritten to {path}")

def cmd_architecture(conn, args):
    from .architecture import MEASURES, measures
    from .report import _write
    tag = _tag(conn, args)
    pid = _resolve(conn, args.patent)
    have = db.latest_structure(conn, pid, tag)
    if have is None:
        sys.exit(f"{pid} [{tag}]: no structure; run extract first")
    row = {"patent": pid, **measures(have[1])}
    path = Path("results") / tag
    path.mkdir(parents=True, exist_ok=True)
    _write(path / f"{pid}.architecture.csv", [row])
    print(f"{pid} [{tag}]: " + ", ".join(f"{k} {row[k]}" for k in MEASURES) + f"\nwritten to {path}")
    

def cmd_agree(conn, args):
    from .agree import agreement, format_report
    citing = [_resolve(conn, p) for p in args.patents]
    tag_list = args.tags or db.tags(conn)
    if len(tag_list) < 2:
        sys.exit("agree needs at least two model profiles; list them with --tags")
    rep = agreement(conn, citing, tag_list)
    print(format_report(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nfull report: {args.out}")


def cmd_handcode(conn, args):
    from .handcode import template
    pid = _resolve(conn, args.patent)
    print(f"{pid}: {template(conn, pid)}")


def cmd_import(conn, args):
    from .handcode import store
    pid, version, warnings = store(conn, Path(args.file), args.tag)
    print(f"{pid} [{args.tag}]: structure v{version}" + (f", {len(warnings)} warnings" if warnings else ""))
    for w in warnings[:10]:
        print(f"   - {w}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="patsf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--use", help="model profile from pyproject.toml (default: its `default`)")
    sub = p.add_subparsers(dest="cmd", required=True)
    patent = dict(help="patent number, e.g. US11059478 or 11,059,478")

    sp = sub.add_parser("fetch", help="(a) fetch the patent and all patents it cites")
    sp.add_argument("patent", **patent)

    sp = sub.add_parser("extract", help="(b) extract structures for the patent and its cited patents")
    sp.add_argument("patent", **patent)
    sp.add_argument("--force", action="store_true", help="extract again even if a structure is stored")

    sp = sub.add_parser("architecture", help="architecture measures of the patent's own structure")
    sp.add_argument("patent", **patent)

    sp = sub.add_parser("compare", help="(c) compare the patent with its cited patents, by fixed rules")
    sp.add_argument("patent", **patent)

    sp = sub.add_parser("agree", help="structural F1 between codings of the same patents")
    sp.add_argument("patents", nargs="+", help="citing patents to evaluate")
    sp.add_argument("--tags", nargs="+", help="model profiles to compare (default: all in the database)")
    sp.add_argument("--out", help="write the full report (JSON) here")

    sp = sub.add_parser("handcode", help="write an Excel workbook for coding the patent by hand")
    sp.add_argument("patent", **patent)

    sp = sub.add_parser("import", help="store a filled-in hand-coding workbook")
    sp.add_argument("file", help="the workbook, e.g. handcode/US11059478B2.xlsx")
    sp.add_argument("--tag", default="human")

    args = p.parse_args(argv)
    conn = db.connect("patsf.db")
    {"fetch": cmd_fetch, "extract": cmd_extract, "compare": cmd_compare, 
     "architecture": cmd_architecture, "agree": cmd_agree, "handcode": cmd_handcode, "import": cmd_import}[args.cmd](conn, args)


if __name__ == "__main__":
    sys.exit(main())
