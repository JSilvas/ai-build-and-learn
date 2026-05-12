"""
Step 1: Obsidian Vault Explorer (read-only)

Walks the vault, parses notes, and prints a structural report showing
what the Neo4j graph would look like — without touching the database.

Usage:
    uv run python vault_explorer.py
    uv run python vault_explorer.py --cypher     # also print sample Cypher
    uv run python vault_explorer.py --vault /path/to/other/vault
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

VAULT_PATH = Path(
    "/Users/jaysilvas/Library/CloudStorage/OneDrive-Personal"
    "/Synthetic Knowledge/LLM Knowledge WIki"
)

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_/-]+)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_note(path: Path, vault: Path = VAULT_PATH) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")

    frontmatter = {}
    body = raw
    m = FRONTMATTER_RE.match(raw)
    if m:
        try:
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            pass
        body = raw[m.end():]

    wikilinks = [t.strip() for t in WIKILINK_RE.findall(body)]
    inline_tags = HASHTAG_RE.findall(body)

    fm_tags = frontmatter.get("tags", [])
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    all_tags = list(dict.fromkeys(fm_tags + inline_tags))  # dedupe, order-stable

    return {
        "title": path.stem,
        "path": str(path.relative_to(vault)),
        "frontmatter": frontmatter,
        "wikilinks": wikilinks,
        "tags": all_tags,
        "word_count": len(body.split()),
        "body_preview": body.strip()[:120].replace("\n", " "),
    }


def walk_vault(vault: Path) -> list[dict]:
    notes = []
    for md in sorted(vault.rglob("*.md")):
        if ".obsidian" in md.parts:
            continue
        notes.append(parse_note(md, vault))
    return notes


# ── reporting ─────────────────────────────────────────────────────────────────

def report(notes: list[dict], show_cypher: bool, vault: Path = VAULT_PATH) -> None:
    titles = {n["title"] for n in notes}

    # link graph
    links: dict[str, list[str]] = defaultdict(list)
    dangling: dict[str, list[str]] = defaultdict(list)  # links to non-existent notes
    for n in notes:
        for target in n["wikilinks"]:
            if target in titles:
                links[n["title"]].append(target)
            else:
                dangling[n["title"]].append(target)

    # tag index
    tag_index: dict[str, list[str]] = defaultdict(list)
    for n in notes:
        for t in n["tags"]:
            tag_index[t].append(n["title"])

    # orphans (no inbound or outbound links)
    linked = set(links.keys()) | {t for targets in links.values() for t in targets}
    orphans = [n["title"] for n in notes if n["title"] not in linked]

    # ── print ──
    sep = "─" * 60

    print(f"\n{'OBSIDIAN VAULT EXPLORER':^60}")
    print(f"{'Vault: ' + str(vault):^60}")
    print(sep)

    print(f"\n  Notes found  : {len(notes)}")
    print(f"  Link edges   : {sum(len(v) for v in links.values())}")
    print(f"  Unique tags  : {len(tag_index)}")
    print(f"  Orphan notes : {len(orphans)}")

    print(f"\n{sep}")
    print("NOTES")
    print(sep)
    for n in notes:
        print(f"\n  [{n['title']}]  ({n['word_count']} words)")
        print(f"    path     : {n['path']}")
        if n["tags"]:
            print(f"    tags     : {', '.join('#' + t for t in n['tags'])}")
        if n["wikilinks"]:
            print(f"    links to : {', '.join(n['wikilinks'])}")
        if n["body_preview"]:
            print(f"    preview  : {n['body_preview']!r}")

    if links:
        print(f"\n{sep}")
        print("LINK GRAPH (resolved wikilinks)")
        print(sep)
        for src, targets in sorted(links.items()):
            for tgt in targets:
                print(f"  {src}  →  {tgt}")

    if dangling:
        print(f"\n{sep}")
        print("DANGLING LINKS (target note not in vault)")
        print(sep)
        for src, targets in sorted(dangling.items()):
            for tgt in targets:
                print(f"  {src}  →  [{tgt}]  (missing)")

    if tag_index:
        print(f"\n{sep}")
        print("TAGS")
        print(sep)
        for tag, note_titles in sorted(tag_index.items()):
            print(f"  #{tag:<20} {', '.join(note_titles)}")

    if orphans:
        print(f"\n{sep}")
        print("ORPHAN NOTES (no links in or out)")
        print(sep)
        for o in orphans:
            print(f"  {o}")

    if show_cypher:
        print(f"\n{sep}")
        print("SAMPLE CYPHER (what Step 4 would execute — not run now)")
        print(sep)
        print()
        print("  // Create Note nodes")
        for n in notes[:3]:
            escaped = n["title"].replace("'", "\\'")
            print(f"  MERGE (:Note {{title: '{escaped}', path: '{n['path']}'}});")
        if len(notes) > 3:
            print(f"  // ... and {len(notes) - 3} more")

        print()
        print("  // Create LINKS_TO edges")
        for src, targets in list(links.items())[:3]:
            for tgt in targets:
                print(f"  MATCH (a:Note {{title: '{src}'}}), (b:Note {{title: '{tgt}'}})")
                print(f"  MERGE (a)-[:LINKS_TO]->(b);")

        if tag_index:
            print()
            print("  // Create Tag nodes and HAS_TAG edges")
            for tag, note_titles in list(tag_index.items())[:3]:
                print(f"  MERGE (:Tag {{name: '{tag}'}});")
                print(f"  MATCH (n:Note {{title: '{note_titles[0]}'}}), (t:Tag {{name: '{tag}'}})")
                print(f"  MERGE (n)-[:HAS_TAG]->(t);")

    print(f"\n{sep}")
    print("Next step: run with --cypher to see Cypher preview, then proceed to Step 2.")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default=str(VAULT_PATH))
    parser.add_argument("--cypher", action="store_true", help="Print sample Cypher")
    args = parser.parse_args()

    vault = Path(args.vault)
    if not vault.exists():
        print(f"ERROR: vault not found at {vault}", file=sys.stderr)
        sys.exit(1)

    notes = walk_vault(vault)
    if not notes:
        print("No .md files found in vault. Check the path or wait for OneDrive to sync.")
        sys.exit(0)

    report(notes, show_cypher=args.cypher, vault=vault)


if __name__ == "__main__":
    main()
