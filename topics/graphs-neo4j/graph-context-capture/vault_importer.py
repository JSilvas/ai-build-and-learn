"""
Step 3 / Step 4: Obsidian Vault Importer

Step 3 usage (dry-run — no Neo4j writes):
    uv run python vault_importer.py --dry-run

Step 4 usage (live import):
    uv run python vault_importer.py

Options:
    --vault PATH    Override vault location
    --dry-run       Print what would be imported; touch nothing
    --no-embed      Skip embedding generation (faster dry-run inspection)
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────

VAULT_PATH = Path(
    "/Users/jaysilvas/Library/CloudStorage/OneDrive-Personal"
    "/Synthetic Knowledge/LLM Knowledge WIki"
)

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")

EMBED_MODEL = "all-MiniLM-L6-v2"   # must match app.py
EMBED_DIM   = 384

# cosine similarity floor for embedding-based bridges (Step 5)
BRIDGE_THRESHOLD = 0.65

# ── regex ─────────────────────────────────────────────────────────────────────

WIKILINK_RE   = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
HASHTAG_RE    = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_/-]+)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class ParsedNote:
    id: str                          # stable sha256 of vault-relative path
    title: str
    path: str                        # vault-relative
    text: str                        # body only (no frontmatter)
    tags: list[str]
    wikilinks: list[str]             # raw target titles
    frontmatter: dict
    word_count: int
    embedding: list[float] = field(default_factory=list)

    @property
    def node_labels(self) -> str:
        return "Note:Vault"


# ── parsing ───────────────────────────────────────────────────────────────────

def _stable_id(vault_relative_path: str) -> str:
    return hashlib.sha256(vault_relative_path.encode()).hexdigest()[:16]


def parse_note(path: Path, vault: Path) -> ParsedNote:
    raw = path.read_text(encoding="utf-8", errors="replace")

    frontmatter: dict = {}
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
    all_tags = list(dict.fromkeys(fm_tags + inline_tags))

    rel_path = str(path.relative_to(vault))

    return ParsedNote(
        id=_stable_id(rel_path),
        title=path.stem,
        path=rel_path,
        text=body.strip(),
        tags=all_tags,
        wikilinks=wikilinks,
        frontmatter=frontmatter,
        word_count=len(body.split()),
    )


def walk_vault(vault: Path) -> list[ParsedNote]:
    notes = []
    for md in sorted(vault.rglob("*.md")):
        if ".obsidian" in md.parts:
            continue
        notes.append(parse_note(md, vault))
    return notes


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_notes(notes: list[ParsedNote], model: SentenceTransformer) -> None:
    """Attach embeddings to each note in-place. Uses title + body for richer signal."""
    texts = [f"{n.title}\n{n.text}" for n in notes]
    print(f"  Embedding {len(texts)} note(s)…", flush=True)
    vecs = model.encode(texts, show_progress_bar=True, batch_size=32)
    for note, vec in zip(notes, vecs):
        note.embedding = vec.tolist()


# ── dry-run report ────────────────────────────────────────────────────────────

def dry_run_report(notes: list[ParsedNote]) -> None:
    title_map = {n.title: n for n in notes}
    all_tags = sorted({t for n in notes for t in n.tags})
    sep = "─" * 64

    print(f"\n{'DRY-RUN IMPORT PLAN':^64}")
    print(sep)
    print(f"  Notes to import  : {len(notes)}")
    print(f"  Unique tags      : {len(all_tags)}")

    link_count = sum(
        1 for n in notes for t in n.wikilinks if t in title_map
    )
    dangling_count = sum(
        1 for n in notes for t in n.wikilinks if t not in title_map
    )
    print(f"  LINKS_TO edges   : {link_count}  ({dangling_count} dangling — skipped)")
    has_embed = sum(1 for n in notes if n.embedding)
    print(f"  Notes with embed : {has_embed}")

    print(f"\n{sep}")
    print("NODES THAT WOULD BE CREATED / MERGED")
    print(sep)
    for n in notes:
        embed_status = f"embed:{EMBED_DIM}d" if n.embedding else "no embed"
        print(f"\n  MERGE (:{n.node_labels} {{id: '{n.id}'}})")
        print(f"    title  : {n.title!r}")
        print(f"    path   : {n.path}")
        print(f"    words  : {n.word_count}  |  {embed_status}")
        if n.tags:
            print(f"    tags   : {', '.join('#' + t for t in n.tags)}")
        if n.wikilinks:
            resolved = [t for t in n.wikilinks if t in title_map]
            unresolved = [t for t in n.wikilinks if t not in title_map]
            if resolved:
                print(f"    links  : {', '.join(resolved)}")
            if unresolved:
                print(f"    skip   : {', '.join(unresolved)}  (not in vault)")

    if all_tags:
        print(f"\n{sep}")
        print("TAG NODES THAT WOULD BE CREATED / MERGED")
        print(sep)
        for t in all_tags:
            users = [n.title for n in notes if t in n.tags]
            print(f"  MERGE (:Tag:Vault {{name: '{t}'}})  ← used by: {', '.join(users)}")

    print(f"\n{sep}")
    print("CYPHER PREVIEW  (first 5 notes)")
    print(sep)
    _print_cypher_preview(notes[:5], title_map)

    print(f"\n{sep}")
    print("BRIDGE EDGES  (Step 5 — not part of this import)")
    print(sep)
    print("  After import, Step 5 will compare :Note embeddings against")
    print("  :Entity nodes in Neo4j and create :RELATES_TO edges where")
    print(f"  cosine similarity >= {BRIDGE_THRESHOLD}.")
    print()


def _print_cypher_preview(notes: list[ParsedNote], title_map: dict) -> None:
    for n in notes:
        print(f"""
  // ── {n.title} ──
  MERGE (note:`Note`:`Vault` {{id: '{n.id}'}})
  SET note.title     = '{n.title.replace("'", "\\'")}',
      note.path      = '{n.path.replace("'", "\\'")}',
      note.text      = <{n.word_count} words>,
      note.tags      = {n.tags},
      note.embedding = [<{EMBED_DIM} floats>];""")

        for tag in n.tags:
            print(f"""
  MERGE (t:Tag:Vault {{name: '{tag}'}})
  MERGE (note)-[:HAS_TAG]->(t);""")

        for target in n.wikilinks:
            if target in title_map:
                tid = title_map[target].id
                print(f"""
  MATCH (b:Note:Vault {{id: '{tid}'}})
  MERGE (note)-[:LINKS_TO]->(b);""")


# ── live import (Step 4) ──────────────────────────────────────────────────────

def live_import(notes: list[ParsedNote]) -> None:
    """Executes the import against Neo4j. Called only when --dry-run is absent."""
    from neo4j import GraphDatabase  # local import — not needed for dry-run

    title_map = {n.title: n for n in notes}
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    with driver.session() as session:
        # 1. vector index for :Note nodes
        session.run("""
            CREATE VECTOR INDEX note_embedding IF NOT EXISTS
            FOR (n:Note) ON (n.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}
        """, dim=EMBED_DIM)

        # 2. upsert each Note node
        for n in notes:
            session.run("""
                MERGE (note:Note:Vault {id: $id})
                SET note.title     = $title,
                    note.path      = $path,
                    note.text      = $text,
                    note.tags      = $tags,
                    note.embedding = $embedding
            """, id=n.id, title=n.title, path=n.path,
                 text=n.text, tags=n.tags, embedding=n.embedding)

        # 3. upsert Tag nodes + HAS_TAG edges
        for n in notes:
            for tag in n.tags:
                session.run("""
                    MERGE (t:Tag:Vault {name: $tag})
                    WITH t
                    MATCH (note:Note:Vault {id: $id})
                    MERGE (note)-[:HAS_TAG]->(t)
                """, tag=tag, id=n.id)

        # 4. LINKS_TO edges (only between notes that exist in this vault)
        for n in notes:
            for target_title in n.wikilinks:
                if target_title in title_map:
                    target = title_map[target_title]
                    session.run("""
                        MATCH (a:Note:Vault {id: $src}), (b:Note:Vault {id: $tgt})
                        MERGE (a)-[:LINKS_TO]->(b)
                    """, src=n.id, tgt=target.id)

    driver.close()
    print(f"  Imported {len(notes)} notes into Neo4j.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default=str(VAULT_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print import plan; do not touch Neo4j")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip embedding (faster dry-run inspection)")
    args = parser.parse_args()

    vault = Path(args.vault)
    if not vault.exists():
        print(f"ERROR: vault not found at {vault}", file=sys.stderr)
        sys.exit(1)

    print(f"\nVault : {vault}")
    print("Parsing notes…")
    notes = walk_vault(vault)
    if not notes:
        print("No .md files found. Check vault path or wait for sync.")
        sys.exit(0)
    print(f"  Found {len(notes)} note(s).")

    if not args.no_embed:
        print(f"Loading embedding model ({EMBED_MODEL})…")
        model = SentenceTransformer(EMBED_MODEL)
        embed_notes(notes, model)

    if args.dry_run:
        dry_run_report(notes)
        print("Dry run complete. Re-run without --dry-run to import into Neo4j.")
    else:
        print("Importing into Neo4j…")
        live_import(notes)

    print("Done.\n")


if __name__ == "__main__":
    main()
