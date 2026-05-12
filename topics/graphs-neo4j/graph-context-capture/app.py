"""
Graph Context Capture — screenshot → Gemma 4 → caption + Neo4j graph store.

Two RAG modes in the chat tab:
  Vector RAG  — cosine similarity on activity embeddings
  GraphRAG    — vector search + entity graph traversal (Activity → Entity ← Activity)

Prerequisites:
  docker compose up -d   # wait ~15s for Neo4j to be ready
  uv run python app.py   # → http://localhost:7868

Neo4j Browser: http://localhost:7474  (neo4j / password)
macOS: grant Screen Recording permission to the terminal before running.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import io
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

import gradio as gr
from neo4j import GraphDatabase, RoutingControl
from neo4j_graphrag.embeddings.base import Embedder as _GraphRAGEmbedder
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_viz.neo4j import from_neo4j
import mss
import ollama
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma4:26b"

CAPTURE_CADENCE_S     = 5
MAX_SIDE              = 768
MAX_OUT_TOKENS        = 80
BUFFER_MAX            = 12

CONSOLIDATE_CADENCE_S = 60
OUTLINE_MAX_CHARS     = 600

GRAPH_REFRESH_CADENCE_S = 30

RAG_TOP_K            = 3
RAG_SIMILARITY_FLOOR = 0.45

LEVEL_MINUTE = "minute"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384

LOG_MAX_LINES = 50

VAULT_PATH       = Path(
    "/Users/jaysilvas/Library/CloudStorage/OneDrive-Personal"
    "/Synthetic Knowledge/LLM Knowledge WIki"
)
BRIDGE_THRESHOLD = 0.65

# ── INIT ──────────────────────────────────────────────────────────────────────

print("Loading embedding model…")
_embedder = SentenceTransformer(EMBED_MODEL)
print("Connecting to Neo4j…")
_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

_caption_lock     = threading.Lock()
_consolidate_lock = threading.Lock()
_running = False


def _bootstrap_db() -> None:
    for attempt in range(12):
        try:
            with _driver.session() as s:
                s.run(
                    """
                    CREATE VECTOR INDEX activity_embedding IF NOT EXISTS
                    FOR (a:Activity) ON (a.embedding)
                    OPTIONS { indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """,
                    dim=EMBED_DIM,
                )
            print("Neo4j ready.")
            return
        except Exception as exc:
            if attempt == 11:
                raise RuntimeError(f"Neo4j not reachable at {NEO4J_URI}: {exc}") from exc
            print(f"  Neo4j not ready yet (attempt {attempt + 1}/12) — retrying in 3s…")
            time.sleep(3)


_bootstrap_db()

# ── GRAPHRAG SETUP ────────────────────────────────────────────────────────────

class _STEmbedder(_GraphRAGEmbedder):
    """Wraps our loaded SentenceTransformer so neo4j-graphrag can use it."""
    def embed_query(self, text: str) -> list[float]:
        return _embedder.encode(text).tolist()

    async def async_embed_query(self, text: str) -> list[float]:  # type: ignore[override]
        return self.embed_query(text)


# Cypher that the VectorCypherRetriever appends after vector search.
# node  = matched Activity; score = cosine similarity
# Traverses Activity → Entity ← Activity to surface related context.
GRAPHRAG_CYPHER = """
WITH node AS activity, score
OPTIONAL MATCH (activity)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Activity)
WHERE related.id <> activity.id
WITH activity, score,
     collect(DISTINCT e.name)[..5] AS entities,
     collect(DISTINCT related.text)[..2] AS related_texts
RETURN activity.text       AS text,
       activity.timestamp  AS metadata_timestamp,
       activity.level      AS metadata_level,
       entities            AS metadata_entities,
       related_texts       AS metadata_related,
       score
"""

_graphrag_retriever = VectorCypherRetriever(
    driver=_driver,
    index_name="activity_embedding",
    embedder=_STEmbedder(),
    retrieval_query=GRAPHRAG_CYPHER,
)

print("GraphRAG retriever ready.")

# ── PROMPTS ───────────────────────────────────────────────────────────────────

CAPTION_PROMPT = (
    "You are watching a live screen recording. "
    "Describe what the person is doing RIGHT NOW in one direct sentence. "
    "Be specific: name the app, file, or content visible. "
    "No preamble. Present tense."
)

CONSOLIDATE_PROMPT = (
    "Recent screen activity (last ~60s):\n{captions}\n\n"
    "Current context outline:\n{outline}\n\n"
    "{prior_context}"
    "Update the outline to reflect the current session. "
    "Keep it under 500 characters. If prior context is relevant, briefly reference it. "
    "Focus on: what app/task is active, what they've been working on, any recent transitions. "
    "Remove stale or redundant detail. No preamble."
)

COMPACT_PROMPT = (
    "Compress this context outline to under 350 characters. "
    "Preserve the most important recent activity:\n\n{outline}"
)

ENTITY_EXTRACT_PROMPT = (
    "Extract named entities from this screen activity description. "
    "Return only valid JSON, nothing else.\n"
    '{"entities": [{"name": "...", "type": "APP|FILE|TASK|URL|CONCEPT"}]}\n\n'
    "Description: {text}"
)

CHAT_SYSTEM_PROMPT = (
    "You are a personal context assistant with access to a log of the user's screen activity.\n\n"
    "{context_block}"
    "Answer the user's question using the activity log. Be specific and direct. "
    "If the log doesn't contain enough information to answer, say so clearly."
)

CHAT_SYSTEM_PROMPT_EMPTY = (
    "You are a personal context assistant. The user's screen activity log is currently empty — "
    "no outlines have been stored yet. Let the user know and offer to help once sessions have run."
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def resolve_host(url: str) -> str:
    host = url.strip().rstrip("/")
    return host if host else "http://localhost:11434"


def make_client(url: str) -> ollama.Client:
    return ollama.Client(host=resolve_host(url))


def host_status(url: str) -> str:
    host = resolve_host(url)
    if "localhost" in host or "127.0.0.1" in host:
        return f"🟡 Local — {host}"
    return "🟢 Remote — connected"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log: str, line: str) -> str:
    lines = log.splitlines() if log else []
    lines.append(f"[{ts()}] {line}")
    return "\n".join(lines[-LOG_MAX_LINES:])


def embed(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


# ── STORE OPERATIONS ──────────────────────────────────────────────────────────

def save_outline(outline: str, level: str = LEVEL_MINUTE) -> tuple[int, str]:
    node_id   = f"{level}_{int(time.time() * 1000)}"
    now_ts    = ts()
    now_epoch = int(time.time())
    vec       = embed(outline)

    with _driver.session() as s:
        s.run(
            """
            CREATE (a:Activity {
                id: $id, text: $text, timestamp: $ts,
                ts_epoch: $epoch, level: $level, embedding: $vec
            })
            """,
            id=node_id, text=outline, ts=now_ts, epoch=now_epoch, level=level, vec=vec,
        )
        s.run(
            """
            MATCH (prev:Activity)
            WHERE prev.id <> $id AND prev.level = $level
            WITH prev ORDER BY prev.ts_epoch DESC LIMIT 1
            MATCH (new:Activity {id: $id})
            CREATE (prev)-[:NEXT]->(new)
            """,
            id=node_id, level=level,
        )
        count = s.run("MATCH (a:Activity) RETURN count(a) AS n").single()["n"]
    return count, node_id


def latest_outline() -> tuple[str, dict | None]:
    with _driver.session() as s:
        row = s.run(
            """
            MATCH (a:Activity)
            RETURN a.text AS text, a.timestamp AS timestamp,
                   a.level AS level, a.ts_epoch AS ts_epoch
            ORDER BY a.ts_epoch DESC LIMIT 1
            """
        ).single()
    if row is None:
        return "", None
    return row["text"], dict(row)


def search_store(query: str) -> list[dict]:
    vec = embed(query)
    with _driver.session() as s:
        rows = s.run(
            """
            CALL db.index.vector.queryNodes('activity_embedding', $k, $vec)
            YIELD node, score
            WHERE score >= $floor
            RETURN node.text AS text, node.timestamp AS timestamp,
                   node.level AS level, score
            """,
            k=RAG_TOP_K, vec=vec, floor=RAG_SIMILARITY_FLOOR,
        )
        return [dict(r) for r in rows]


def search_store_graphrag(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    result = _graphrag_retriever.search(query_text=query, top_k=top_k)
    hits = []
    for item in result.items:
        meta = item.metadata or {}
        hits.append({
            "text":      item.content,
            "timestamp": meta.get("metadata_timestamp", ""),
            "level":     meta.get("metadata_level", LEVEL_MINUTE),
            "entities":  meta.get("metadata_entities", []),
            "related":   meta.get("metadata_related", []),
            "score":     meta.get("score", 0.0),
        })
    return hits


def store_summary() -> str:
    with _driver.session() as s:
        a = s.run("MATCH (a:Activity) RETURN count(a) AS n").single()["n"]
        e = s.run("MATCH (e:Entity)   RETURN count(e) AS n").single()["n"]
    if a == 0:
        return "**0** activities stored"
    return f"**{a}** activities · **{e}** entities"


def format_rag_hits(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = "\n".join(
        f"  [{h['timestamp']} {h['level']}] {h['text']}" for h in hits
    )
    return f"Relevant prior context (cite if applicable):\n{lines}\n\n"


# ── ENTITY EXTRACTION ─────────────────────────────────────────────────────────

def extract_entities(text: str, host_url: str, model: str) -> list[dict]:
    try:
        resp = make_client(host_url).chat(
            model=model,
            messages=[{"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(text=text)}],
            stream=False,
            think=False,
            options={"temperature": 0.1, "num_predict": 200},
            format="json",
        )
        data = json.loads(resp.message.content or "{}")
        return [e for e in data.get("entities", []) if e.get("name")]
    except Exception:
        return []


def save_entities(entities: list[dict], activity_id: str) -> None:
    with _driver.session() as s:
        for ent in entities:
            name  = str(ent.get("name", "")).strip()
            etype = str(ent.get("type", "CONCEPT")).strip()
            if name:
                s.run(
                    """
                    MERGE (e:Entity {name: $name, type: $type})
                    WITH e
                    MATCH (a:Activity {id: $aid})
                    MERGE (a)-[:MENTIONS]->(e)
                    """,
                    name=name, type=etype, aid=activity_id,
                )


def _async_extract_entities(outline: str, activity_id: str, host_url: str, model: str) -> None:
    """Fire-and-forget thread — entity extraction runs after consolidation."""
    entities = extract_entities(outline, host_url, model)
    if entities:
        save_entities(entities, activity_id)


# ── VAULT PARSING ────────────────────────────────────────────────────────────

_WIKILINK_RE    = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_HASHTAG_RE     = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_/-]+)")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_vault_note(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter: dict = {}
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        try:
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            pass
        body = raw[m.end():]
    wikilinks   = [t.strip() for t in _WIKILINK_RE.findall(body)]
    inline_tags = _HASHTAG_RE.findall(body)
    fm_tags     = frontmatter.get("tags", [])
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    rel_path = str(path.relative_to(VAULT_PATH))
    return {
        "id":        hashlib.sha256(rel_path.encode()).hexdigest()[:16],
        "title":     path.stem,
        "path":      rel_path,
        "text":      body.strip(),
        "tags":      list(dict.fromkeys(fm_tags + inline_tags)),
        "wikilinks": wikilinks,
    }


def _walk_vault() -> list[dict]:
    if not VAULT_PATH.exists():
        return []
    return [
        _parse_vault_note(md)
        for md in sorted(VAULT_PATH.rglob("*.md"))
        if ".obsidian" not in md.parts
    ]


# ── VAULT IMPORT + BRIDGES ────────────────────────────────────────────────────

def import_vault_notes() -> str:
    notes = _walk_vault()
    if not notes:
        return "No .md files found in vault — check VAULT_PATH or wait for sync."

    texts = [f"{n['title']}\n{n['text']}" for n in notes]
    vecs  = [v.tolist() for v in _embedder.encode(texts, batch_size=32, show_progress_bar=False)]

    with _driver.session() as s:
        s.run(
            """
            CREATE VECTOR INDEX note_embedding IF NOT EXISTS
            FOR (n:Note) ON (n.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            dim=EMBED_DIM,
        )
        title_map = {n["title"]: n["id"] for n in notes}
        for note, vec in zip(notes, vecs):
            s.run(
                """
                MERGE (n:Note:Vault {id: $id})
                SET n.title     = $title,
                    n.path      = $path,
                    n.text      = $text,
                    n.tags      = $tags,
                    n.embedding = $emb
                """,
                id=note["id"], title=note["title"], path=note["path"],
                text=note["text"], tags=note["tags"], emb=vec,
            )
            for tag in note["tags"]:
                s.run(
                    """
                    MERGE (t:Tag:Vault {name: $tag})
                    WITH t
                    MATCH (n:Note:Vault {id: $id})
                    MERGE (n)-[:HAS_TAG]->(t)
                    """,
                    tag=tag, id=note["id"],
                )
        for note in notes:
            for target in note["wikilinks"]:
                if target in title_map:
                    s.run(
                        """
                        MATCH (a:Note:Vault {id: $src}), (b:Note:Vault {id: $tgt})
                        MERGE (a)-[:LINKS_TO]->(b)
                        """,
                        src=note["id"], tgt=title_map[target],
                    )
    return f"Imported {len(notes)} note(s) into Neo4j."


def build_bridges() -> str:
    with _driver.session() as s:
        # name-match: Note title == Entity name (case-insensitive)
        name_count = (s.run(
            """
            MATCH (n:Note:Vault), (e:Entity)
            WHERE toLower(e.name) = toLower(n.title)
            MERGE (n)-[r:RELATES_TO]->(e)
            ON CREATE SET r.score = 1.0, r.method = 'name_match'
            RETURN count(r) AS c
            """
        ).single() or {"c": 0})["c"]

        # embedding similarity: Note → Activity via vector index
        note_rows = s.run(
            "MATCH (n:Note:Vault) WHERE n.embedding IS NOT NULL RETURN n.id AS id, n.embedding AS emb"
        ).data()

        embed_count = 0
        for row in note_rows:
            r = s.run(
                """
                CALL db.index.vector.queryNodes('activity_embedding', 5, $vec)
                YIELD node AS activity, score
                WHERE score >= $threshold
                WITH activity, score
                MATCH (n:Note:Vault {id: $nid})
                MERGE (n)-[rel:RELATES_TO]->(activity)
                ON CREATE SET rel.score = score, rel.method = 'embedding'
                RETURN count(rel) AS c
                """,
                vec=row["emb"], threshold=BRIDGE_THRESHOLD, nid=row["id"],
            ).single()
            embed_count += r["c"] if r else 0

    parts = []
    if name_count:
        parts.append(f"{name_count} name-match")
    if embed_count:
        parts.append(f"{embed_count} embedding")
    if not parts:
        return f"No bridges found (threshold {BRIDGE_THRESHOLD}) — run more capture sessions first."
    return "Built " + " + ".join(parts) + " bridge(s)."


def vault_stats() -> str:
    with _driver.session() as s:
        n = (s.run("MATCH (n:Note:Vault) RETURN count(n) AS c").single() or {"c": 0})["c"]
        t = (s.run("MATCH (t:Tag:Vault)  RETURN count(t) AS c").single() or {"c": 0})["c"]
        b = (s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single() or {"c": 0})["c"]
    return f"**{n}** notes · **{t}** tags · **{b}** bridges"


def build_combined_graph() -> str:
    try:
        eager = _driver.execute_query(
            """
            MATCH (a)-[r]->(b)
            WHERE (a:Activity OR a:Entity OR a:Note OR a:Tag)
              AND (b:Activity OR b:Entity OR b:Note OR b:Tag)
            RETURN a, r, b
            LIMIT 400
            """,
            routing_=RoutingControl.READ,
        )
        if not eager.records:
            return _EMPTY_GRAPH_HTML

        vg       = from_neo4j(eager)
        html_str = vg.render(theme="dark").data or ""
        safe     = html_lib.escape(html_str)
        return (
            f'<iframe srcdoc="{safe}" width="100%" height="620px" '
            f'style="border:none;" title="Knowledge Graph"></iframe>'
        )
    except Exception as exc:
        return f"<p style='color:red;padding:1rem;font-family:monospace;'>Graph error: {exc}</p>"


def _import_and_stats() -> tuple[str, str]:
    return import_vault_notes(), vault_stats()


def _bridge_and_stats() -> tuple[str, str]:
    return build_bridges(), vault_stats()


# ── GRAPH VISUALIZATION ───────────────────────────────────────────────────────

_EMPTY_GRAPH_HTML = (
    "<div style='padding:2rem;color:#888;text-align:center;font-family:sans-serif;'>"
    "No activity data yet — press <b>Start</b> and wait ~60s for first consolidation."
    "</div>"
)


def build_activity_graph() -> str:
    try:
        eager = _driver.execute_query(
            """
            MATCH (a)-[r]->(b)
            WHERE (a:Activity OR a:Entity) AND (b:Activity OR b:Entity)
            RETURN a, r, b
            LIMIT 300
            """,
            routing_=RoutingControl.READ,
        )
        if not eager.records:
            return _EMPTY_GRAPH_HTML

        vg = from_neo4j(eager)
        html_str = vg.render(theme="dark").data
        safe = html_lib.escape(html_str)
        return (
            f'<iframe srcdoc="{safe}" width="100%" height="620px" '
            f'style="border:none;" title="Activity Graph"></iframe>'
        )
    except Exception as exc:
        return f"<p style='color:red;padding:1rem;font-family:monospace;'>Graph error: {exc}</p>"


# ── CAPTURE LOOP ──────────────────────────────────────────────────────────────

def capture_screen() -> tuple[str, Image.Image]:
    with mss.MSS() as sct:
        raw = sct.grab(sct.monitors[1])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode(), img


def caption_tick(host_url: str, model: str, caption_buffer: list[str], log: str):
    if not _running:
        return

    if not _caption_lock.acquire(blocking=False):
        log = append_log(log, "⏭ Caption skipped — previous call in flight")
        yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()
        return

    try:
        log = append_log(log, "📸 Capturing screen…")
        yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()

        try:
            b64, img = capture_screen()
        except Exception as e:
            log = append_log(log, f"✗ Capture failed: {e}")
            yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()
            return

        kb = len(b64) * 3 // 4 // 1024
        log = append_log(log, f"  {img.size[0]}×{img.size[1]}px ~{kb}KB — querying {model} @ {host_status(host_url)}")
        yield img, gr.skip(), caption_buffer, log, gr.skip()

        t0 = time.time()
        try:
            stream = make_client(host_url).chat(
                model=model,
                messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [b64]}],
                stream=True,
                think=False,
                options={"temperature": 0.3, "num_predict": MAX_OUT_TOKENS},
            )
        except Exception as e:
            log = append_log(log, f"✗ LLM error: {e}")
            yield gr.skip(), f"**Error:** {e}", caption_buffer, log, gr.skip()
            return

        reply = ""
        for chunk in stream:
            reply += chunk.message.content or ""
            yield gr.skip(), reply, caption_buffer, log, gr.skip()

        reply = reply.strip()
        elapsed = time.time() - t0

        if not reply:
            log = append_log(log, f"✗ Empty response ({elapsed:.1f}s)")
            yield gr.skip(), "_Empty response._", caption_buffer, log, gr.skip()
            return

        log = append_log(log, f"✓ {elapsed:.1f}s: \"{reply[:80]}{'…' if len(reply) > 80 else ''}\"")
        new_buffer = (caption_buffer + [reply])[-BUFFER_MAX:]
        buffer_md = "\n".join(f"{i+1}. {c}" for i, c in enumerate(new_buffer))
        yield gr.skip(), reply, new_buffer, log, buffer_md

    finally:
        _caption_lock.release()


def consolidate_tick(host_url: str, model: str, caption_buffer: list[str], outline: str, log: str):
    if not _running:
        return

    if not _consolidate_lock.acquire(blocking=False):
        log = append_log(log, "⏭ Consolidation skipped — previous cycle running")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()
        return

    try:
        if not caption_buffer:
            log = append_log(log, "⏭ Consolidation skipped — no captions yet")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        if len(outline) > OUTLINE_MAX_CHARS:
            log = append_log(log, f"⚙ Outline {len(outline)} chars — compacting…")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            try:
                resp = make_client(host_url).chat(
                    model=model,
                    messages=[{"role": "user", "content": COMPACT_PROMPT.format(outline=outline)}],
                    stream=False,
                    think=False,
                    options={"temperature": 0.2, "num_predict": 120},
                )
                outline = (resp.message.content or outline).strip()
                log = append_log(log, f"✓ Compacted to {len(outline)} chars")
            except Exception as e:
                log = append_log(log, f"✗ Compaction failed: {e}")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()

        captions_text = "\n".join(f"- {c}" for c in caption_buffer)

        log = append_log(log, "🔍 Searching Neo4j for relevant prior context…")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        hits      = search_store(captions_text)
        prior_ctx = format_rag_hits(hits)
        log = append_log(
            log,
            f"  {len(hits)} hit(s) found" if hits else "  No relevant prior context",
        )
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        log = append_log(log, f"🗂 Consolidating {len(caption_buffer)} captions…")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        try:
            stream = make_client(host_url).chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": CONSOLIDATE_PROMPT.format(
                        captions=captions_text,
                        outline=outline or "(empty)",
                        prior_context=prior_ctx,
                    ),
                }],
                stream=True,
                think=False,
                options={"temperature": 0.3, "num_predict": 200},
            )
        except Exception as e:
            log = append_log(log, f"✗ Consolidation LLM error: {e}")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        new_outline = ""
        for chunk in stream:
            new_outline += chunk.message.content or ""
            yield new_outline, new_outline, caption_buffer, log, gr.skip()

        new_outline = new_outline.strip()
        if not new_outline:
            log = append_log(log, "✗ Empty consolidation response")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        count, node_id = save_outline(new_outline)
        log = append_log(log, f"✓ Activity node #{count} saved — extracting entities…")

        # Entity extraction runs in a background thread (non-blocking)
        threading.Thread(
            target=_async_extract_entities,
            args=(new_outline, node_id, host_url, model),
            daemon=True,
        ).start()

        yield new_outline, new_outline, [], log, store_summary()

    finally:
        _consolidate_lock.release()


# ── CONTEXT CHAT ──────────────────────────────────────────────────────────────

def context_chat(
    message, history, host_url, model,
    top_k, min_similarity, temperature, max_tokens,
    rag_mode,
):
    if not message.strip():
        return history, "", ""

    use_graphrag = (rag_mode == "GraphRAG")

    if use_graphrag:
        raw_hits = search_store_graphrag(message, top_k=top_k)
        debug_lines = [f"**GraphRAG — {len(raw_hits)} candidates** (vector + entity graph):\n"]
        context_lines = []
        for h in raw_hits:
            score  = round(float(h.get("score") or 0), 3)
            passed = score >= min_similarity
            entities_str = ", ".join(h.get("entities") or [])
            debug_lines.append(
                f"{'✓' if passed else '✗ filtered'} **[{h['timestamp']}]** "
                f"`{h['level']}` score `{score}`"
                + (f"  entities: _{entities_str}_" if entities_str else "")
                + f"  \n> {h['text']}"
            )
            if passed:
                line = f"[{h['timestamp']} {h['level']}] (score {score}) {h['text']}"
                if entities_str:
                    line += f"  [entities: {entities_str}]"
                context_lines.append(line)
    else:
        vec = embed(message)
        with _driver.session() as s:
            raw_hits = list(s.run(
                """
                CALL db.index.vector.queryNodes('activity_embedding', $k, $vec)
                YIELD node, score
                RETURN node.text AS text, node.timestamp AS timestamp,
                       node.level AS level, score
                ORDER BY score DESC
                """,
                k=top_k, vec=vec,
            ))
        debug_lines = [f"**Vector RAG — {len(raw_hits)} candidates** (min similarity: {min_similarity}):\n"]
        context_lines = []
        for row in raw_hits:
            sim    = round(row["score"], 3)
            passed = sim >= min_similarity
            debug_lines.append(
                f"{'✓' if passed else '✗ filtered'} **[{row['timestamp']}]** "
                f"`{row['level']}` similarity `{sim}`  \n> {row['text']}"
            )
            if passed:
                context_lines.append(
                    f"[{row['timestamp']} {row['level']}] (similarity {sim}) {row['text']}"
                )

    retrieved_md = "\n\n".join(debug_lines)
    if not context_lines:
        retrieved_md += "\n\n_All results filtered — LLM receives no context._"
        system = CHAT_SYSTEM_PROMPT_EMPTY
    else:
        context_block = (
            f"Activity log ({len(context_lines)} entries):\n"
            + "\n".join(context_lines)
            + "\n\n"
        )
        system = CHAT_SYSTEM_PROMPT.format(context_block=context_block)

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": extract_text(turn["content"])})
    messages.append({"role": "user", "content": message})

    history = history + [{"role": "user", "content": message}]
    yield history, "", retrieved_md

    try:
        stream = make_client(host_url).chat(
            model=model, messages=messages, stream=True, think=False,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"**Error:** {e}"}], "", retrieved_md
        return

    reply = ""
    for chunk in stream:
        reply += chunk.message.content or ""
        yield history + [{"role": "assistant", "content": reply}], "", retrieved_md

    history = history + [{"role": "assistant", "content": reply.strip()}]
    yield history, "", retrieved_md


# ── START / STOP ──────────────────────────────────────────────────────────────

def start(log: str):
    global _running
    _running = True

    doc, meta = latest_outline()
    if doc and meta:
        seed_ts = meta.get("timestamp", "unknown time")
        level   = meta.get("level", LEVEL_MINUTE)
        log     = append_log(log, f"▶ Started · seeded from {seed_ts} ({level})")
        display = f"_Seeded {seed_ts} ({level})_\n\n{doc}"
    else:
        doc     = ""
        log     = append_log(log, "▶ Started · no prior context in Neo4j")
        display = "_Waiting for first consolidation (~60s)…_"

    log = append_log(log, f"  {store_summary()}")
    return [], doc, "_Starting — first capture in ~5s…_", display, log


def stop(log: str):
    global _running
    _running = False
    return append_log(log, "■ Session stopped")


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Graph Context Capture") as demo:
        gr.Markdown(
            "# Graph Context Capture\n"
            f"Primary monitor → Gemma 4 vision → caption every **{CAPTURE_CADENCE_S}s**, "
            f"outline consolidated every **{CONSOLIDATE_CADENCE_S}s** into Neo4j. "
            "Activity nodes linked by `:NEXT` edges · entities extracted into `:Entity` nodes. "
            "Chat supports **Vector RAG** or **GraphRAG** retrieval."
        )

        with gr.Row():
            host_url = gr.Textbox(
                label="Ollama URL",
                placeholder="https://your-ngrok-url.ngrok-free.app  (leave blank → localhost:11434)",
                scale=4,
                type="password",
            )
            model_select = gr.Dropdown(
                label="Model", choices=[DEFAULT_MODEL], value=DEFAULT_MODEL, scale=2,
            )
            start_btn = gr.Button("Start", variant="primary", scale=1)
            stop_btn  = gr.Button("Stop",  scale=1)

        status_md = gr.Markdown(host_status(""))
        host_url.change(host_status, inputs=[host_url], outputs=[status_md])

        with gr.Tabs():

            # ── Capture ───────────────────────────────────────────────────────
            with gr.Tab("Capture"):
                with gr.Row():
                    with gr.Column(scale=1):
                        screenshot_img = gr.Image(
                            label="Latest Capture", type="pil",
                            interactive=False, height=360,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Current Focus")
                        focus_md = gr.Markdown("_Press **Start** to begin…_")
                        gr.Markdown("### Recent Summary")
                        summary_md = gr.Markdown("_Appears after first consolidation cycle (~60s)…_")

                log_box = gr.Textbox(
                    label="Process Log", lines=10, max_lines=10,
                    interactive=False, autoscroll=True,
                )
                with gr.Accordion("Caption Buffer (debug)", open=False):
                    buffer_md = gr.Markdown("_No captions yet._")

            # ── Activity Graph ────────────────────────────────────────────────
            with gr.Tab("Activity Graph"):
                with gr.Row():
                    stats_md = gr.Markdown(store_summary())
                    refresh_btn = gr.Button("Refresh", size="sm", scale=0)

                graph_html = gr.HTML(_EMPTY_GRAPH_HTML)
                refresh_btn.click(build_activity_graph, outputs=[graph_html])

            # ── Knowledge Graph ───────────────────────────────────────────────
            with gr.Tab("Knowledge Graph"):
                with gr.Row():
                    kg_stats_md = gr.Markdown(vault_stats())
                with gr.Row():
                    import_btn    = gr.Button("Import Vault Notes", variant="primary", scale=1)
                    bridge_btn    = gr.Button("Build Bridges", scale=1)
                    kg_refresh_btn = gr.Button("Refresh", size="sm", scale=0)
                kg_status_md  = gr.Markdown(
                    "_Import your vault notes, then build bridges to link them with screen activity._"
                )
                kg_graph_html = gr.HTML(_EMPTY_GRAPH_HTML)

                import_btn.click(
                    _import_and_stats, outputs=[kg_status_md, kg_stats_md]
                ).then(
                    build_combined_graph, outputs=[kg_graph_html]
                )
                bridge_btn.click(
                    _bridge_and_stats, outputs=[kg_status_md, kg_stats_md]
                ).then(
                    build_combined_graph, outputs=[kg_graph_html]
                )
                kg_refresh_btn.click(build_combined_graph, outputs=[kg_graph_html])

            # ── Context Chat ──────────────────────────────────────────────────
            with gr.Tab("Context Chat"):
                with gr.Row():
                    stats_chat_md = gr.Markdown(store_summary())
                    clear_btn = gr.Button("Clear chat", size="sm", scale=0)

                with gr.Row():
                    rag_mode = gr.Radio(
                        choices=["Vector RAG", "GraphRAG"],
                        value="Vector RAG",
                        label="Retrieval Mode",
                        info="Vector RAG: cosine similarity only · GraphRAG: vector + entity graph traversal",
                    )

                with gr.Accordion("RAG Parameters", open=True):
                    with gr.Row():
                        slider_top_k = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Top K — candidates retrieved",
                        )
                        slider_min_similarity = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="Min similarity / score — relevance filter",
                            info="Cosine similarity (0 = any match, 1 = near-identical).",
                        )
                    with gr.Row():
                        slider_temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="Temperature",
                        )
                        slider_max_tokens = gr.Slider(
                            minimum=100, maximum=800, value=400, step=50,
                            label="Max tokens",
                        )

                chat_history = gr.State([])
                chatbot = gr.Chatbot(label="Ask about your activity", height=400)

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask a question about your screen activity…",
                        scale=5, show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Accordion("Retrieved Context (debug)", open=False):
                    retrieved_md = gr.Markdown("_Chunks appear here after your first query._")

                chat_inputs = [
                    chat_input, chat_history, host_url, model_select,
                    slider_top_k, slider_min_similarity, slider_temperature, slider_max_tokens,
                    rag_mode,
                ]
                chat_outputs = [chatbot, chat_input, retrieved_md]

                send_btn.click(context_chat, inputs=chat_inputs, outputs=chat_outputs)
                chat_input.submit(context_chat, inputs=chat_inputs, outputs=chat_outputs)
                chatbot.change(lambda h: h, inputs=[chatbot], outputs=[chat_history])
                clear_btn.click(
                    lambda: ([], [], "_Cleared._"),
                    outputs=[chatbot, chat_history, retrieved_md],
                )

        # ── Shared state & timers ─────────────────────────────────────────────
        caption_buffer = gr.State([])
        outline_state  = gr.State("")

        capture_timer     = gr.Timer(value=CAPTURE_CADENCE_S,       active=True)
        consolidate_timer = gr.Timer(value=CONSOLIDATE_CADENCE_S,   active=True)
        graph_timer       = gr.Timer(value=GRAPH_REFRESH_CADENCE_S, active=True)

        capture_timer.tick(
            caption_tick,
            inputs=[host_url, model_select, caption_buffer, log_box],
            outputs=[screenshot_img, focus_md, caption_buffer, log_box, buffer_md],
        )

        consolidate_timer.tick(
            consolidate_tick,
            inputs=[host_url, model_select, caption_buffer, outline_state, log_box],
            outputs=[summary_md, outline_state, caption_buffer, log_box, stats_chat_md],
        )

        graph_timer.tick(build_activity_graph, outputs=[graph_html])

        start_btn.click(
            start,
            inputs=[log_box],
            outputs=[caption_buffer, outline_state, focus_md, summary_md, log_box],
        )
        stop_btn.click(stop, inputs=[log_box], outputs=[log_box])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7868, share=False)
