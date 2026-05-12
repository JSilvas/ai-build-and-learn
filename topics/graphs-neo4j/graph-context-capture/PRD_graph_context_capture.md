# PRD: Graph Context Capture

## Problem Space & Research Questions

The existing screen context harness stores activity outlines as flat vector chunks in ChromaDB. This gives us semantic similarity search, but no structure: we can't answer "what led to what?", traverse a session timeline, or see how activities relate over time. Neo4j's property graph model is a natural fit — outlines become nodes, temporal sequence becomes edges, and the built-in vector index keeps RAG working.

The core research question: **Can we replace ChromaDB with Neo4j (graph + vector index) without losing RAG quality, while gaining a visual activity timeline as a first-class feature?**

## Hypotheses

1. Neo4j's vector index can replace ChromaDB's cosine similarity search with equivalent retrieval quality for this use case.
2. Storing outlines as a temporal graph (`:Activity` nodes connected by `:NEXT` edges) enables a meaningful visual timeline of screen activity.
3. The neovis.js library can be embedded in a Gradio `gr.HTML` component and connect to Neo4j Bolt directly from the browser.

## Validation / Success Metrics

- **Quantitative**: App starts, connects to Neo4j, and captures/stores at least 3 activity nodes without error. Vector similarity query returns results within 500ms.
- **Qualitative**: The graph viz panel shows nodes appearing and linking as captures are stored. A user can click a node and see its outline text. RAG chat returns relevant context.

## Prototype Tech Spec

- [ ] Neo4j running locally via Docker Compose
- [ ] `app.py` ported from screen-context-harness with ChromaDB replaced by Neo4j driver
- [ ] `:Activity` nodes stored with `text`, `timestamp`, `ts_epoch`, `level`, and embedding properties
- [ ] `:NEXT` temporal edges between consecutive same-level nodes
- [ ] Neo4j vector index created on `:Activity` embedding property for RAG search
- [ ] Gradio tab with `gr.HTML` component rendering neovis.js timeline graph
- [ ] Graph viz auto-refreshes every 30s to show new nodes
- [ ] RAG chat working against Neo4j vector index (replaces ChromaDB query)
- [ ] Docker Compose file + uv pyproject.toml with all dependencies

## Scope & Intentional Omissions

**Not building:**
- hourly/daily compaction hierarchy (carry it over if trivial, skip if it complicates Neo4j schema)
- Topic/entity extraction (that's a separate knowledge graph hypothesis)
- Auth, HTTPS, multi-user support
- Custom neovis.js styling beyond minimal readable defaults
- Neo4j Aura cloud — local Docker only

## Implementation Path

Gradio + Neo4j Python driver for backend logic, neovis.js embedded via `gr.HTML` for visualization. This produces real signal on both hypotheses (Neo4j as drop-in vector store, graph viz in Gradio) without a separate frontend build step.

## Decomposition

Not needed — the prototype is narrow enough to test as one app.

## Evaluation Plan

1. `docker compose up` → verify Neo4j is reachable at bolt://localhost:7687
2. `uv run python app.py` → press Start → let 3 consolidation cycles run
3. Check Neo4j Browser (localhost:7474) to confirm Activity nodes + NEXT edges
4. Check Gradio graph tab — nodes should be visible and linked
5. Ask RAG chat "what have I been working on?" → verify relevant outlines returned

## Learnings / Next Steps

[Filled in after validation]
