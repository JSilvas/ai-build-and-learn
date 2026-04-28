"""
app.py — Gradio UI for the Everstorm RAG chatbot.

Two tabs:
  Ingest Documents — select PDFs, configure chunking, run ingest_pipeline on Union
  Chat             — ask questions, get answers with collapsible source accordions

Run:
    python app.py
"""

import base64
import json
import time
from pathlib import Path

import flyte
import gradio as gr

from workflows import ingest_pipeline, query_pipeline

DATA_DIR  = Path(__file__).parent / "data"
CSS_FILE  = Path(__file__).parent / "styles.css"

# ── HTML builders ─────────────────────────────────────────────────────────────

def _score_class(score: float) -> str:
    if score >= 0.80:
        return "score-high"
    if score >= 0.60:
        return "score-mid"
    return "score-low"


def build_source_item(chunk: dict) -> str:
    cls = _score_class(chunk["score"])
    return (
        '<div class="source-item">'
        '  <div class="source-header">'
        f'    <span class="{cls}">{chunk["score"]}</span>'
        f'    <span class="source-doc">{chunk["source_doc"]}</span>'
        '  </div>'
        f'  <p class="source-text">{chunk["chunk_text"][:220]}...</p>'
        '</div>'
    )


def build_sources_accordion(context_chunks: list[dict]) -> str:
    items = "".join(build_source_item(c) for c in context_chunks)
    return (
        '<details class="source-accordion">'
        f'<summary>📄 {len(context_chunks)} source chunks retrieved</summary>'
        f'{items}'
        '</details>'
    )


def build_sources_accordion_names_only(source_names: list[str]) -> str:
    items = "".join(
        f'<div class="source-item"><span class="source-doc">{s}</span></div>'
        for s in source_names
    )
    return (
        '<details class="source-accordion">'
        f'<summary>📄 {len(source_names)} sources used</summary>'
        f'{items}'
        '</details>'
    )


def build_run_link(run) -> str:
    try:
        url = run.url
        return f'<a href="{url}" target="_blank">🔗 View run on Union</a>'
    except Exception:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pdf_choices() -> list[str]:
    return sorted(p.name for p in DATA_DIR.glob("*.pdf"))


# ── Ingest handler ────────────────────────────────────────────────────────────

def run_ingest(selected_files, collection_name, chunk_size, chunk_overlap):
    if not selected_files:
        yield "⚠️  No files selected.", ""
        return

    if not collection_name.strip():
        yield "⚠️  Collection name cannot be empty.", ""
        return

    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    yield emit(f"⏳ Encoding {len(selected_files)} PDFs..."), ""

    filenames, pdf_bytes_b64 = [], []
    for fname in selected_files:
        b64 = base64.b64encode((DATA_DIR / fname).read_bytes()).decode()
        filenames.append(fname)
        pdf_bytes_b64.append(b64)
        yield emit(f"   ✅ {fname}"), ""

    yield emit("\n🚀 Dispatching ingest_pipeline → Union cluster..."), ""

    try:
        run = flyte.run(
            ingest_pipeline,
            filenames=filenames,
            pdf_bytes_b64=pdf_bytes_b64,
            collection_name=collection_name.strip(),
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )

        link = build_run_link(run)
        yield emit("⏳ Running on Union — waiting for results..."), link

        stats = json.loads(run.outputs().o0)

        yield emit(
            f"\n✅ Ingest complete!\n"
            f"   Collection : {stats['collection_name']}\n"
            f"   Chunks     : {stats['total_chunks']}\n"
            f"   Vectors    : {stats['vectors_upserted']}\n"
            f"   Model      : {stats['embed_model']} ({stats['embed_dim']}D)"
        ), link

    except Exception as exc:
        yield emit(f"\n❌ Error: {exc}"), ""


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat(query, history, collection_name, top_k):
    query = query.strip()
    if not query:
        return history, ""

    history = history or []
    history.append({"role": "user", "content": query})

    if not collection_name.strip():
        history.append({
            "role": "assistant",
            "content": "⚠️ Please set a collection name before chatting.",
        })
        return history, ""

    try:
        run = flyte.run(
            query_pipeline,
            query=query,
            collection_name=collection_name.strip(),
            k=int(top_k),
        )
        outputs = run.outputs()
        deadline = time.time() + 180
        while (outputs is None or outputs.o0 is None) and time.time() < deadline:
            time.sleep(3)
            outputs = run.outputs()
        if outputs is None or outputs.o0 is None:
            raise TimeoutError("Query pipeline timed out waiting for output")
        result = json.loads(outputs.o0)
        answer  = result["answer"]
        sources = result.get("sources", [])

        # Try to get full context chunks for rich accordion; fall back to names only
        try:
            context_chunks = json.loads(run.outputs().o1)
            accordion = build_sources_accordion(context_chunks)
        except Exception:
            accordion = build_sources_accordion_names_only(sources)

        history.append({
            "role": "assistant",
            "content": f"{answer}\n\n{accordion}",
        })

    except Exception as exc:
        history.append({
            "role": "assistant",
            "content": f"❌ Error: {exc}",
        })

    return history, ""


# ── UI layout ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Everstorm RAG Chatbot") as app:

        gr.Markdown("# Everstorm Outfitters — RAG Chatbot")
        gr.Markdown(
            "Customer support Q&A powered by pgvector semantic search + Claude, "
            "compute running on Union."
        )

        with gr.Tabs():

            # ── Tab 1: Ingest ─────────────────────────────────────────────────
            with gr.Tab("📥 Ingest Documents"):

                with gr.Row():
                    ingest_collection = gr.Textbox(
                        label="Collection Name",
                        value="everstorm_docs",
                        scale=2,
                    )
                    chunk_size = gr.Slider(
                        minimum=100, maximum=800, value=300, step=50,
                        label="Chunk Size (chars)",
                        scale=2,
                    )
                    chunk_overlap = gr.Slider(
                        minimum=0, maximum=150, value=30, step=10,
                        label="Chunk Overlap (chars)",
                        scale=2,
                    )

                file_selector = gr.CheckboxGroup(
                    choices=_pdf_choices(),
                    value=_pdf_choices(),
                    label="PDFs to Ingest  (all selected by default)",
                )

                ingest_btn = gr.Button("🚀 Run Ingest on Union", variant="primary")

                run_link = gr.HTML(elem_classes=["run-link"])

                status_log = gr.Textbox(
                    label="Status Log",
                    lines=14,
                    interactive=False,
                    elem_classes=["log-box"],
                )

                ingest_btn.click(
                    fn=run_ingest,
                    inputs=[file_selector, ingest_collection, chunk_size, chunk_overlap],
                    outputs=[status_log, run_link],
                )

            # ── Tab 2: Chat ───────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):

                with gr.Row():
                    chat_collection = gr.Textbox(
                        label="Collection Name",
                        value="everstorm_docs",
                        scale=3,
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Top-k Chunks",
                        scale=1,
                    )

                chatbot = gr.Chatbot(
                    label="Everstorm Support",
                    height=480,
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask anything about Everstorm Outfitters...",
                        label="",
                        scale=5,
                        submit_btn=True,
                    )
                    clear_btn = gr.Button("🗑 Clear", scale=1)

                query_input.submit(
                    fn=chat,
                    inputs=[query_input, chatbot, chat_collection, top_k],
                    outputs=[chatbot, query_input],
                )

                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, query_input],
                )

                # Keep chat collection in sync when ingest collection changes
                ingest_collection.change(
                    fn=lambda v: v,
                    inputs=[ingest_collection],
                    outputs=[chat_collection],
                )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    css = (Path(__file__).parent / "styles.css").read_text()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
