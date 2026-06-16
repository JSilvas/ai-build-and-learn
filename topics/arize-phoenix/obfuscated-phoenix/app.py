"""
Obfuscated Phoenix - RAG demo with Phoenix tracing and LLM-as-judge evals

Split-panel Gradio app:
  Left:  folder of docs → load → Q&A via LM Studio
  Right: Phoenix trace viewer + LLM-as-judge eval scores

Reuses the graph-context-capture pattern:
  - openai SDK → LM Studio (OpenAI-compatible, no LangChain)
  - SentenceTransformer for local embeddings
  - FAISS in-memory vector store with cosine similarity
  - Direct retrieve → format context → call LM Studio

Phoenix tracing: openinference-instrumentation-openai auto-captures
every openai SDK call (chat completions → spans in Phoenix UI).

Prerequisites:
  - LM Studio running at http://localhost:1234 with a model loaded
  - uv run python app.py   →  http://localhost:7860
  Phoenix UI auto-launches at http://localhost:6006
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ── Phoenix ────────────────────────────────────────────────────────────────────

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.client import Client as PhoenixClient

PROJECT = "obfuscated-phoenix-demo"
PHOENIX_URL = "http://localhost:6006"

try:
    px.launch_app(use_temp_dir=False)
    print(f"Phoenix UI: {PHOENIX_URL}")
except Exception as e:
    print(f"Phoenix auto-launch failed ({e}) — run `phoenix serve` manually, then visit {PHOENIX_URL}")

tracer_provider = register(
    endpoint=f"{PHOENIX_URL}/v1/traces",
    project_name=PROJECT,
    batch=True,
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
_px = PhoenixClient(base_url=PHOENIX_URL)

# ── Embeddings ─────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"
print(f"Loading embedding model {EMBED_MODEL}…")
_st = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = _st.get_sentence_embedding_dimension()

# ── RAG ────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 500   # words per chunk
CHUNK_OVERLAP = 50
TOP_K         = 3
MAX_PDF_PAGES = 50   # cap per book — enough for RAG without multi-minute extraction

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the context below to answer the question. "
    "Be specific and direct. If the context doesn't contain the answer, say so clearly."
)


def _chunk(text: str) -> list[str]:
    words = text.split()
    step  = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        " ".join(words[i : i + CHUNK_SIZE])
        for i in range(0, len(words), step)
        if words[i : i + CHUNK_SIZE]
    ]


def _read_pdf(fp: Path) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(fp))
        pages = reader.pages[:MAX_PDF_PAGES]
        text = "\n".join(page.extract_text() or "" for page in pages)
        if len(reader.pages) > MAX_PDF_PAGES:
            print(f"  {fp.name}: capped at {MAX_PDF_PAGES}/{len(reader.pages)} pages")
        return text
    except Exception as e:
        print(f"  PDF read error {fp.name}: {e}")
        return ""


def load_docs(folder: str):
    path = Path(folder)
    if not path.exists():
        return None, None, f"Folder not found: `{folder}`"

    records: list[dict] = []
    for pattern, reader in (
        ("**/*.txt",  lambda fp: fp.read_text(encoding="utf-8", errors="replace")),
        ("**/*.md",   lambda fp: fp.read_text(encoding="utf-8", errors="replace")),
        ("**/*.pdf",  _read_pdf),
    ):
        for fp in sorted(path.glob(pattern)):
            try:
                text = reader(fp)
            except Exception:
                continue
            if not text.strip():
                continue
            for chunk in _chunk(text):
                records.append({"text": chunk, "source": fp.name})

    if not records:
        return None, None, f"No .txt, .md, or .pdf files found in `{folder}`"

    texts = [r["text"] for r in records]
    vecs  = _st.encode(texts, batch_size=32, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(EMBED_DIM)  # cosine via inner-product on L2-normalized vecs
    index.add(vecs)

    n_files = len({r["source"] for r in records})
    return index, records, f"Loaded **{n_files}** file(s) → **{len(records)}** chunks"


def retrieve(question: str, index: faiss.Index, records: list[dict]) -> list[dict]:
    q_vec = _st.encode([question]).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, TOP_K)
    return [
        {**records[i], "score": float(scores[0][j])}
        for j, i in enumerate(idxs[0]) if i >= 0
    ]


def rag_answer(question: str, index: faiss.Index, records: list[dict], client: OpenAI, model: str) -> tuple[str, str]:
    hits    = retrieve(question, index, records)
    context = "\n\n".join(h["text"] for h in hits)
    sources = "\n".join(f"- `{h['source']}` (score {h['score']:.3f})" for h in hits)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    answer = resp.choices[0].message.content or ""
    return answer, f"**Retrieved ({len(hits)} chunks):**\n{sources}"


# ── Trace helpers ──────────────────────────────────────────────────────────────

def _read_blob(blob: str) -> str:
    """Extract readable text from an OpenInference attribute JSON blob."""
    if not isinstance(blob, str) or not blob.strip():
        return ""
    try:
        data = json.loads(blob)
        msgs = data.get("messages", [])
        if msgs:
            last    = msgs[-1]
            content = last.get("content") or (last.get("data") or {}).get("content", "")
            return str(content)[:200]
    except Exception:
        pass
    return str(blob)[:200]


def _read_input(blob: str) -> str:
    """Extract the user question from a RAG input blob, stripping the context preamble."""
    if not isinstance(blob, str) or not blob.strip():
        return ""
    try:
        data = json.loads(blob)
        for m in data.get("messages", []):
            if m.get("role") == "user":
                content = str(m.get("content", ""))
                # RAG prompt format: "Context:\n...\n\nQuestion: <question>"
                if "Question:" in content:
                    return content.split("Question:")[-1].strip()[:200]
                return content[:200]
    except Exception:
        pass
    return _read_blob(blob)


def _read_output(blob: str) -> str:
    """Extract the assistant response text from an output blob."""
    if not isinstance(blob, str) or not blob.strip():
        return ""
    try:
        data = json.loads(blob)
        # OpenInference wraps completions as {"choices": [{"message": {"content": ...}}]}
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return str(msg.get("content", ""))[:200]
        # Fallback: plain {"content": ...}
        if "content" in data:
            return str(data["content"])[:200]
    except Exception:
        pass
    return _read_blob(blob)


def fetch_traces() -> pd.DataFrame:
    try:
        df = _px.spans.get_spans_dataframe(project_identifier=PROJECT)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

    if df is None or df.empty:
        return pd.DataFrame({"Status": ["No spans yet — ask a question first"]})

    in_col  = next((c for c in ("attributes.input.value",  "input.value")  if c in df.columns), None)
    out_col = next((c for c in ("attributes.output.value", "output.value") if c in df.columns), None)

    # Exclude eval-generated spans so the table shows only RAG query spans
    EVAL_SPAN_NAMES = {"LLM.generate_object", "answer_relevance.evaluate", "answer_completeness.evaluate"}
    rag_df = df[~df["name"].isin(EVAL_SPAN_NAMES)]
    if rag_df.empty:
        rag_df = df

    rows = []
    for _, row in rag_df.tail(20).iterrows():
        latency = ""
        try:
            latency = f"{int((row['end_time'] - row['start_time']).total_seconds() * 1000)} ms"
        except Exception:
            pass
        rows.append({
            "Span":    str(row.get("name", "")),
            "Input":   _read_input(str(row.get(in_col,  "") if in_col  else ""))[:120],
            "Output":  _read_output(str(row.get(out_col, "") if out_col else ""))[:120],
            "Latency": latency,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame({"Status": ["No spans found"]})


# ── Evals ──────────────────────────────────────────────────────────────────────

def run_evals(lm_url: str) -> pd.DataFrame:
    try:
        from phoenix.evals import create_classifier, evaluate_dataframe
        from phoenix.evals.llm import LLM
    except ImportError as e:
        return pd.DataFrame({"Error": [f"arize-phoenix-evals not installed: {e}"]})

    try:
        df = _px.spans.get_spans_dataframe(project_identifier=PROJECT)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

    if df is None or df.empty:
        return pd.DataFrame({"Status": ["No traces to evaluate yet"]})

    in_col  = next((c for c in ("attributes.input.value",  "input.value")  if c in df.columns), None)
    out_col = next((c for c in ("attributes.output.value", "output.value") if c in df.columns), None)

    if not in_col or not out_col:
        return pd.DataFrame({"Status": [f"Expected columns not found. Got: {list(df.columns[:10])}"]})

    llm_span_names = {"ChatCompletion", "chat_completion", "llm", "openai.chat", "OpenAI"}
    targets = df[df["name"].isin(llm_span_names)].copy()
    if targets.empty:
        targets = df[df[in_col].notna() & df[out_col].notna()].copy()
    if targets.empty:
        return pd.DataFrame({"Status": ["No LLM spans found to evaluate"]})

    targets = targets.tail(8)
    targets["input"]  = targets[in_col].map(_read_input)
    targets["output"] = targets[out_col].map(_read_blob)
    targets = targets[targets["input"].str.strip() != ""]
    if targets.empty:
        return pd.DataFrame({"Status": ["Spans have no parseable input text"]})

    judge = LLM(
        provider="openai",
        model="local-model",
        base_url=lm_url.rstrip("/"),
        api_key="lm-studio",
    )
    relevance = create_classifier(
        name="answer_relevance",
        llm=judge,
        prompt_template=(
            "QUESTION:\n{input}\n\nANSWER:\n{output}\n\n"
            "Is the ANSWER relevant to the QUESTION? Reply exactly: relevant or irrelevant."
        ),
        choices={"relevant": 1.0, "irrelevant": 0.0},
    )
    completeness = create_classifier(
        name="answer_completeness",
        llm=judge,
        prompt_template=(
            "QUESTION:\n{input}\n\nANSWER:\n{output}\n\n"
            "Does the ANSWER thoroughly address the QUESTION? Reply exactly: complete or incomplete."
        ),
        choices={"complete": 1.0, "incomplete": 0.0},
    )

    try:
        results = evaluate_dataframe(dataframe=targets, evaluators=[relevance, completeness])
    except Exception as e:
        return pd.DataFrame({"Error": [f"Eval failed: {e}"]})

    for eval_name in ("answer_relevance", "answer_completeness"):
        try:
            scores = results[f"{eval_name}_score"]
            ann = pd.DataFrame(
                {
                    "label":       scores.map(lambda d: d.get("label")       if isinstance(d, dict) else None),
                    "score":       scores.map(lambda d: d.get("score")       if isinstance(d, dict) else None),
                    "explanation": scores.map(lambda d: d.get("explanation") if isinstance(d, dict) else None),
                },
                index=results.index,
            )
            ann.index.name = "span_id"
            _px.spans.log_span_annotations_dataframe(
                dataframe=ann, annotation_name=eval_name, annotator_kind="LLM", sync=True
            )
        except Exception:
            pass

    rows = []
    for span_id in results.index:
        try:
            rel  = results.loc[span_id, "answer_relevance_score"]
            comp = results.loc[span_id, "answer_completeness_score"]
            q    = targets.loc[span_id, "input"] if span_id in targets.index else str(span_id)
            rows.append({
                "Query":        str(q)[:80],
                "Relevance":    f"{rel.get('label','?')} ({rel.get('score','?')})"   if isinstance(rel,  dict) else str(rel),
                "Completeness": f"{comp.get('label','?')} ({comp.get('score','?')})" if isinstance(comp, dict) else str(comp),
            })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame({"Status": ["Eval produced no results"]})


# ── Gradio handlers ────────────────────────────────────────────────────────────

def fetch_models(lm_url: str):
    """Query LM Studio for available models and populate the dropdown."""
    try:
        c = OpenAI(base_url=lm_url.rstrip("/"), api_key="lm-studio")
        all_models = [m.id for m in c.models.list().data]
        chat   = [m for m in all_models if "embed" not in m.lower()]
        embeds = [m for m in all_models if "embed" in m.lower()]
        ordered = chat + embeds
        value = ordered[0] if ordered else None
        return gr.Dropdown(choices=ordered, value=value)
    except Exception as e:
        return gr.Dropdown(choices=[], value=None, label=f"Model (error: {e})")


def handle_load(folder: str, lm_url: str, model_name: str):
    index, records, status = load_docs(folder)
    client = OpenAI(base_url=lm_url.rstrip("/"), api_key="lm-studio") if index is not None else None
    return index, records, client, status


def handle_chat(message: str, history: list, index, records, client, model_name: str):
    if not message.strip():
        return history, "", ""
    if index is None:
        msg = "Please load docs first — enter a folder path and click **Load Docs**."
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": msg}], "", ""
    try:
        answer, sources = rag_answer(message, index, records, client, model_name)
    except Exception as e:
        answer, sources = f"Error: {e}", ""
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], "", sources


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Phoenix RAG Demo") as demo:
        gr.Markdown(
            f"# Phoenix RAG Demo\n"
            f"Folder of docs → RAG pipeline → **Arize Phoenix** traces + LLM-as-judge evals.\n\n"
            f"Phoenix UI → [{PHOENIX_URL}]({PHOENIX_URL})"
        )

        index_state   = gr.State(None)
        records_state = gr.State(None)
        client_state  = gr.State(None)

        with gr.Row():
            lm_url           = gr.Textbox(label="LM Studio base URL", value="http://localhost:1234/v1", scale=3)
            refresh_model_btn = gr.Button("⟳ Models", scale=1, min_width=90)
            model_dropdown   = gr.Dropdown(label="Model", choices=[], value=None, scale=2, interactive=True)

        with gr.Row():

            # ── Left: RAG Q&A ──────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## RAG Q&A")

                with gr.Row():
                    folder_input = gr.Textbox(label="Docs folder", value="docs/", scale=4)
                    load_btn     = gr.Button("Load Docs", variant="primary", scale=1)
                load_status = gr.Markdown("_No docs loaded._")

                chatbot = gr.Chatbot(label="Chat", height=380)

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask a question about your docs…",
                        show_label=False, scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                sources_box = gr.Markdown("_Retrieved chunks appear here after each query._")

            # ── Right: Phoenix traces + evals ──────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## Phoenix Traces & Evals")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh Traces")
                    eval_btn    = gr.Button("Run Evals", variant="primary")

                traces_df = gr.Dataframe(
                    label="Recent Spans",
                    headers=["Span", "Input", "Output", "Latency"],
                    wrap=True,
                )
                evals_df = gr.Dataframe(
                    label="Eval Scores (LLM-as-judge via LM Studio)",
                    headers=["Query", "Relevance", "Completeness"],
                    wrap=True,
                )

        # ── Wiring ────────────────────────────────────────────────────────────
        refresh_model_btn.click(fetch_models, inputs=[lm_url], outputs=[model_dropdown])
        lm_url.change(fetch_models, inputs=[lm_url], outputs=[model_dropdown])

        load_btn.click(
            handle_load,
            inputs=[folder_input, lm_url, model_dropdown],
            outputs=[index_state, records_state, client_state, load_status],
        )

        chat_ins  = [query_input, chatbot, index_state, records_state, client_state, model_dropdown]
        chat_outs = [chatbot, query_input, sources_box]
        send_btn.click(handle_chat,  inputs=chat_ins, outputs=chat_outs)
        query_input.submit(handle_chat, inputs=chat_ins, outputs=chat_outs)

        refresh_btn.click(fetch_traces, outputs=[traces_df])
        eval_btn.click(run_evals, inputs=[lm_url], outputs=[evals_df])

        demo.load(fetch_models, inputs=[lm_url], outputs=[model_dropdown])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
