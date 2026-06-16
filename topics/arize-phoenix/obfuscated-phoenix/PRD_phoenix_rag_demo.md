# PRD: Phoenix RAG Demo

## Problem Space & Research Questions
Arize Phoenix is an open-source LLM observability platform — but it's hard to internalize what it actually shows you until you see traces and evals appear in real time from your own pipeline. The question is: **what does Phoenix capture when a RAG pipeline runs, and how useful are the evals it can automatically score?**

## Hypotheses
- "If we instrument a simple RAG pipeline with Phoenix's OpenInference tracing, the retrieval and generation spans will appear in the Phoenix UI with enough detail to diagnose retrieval failures."
- "If we run answer_relevance and answer_completeness evals via LLM-as-judge after each query, the scores will correlate visibly with response quality."

## Validation / Success Metrics
- Quantitative: At least 3 Q&A turns produce spans visible in Phoenix; evals return scores for each span.
- Qualitative: A viewer can look at the Gradio right panel and clearly see which queries produced good vs. poor retrievals without opening the Phoenix UI directly.

## Prototype Tech Spec
- [ ] Phoenix server starts embedded (or user visits localhost:6006) when the app launches
- [ ] Documents load from a local folder (txt/md files) via LangChain directory loader
- [ ] RAG pipeline: embed docs with LM Studio embeddings → FAISS vector store → retrieve top-k → generate answer via LM Studio chat
- [ ] All LLM and retrieval calls are traced to Phoenix via OpenInference instrumentation
- [ ] Gradio split layout: left = Q&A input/output, right = live trace table + eval scores
- [ ] Right panel fetches recent spans from Phoenix Python client on demand (Refresh button)
- [ ] After each query, answer_relevance and answer_completeness evals run via LM Studio as judge and write annotations back to Phoenix
- [ ] Eval scores surface in the right panel alongside the trace data

## Scope & Intentional Omissions
- No authentication, no multi-user state
- No persistent vector store (rebuilt in-memory on app start)
- No streaming output
- No chunk overlap tuning or advanced retrieval strategies — this is about observability, not retrieval quality
- No Flyte, no remote deployments

## Implementation Path
Standalone `app.py` in `obfuscated-phoenix/`. LangChain handles the RAG chain (already instrumented by the existing `openinference-instrumentation-langchain`), LM Studio provides both the chat LLM and the embeddings via its OpenAI-compatible API, and Phoenix runs locally. Gradio wires the UI together.

## Decomposition
N/A — narrow enough to build as one script.

## Evaluation Plan
1. Drop 3–5 `.txt` or `.md` files into a `docs/` subfolder
2. Launch the Gradio app (`uv run python app.py`)
3. Ask 3 questions in the left panel — including one clearly off-topic question
4. Click Refresh in the right panel — confirm spans appear
5. Click Run Evals — confirm relevance/completeness scores appear and that the off-topic question scores lower

## Learnings / Next Steps
[Filled in after validation]
