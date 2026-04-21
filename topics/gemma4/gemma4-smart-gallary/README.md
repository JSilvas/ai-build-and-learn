# Gemma 4 Smart Gallery

A vision-powered photo search app built with **Gemma 4** on **Vertex AI**. Point it at a folder of images and it will generate natural language descriptions of every photo, then let you search them by any keyword or phrase — no tags, no filenames, just Gemma's multimodal vision.

Every API call runs as a tracked task in the **Flyte TUI**, so you can watch each image being processed in real time.

---

## What it does

| Feature | How it works |
|---|---|
| **Generate Descriptions** | Gemma 4 analyzes each image and writes a natural language description. Results are cached in SQLite so re-runs are instant. |
| **Search** | Gemma 4 visually inspects each image against your query in real time and returns matches. Live progress shown per image. |

---

## Architecture

```
app.py              Gradio UI — event wiring only, no business logic
workflows.py        Flyte tasks and workflow runners
vision_service.py   Prompt logic — describe and match
gemma_client.py     Vertex AI SDK connection (swap here to change model provider)
db.py               SQLite cache for image descriptions
ui_components.py    HTML builders for the Gradio UI
styles.css          All visual styles as CSS custom properties
agent.py            CLI entry point (bypasses Gradio)
```

Each image is submitted as a discrete `flyte.run()` call, so every image appears as its own task in the Flyte TUI.

---

## Requirements

- Python 3.10+
- A GCP project with **Vertex AI API** enabled
- Gemma 4 access via **Vertex AI Model Garden** (MaaS — no deployment needed)
- `gcloud` CLI installed and authenticated

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/johndell-914/ai-build-and-learn.git
cd ai-build-and-learn/topics/gemma4/gemma4-smart-gallary
pip install -r requirements.txt
```

### 2. Authenticate with GCP

```bash
gcloud auth application-default login
```

### 3. Enable Gemma 4 on Vertex AI Model Garden

1. Go to [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) in your GCP project
2. Search for **Gemma 4**
3. Open **Gemma 4 26B A4B IT** and click **Enable API**

### 4. Create a `.env` file

Copy the example and fill in your GCP project ID:

```bash
cp .env.example .env
```

```env
GCP_PROJECT=your-gcp-project-id
GCP_REGION=global
GEMMA_MODEL=google/gemma-4-26b-a4b-it-maas
```

---

## Running the app

Open two terminals from the project directory.

**Terminal 1 — Flyte TUI** (watch tasks execute):
```bash
flyte start tui
```

**Terminal 2 — Gradio UI**:
```bash
python app.py
```

Open the URL printed by Gradio (usually `http://127.0.0.1:7860`).

---

## Using the UI

1. Click **Browse** to select a folder of images
2. Click **Generate Descriptions** to analyze and cache descriptions for every image
3. Type a query (e.g. `ocean`, `dog at the beach`, `sunset`) and click **Search**

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`

---

## CLI usage

You can also run workflows directly from the terminal without the Gradio UI:

```bash
# Generate descriptions for all images in a folder
python agent.py describe --folder ./images

# Search images by keyword
python agent.py search --folder ./images --query "ocean"
```

---

## Caching

Descriptions are stored in `gemma_photos.db` (SQLite, local). Re-running **Generate Descriptions** on the same folder will overwrite existing entries. The database file is excluded from version control.

---

## Project structure

```
gemma4-smart-gallary/
├── app.py               Gradio UI
├── agent.py             CLI entry point
├── workflows.py         Flyte tasks and workflow runners
├── vision_service.py    Prompt logic
├── gemma_client.py      Vertex AI / google-genai SDK wrapper
├── db.py                SQLite cache
├── ui_components.py     HTML component builders
├── styles.css           UI styles
├── requirements.txt     Python dependencies
├── .env.example         Environment variable template
└── RESEARCH.md          Design decisions and research log
```

---

## Key dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Google's modern SDK for Vertex AI (replaces `google-cloud-aiplatform`) |
| `flyte[tui]` | Workflow orchestration with terminal task viewer |
| `gradio` | Web UI |
| `python-dotenv` | `.env` file loading |
