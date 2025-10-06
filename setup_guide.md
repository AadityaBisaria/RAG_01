# Hotspot Paper RAG – Setup Guide

## 1) Prerequisites

- Python 3.10+
- GPU optional (embeddings run faster with GPU-enabled PyTorch)
- LM Studio installed and a chat server running
  - Open LM Studio → Load a chat model (e.g., Hermes/Llama family) → Start local server
  - Default URL used here: `http://127.0.0.1:1234/v1`

## 2) Install

```bash
# from project root
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Data

- Put the paper at `./data/Hotspot_Paper.pdf` (already present in this repo).
- You can add more docs; ingestion supports `.pdf`, `.md`, `.txt`.

## 4) Build the Vector Store (ChromaDB)

Two options:

- Quick script (includes ingestion + Q&A demo):

```bash
python run_hotspot_rag.py
```

- Or run Phase 1 standalone:

```bash
python rag_phase1.py
```

This creates a persistent Chroma database at `./hotspot_chroma_db`.

## 5) Verify LM Studio connectivity

```bash
python check_model.py
```

This should print the active model and confirm connectivity.

## 6) Run the CLI demo

```bash
python run_hotspot_rag.py
```

You’ll see retrieval logs, chunk previews, and answers with citations.

## 7) Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

- Sidebar lets you tune retrieval/reranking and safety.
- Suggested questions are clickable.
- The answer streams live; retrieved chunks and sources are shown.

## 8) Common issues

- “Couldn’t find relevant information”: ensure ingestion ran and `./hotspot_chroma_db` exists; lower min similarity in UI.
- LM Studio not responding: verify server URL and that a model is loaded.
- PDF extraction is empty: try a different PDF extractor or OCR (not included).

## 9) Useful paths

- Chroma DB: `./hotspot_chroma_db`
- Data folder: `./data`
- Streamlit app: `streamlit_app.py`
- RAG core:
  - Phase 1 (ingestion/embeddings): `rag_phase1.py`
  - Phase 2 (LLM + guardrails): `rag_phase2.py`

## 10) Environment variables (optional)

If you change LM Studio server:

```bash
# Example
set LMSTUDIO_URL=http://localhost:1234/v1
```

Then pass it to `LMStudioClient` (or edit the Streamlit UI sidebar).
