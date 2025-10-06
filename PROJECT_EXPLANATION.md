# Project Explanation – Hotspot Paper RAG



## Chunking and overlap (~10–15%)

- We split documents with a recursive splitter at ~1000 characters and overlap of ~200 characters (≈20%, safely covering the 10–15% goal).
- Why overlap: preserves context that straddles boundaries (e.g., a sentence starting in one chunk and ending in the next). Without overlap, retrieval can surface partial facts and increase hallucination risk.
- Trade‑off: small index growth and slightly more tokens retrieved, but significantly better recall and grounding.

## Embedding model choice (BAAI/bge-large-en-v1.5)

- BGE Large performs strongly on English retrieval (MTEB), is robust on noisy technical PDFs, and integrates easily via `sentence-transformers`.
- It provides high‑quality semantic vectors that make first‑stage recall strong before reranking.

## LLM and LM Studio

- We use LM Studio’s local Chat Completions API (`http://localhost:1234/v1`).
- Benefits: privacy (local-only), zero per‑token cloud cost, easy swapping of models (e.g., Hermes/Llama variants) without code changes.
- In code: `LMStudioClient` handles non‑stream and stream modes.

## Retrieval → Reranking (top X → top N)

- Stage 1: retrieve `retrieve_n` (e.g., 8–10) chunks by vector similarity from Chroma.
- Stage 2: rerank with a cross‑encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring full query–chunk pairs.
- Keep the top `n_results` (e.g., 2–5) for prompting. This corrects embedding‑only errors and improves precision.


## Why Chroma and what it stores

- Chroma is a vector database (not a simple KV store) with indexing, filtering, and persistence.
- Each item includes: `id`, `embedding`, original `document` (chunk text), and `metadata` (e.g., filename, chunk_id). We use these to show chunk previews and cite sources.

## Automatic citations (filename + chunk trail)

- Prompt instructions require per‑sentence citations in the form `[Source: filename]`.
- Post‑generation, we:
  - Extract citations, map any "Document N" to real filenames,
  - Verify cited sources were truly among the retrieved chunks,
  - Compute unique‑source coverage, and
  - Report/stream the exact chunks (source, id, similarity) used.
- We also strip “Document N” text from the final prose to keep clean filename citations.

## Chunk compression (optional)

- For long contexts, we summarize chunks to reduce tokens while preserving key facts.
- This boosts effective context window and lowers latency/cost, while our grounding checks (citations/coverage/overlap) maintain quality.

## Guardrails (brief)

- Refusal policy embedded in prompts for unsafe content.
- Retrieval sanity: require ≥1 doc and a minimum similarity; otherwise answer “not found.”
- Post‑gen: per‑sentence citations, unique‑source coverage threshold, quick n‑gram overlap of sentence vs cited chunk.
- If checks fail, optionally regenerate with stricter settings (lower temperature).

## Tunables

- Chunking: `chunk_size`, `chunk_overlap`.
- Retrieval: `retrieve_n` (X), `n_results` (top N), similarity threshold.
- Reranking: on/off, cross‑encoder model.
- Safety: coverage threshold, regeneration policy.
- UI: streaming on/off, preview length (Streamlit sidebar).
