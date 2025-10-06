#!/usr/bin/env python3
"""
Simple Streamlit UI for the Hotspot Paper RAG System
"""

import os
import textwrap
import streamlit as st

from rag_phase1 import RAGDataPipeline, ChromaDBManager
from rag_phase2 import LMStudioClient, RAGSystem


@st.cache_resource(show_spinner=False)
def get_rag_system(chroma_dir: str, data_path: str, lmstudio_base_url: str) -> RAGSystem:
    # Ensure vector store exists by running ingestion if empty
    db_manager = ChromaDBManager(persist_directory=chroma_dir)
    if db_manager.collection.count() == 0:
        pipeline = RAGDataPipeline(chroma_db_path=chroma_dir, data_path=data_path)
        # Default ingest the hotspot paper PDF if present
        default_doc = os.path.join(data_path, "Hotspot_Paper.pdf")
        if os.path.exists(default_doc):
            pipeline.ingest_documents(default_doc)
    llm_client = LMStudioClient(base_url=lmstudio_base_url)
    return RAGSystem(db_manager, llm_client)


def format_chunk_preview(content: str, max_chars: int = 300) -> str:
    if not content:
        return ""
    content = content.strip()
    return content[:max_chars] + ("..." if len(content) > max_chars else "")


def main():
    st.set_page_config(page_title="Hotspot Paper RAG", page_icon="🔎", layout="wide")
    st.title("🔎 Hotspot Paper RAG")
    st.caption("Ask questions about the Hotspot Paper using your local LM Studio model.")

    with st.sidebar:
        st.header("Settings")
        chroma_dir = st.text_input("Chroma directory", value="./hotspot_chroma_db")
        data_path = st.text_input("Data directory", value="./data")
        lmstudio_base_url = st.text_input("LM Studio base URL", value="http://127.0.0.1:1234/v1")

        st.subheader("Query Options")
        n_results = st.number_input("Top N results", min_value=1, max_value=10, value=3)
        retrieve_n = st.number_input("Retrieve N (rerank)", min_value=1, max_value=50, value=8)
        use_reranking = st.checkbox("Use reranking", value=True)
        stream_mode = st.checkbox("Stream answer", value=False)
        compress_chunks = st.checkbox("Compress chunks", value=False)
        analyze_query = st.checkbox("Analyze query", value=True)
        structured_output = st.checkbox("Structured output", value=False)
        verify_citations = st.checkbox("Verify citations", value=True)

        st.subheader("Safety")
        enforce_safety = st.checkbox("Enforce safety checks", value=True)
        min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.10, 0.01)
        min_docs = st.number_input("Min docs", min_value=1, max_value=5, value=1)
        min_unique_source_coverage = st.slider("Min unique source coverage", 0.0, 1.0, 0.67, 0.01)
        regenerate_on_fail = st.checkbox("Regenerate on safety failure", value=True)
        max_regenerations = st.number_input("Max regenerations", min_value=0, max_value=3, value=1)
        preview_chars = st.number_input("Preview chars", min_value=50, max_value=2000, value=300)

    rag = get_rag_system(chroma_dir, data_path, lmstudio_base_url)

    # Initialize session state for question text
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    if "auto_ask" not in st.session_state:
        st.session_state["auto_ask"] = False

    st.subheader("Suggested Questions")
    suggestions = [
        "What is the main research question in this hotspot paper?",
        "What methodology was used to detect hotspots?",
        "What are the key findings about hotspots?",
        "What datasets were used in this study?",
        "What are the limitations mentioned in the paper?",
        "What future work is suggested?",
        "What statistical methods were applied?",
    ]
    cols = st.columns(3)
    selected_suggestion = None
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q, key=f"suggest_{i}"):
            selected_suggestion = q
            # Persist into the input and auto-trigger ask
            st.session_state["question"] = q
            st.session_state["auto_ask"] = True
            # Rerun so the input reflects selection and auto ask kicks in
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

    st.subheader("Ask a question")
    # Bind directly to session_state without passing a separate value to avoid Streamlit warning
    user_query = st.text_input("Your question", key="question")
    ask = st.button("Ask")

    # Auto-ask if a suggestion was clicked
    if st.session_state.get("auto_ask") and not ask:
        ask = True

    if ask:
        question = st.session_state.get("question", user_query).strip() or user_query.strip()
        if not question:
            st.warning("Please enter a question or select a suggested one.")
            return
        try:
            # Streaming handling: show live tokens with a loading placeholder
            status_area = st.empty()
            live_area = st.empty()
            live_text = ""

            status_area.info("⏳ Loading...")

            def on_token(tok: str):
                nonlocal live_text
                if live_text == "":
                    # first token received -> clear loading
                    status_area.empty()
                live_text += tok
                live_area.markdown(live_text)

            result = rag.query(
                question,
                n_results=int(n_results),
                use_reranking=use_reranking,
                retrieve_n=int(retrieve_n),
                stream=True,  # force streaming for UX
                compress_chunks=compress_chunks,
                structured_output=structured_output,
                analyze_query=analyze_query,
                verify_citations=verify_citations,
                enforce_safety=enforce_safety,
                min_similarity=float(min_similarity),
                min_docs=int(min_docs),
                min_unique_source_coverage=float(min_unique_source_coverage),
                regenerate_on_fail=regenerate_on_fail,
                max_regenerations=int(max_regenerations),
                preview_chars=int(preview_chars),
                on_token=on_token,
            )
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.session_state["auto_ask"] = False
            return
        finally:
            st.session_state["auto_ask"] = False

        st.markdown("### Answer")
        st.write(result.get("answer", live_text))

        citations = result.get("citations", [])
        if citations:
            st.markdown("### Sources")
            st.write(", ".join(citations))

        # Show retrieved chunks table
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            st.markdown("### Retrieved Chunks")
            for i, ch in enumerate(chunks, start=1):
                meta = ch.get("metadata", {})
                st.markdown(f"**{i}. {meta.get('source', 'Unknown')}** | id={ch.get('id', 'N/A')} | similarity={ch.get('similarity', 0.0):.3f}")
                st.code(format_chunk_preview(ch.get("content", ""), int(preview_chars)))


if __name__ == "__main__":
    main()


