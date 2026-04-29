"""
Streamlit app — AI Advanced High School Multi-Document RAG Search
Run with: streamlit run app.py
"""

import os
import streamlit as st
from rag_tool import MultiDocRetriever

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Advanced High School Search",
    page_icon="📚",
    layout="wide",
)

# ── Built-in documents ─────────────────────────────────────────────────────────

BUILTIN_DOCS = {
    "Student Handbook": "handbook.txt",
    "Course Catalog": "course_catalog.txt",
}

# ── Session state: retriever ───────────────────────────────────────────────────
# We store the retriever and the set of active sources in session state so the
# index is only rebuilt when the document selection actually changes.

if "retriever" not in st.session_state:
    st.session_state.retriever = MultiDocRetriever()
    st.session_state.active_sources = set()
    st.session_state.uploaded_texts = {}   # name -> raw text

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Documents")
    st.caption("Select which sources to search across.")

    selected = set()

    st.subheader("Built-in")
    for label, path in BUILTIN_DOCS.items():
        if os.path.exists(path):
            checked = st.checkbox(label, value=True, key="builtin_" + label)
            if checked:
                selected.add(label)

    uploaded_files = st.file_uploader(
        "Upload your own (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.subheader("Uploaded")
        for f in uploaded_files:
            text = f.read().decode("utf-8", errors="ignore")
            st.session_state.uploaded_texts[f.name] = text
            checked = st.checkbox(f.name, value=True, key="upload_" + f.name)
            if checked:
                selected.add(f.name)

    # Rebuild index only when selection changes
    if selected != st.session_state.active_sources:
        st.session_state.active_sources = selected
        r = MultiDocRetriever()

        for label, path in BUILTIN_DOCS.items():
            if label in selected and os.path.exists(path):
                r.load_document(path, source_name=label)

        for name, text in st.session_state.uploaded_texts.items():
            if name in selected:
                r.load_text(text, source_name=name)

        if r.total_chunks > 0:
            r.rebuild()

        st.session_state.retriever = r

    st.divider()
    st.subheader("Index stats")
    r = st.session_state.retriever
    st.metric("Documents loaded", len(r.sources_loaded))
    st.metric("Total chunks", r.total_chunks)
    st.metric("Vocabulary size", r.vocab_size)

    if len(r.sources_loaded) > 1:
        st.success("Multi-source mode active — searching across {} documents.".format(
            len(r.sources_loaded)
        ))
    elif len(r.sources_loaded) == 1:
        st.info("Single-source mode. Add more documents to compare quality.")
    else:
        st.warning("No documents selected.")

# ── Source badge colors ────────────────────────────────────────────────────────

BADGE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def badge(source_name, all_sources):
    idx = all_sources.index(source_name) % len(BADGE_COLORS)
    color = BADGE_COLORS[idx]
    return (
        "<span style='background:{color};color:white;padding:2px 8px;"
        "border-radius:4px;font-size:0.75rem;font-weight:600'>{name}</span>"
    ).format(color=color, name=source_name)

# ── Main ───────────────────────────────────────────────────────────────────────

st.title("📚 AI Advanced High School — Document Search")
st.caption(
    "Search across the Student Handbook, Course Catalog, or any uploaded document. "
    "Results are ranked by TF-IDF cosine similarity."
)

query = st.text_input("Ask a question or enter keywords:", placeholder="e.g. what happens if I miss too many classes")

top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

if query:
    r = st.session_state.retriever
    if r.total_chunks == 0:
        st.warning("Select at least one document in the sidebar before searching.")
    else:
        results = r.search(query, top_k=top_k)
        all_sources = r.sources_loaded

        # ── Quality metrics bar ────────────────────────────────────────────────
        top_score = results[0][0] if results else 0.0
        unique_sources = list(dict.fromkeys(src for _, _, src in results))

        col1, col2, col3 = st.columns(3)
        col1.metric("Top similarity score", "{:.3f}".format(top_score))
        col2.metric("Results returned", len(results))
        col3.metric(
            "Sources contributing",
            "{} / {}".format(len(unique_sources), len(all_sources)),
        )

        if len(unique_sources) > 1:
            st.success(
                "Results span **{}** documents — multi-source retrieval is active.".format(
                    len(unique_sources)
                )
            )

        st.divider()

        # ── Result cards ───────────────────────────────────────────────────────
        for rank, (score, chunk, source) in enumerate(results, start=1):
            with st.container():
                header_col, score_col = st.columns([3, 1])

                with header_col:
                    st.markdown(
                        "**Result {}** &nbsp; {}".format(rank, badge(source, all_sources)),
                        unsafe_allow_html=True,
                    )
                with score_col:
                    st.markdown(
                        "<p style='text-align:right;color:#555;font-size:0.85rem'>"
                        "similarity: <strong>{:.3f}</strong></p>".format(score),
                        unsafe_allow_html=True,
                    )

                # score bar
                st.progress(float(min(score, 1.0)))

                st.markdown(
                    "<div style='background:#f8f9fa;border-left:3px solid #ccc;"
                    "padding:10px 14px;border-radius:4px;font-size:0.9rem'>{}</div>".format(chunk),
                    unsafe_allow_html=True,
                )
                st.write("")

        # ── Source breakdown ───────────────────────────────────────────────────
        if len(all_sources) > 1:
            st.divider()
            st.subheader("Source breakdown")
            st.caption("How many of the top {} results came from each document.".format(top_k))

            counts = {}
            for _, _, src in results:
                counts[src] = counts.get(src, 0) + 1

            for src in all_sources:
                n = counts.get(src, 0)
                pct = n / len(results) if results else 0
                src_col, bar_col = st.columns([2, 5])
                src_col.markdown(
                    badge(src, all_sources), unsafe_allow_html=True
                )
                bar_col.progress(pct, text="{} / {} results".format(n, len(results)))
