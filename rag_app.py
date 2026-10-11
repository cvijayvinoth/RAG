"""
Multi-PDF RAG app with sources.

Run:
    export OPENAI_API_KEY="your-key"
    streamlit run rag_app.py
"""

import os
import tempfile
from collections import Counter

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
dotenv.load_dotenv(override=True)  # must run before the first model call so the env vars are visible

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_DIR = "faiss_index"

PROMPT = ChatPromptTemplate.from_template(
    """You answer questions using ONLY the numbered context passages below.
Each passage is labelled with its file name and page.

Rules:
1. Cite every claim with the passage number in square brackets, e.g. [1] or [2][4].
2. If passages from DIFFERENT files give conflicting information, do not pick one
   silently. Start a section called "Disagreement" and state what each file says,
   with its citation, so the reader can judge.
3. If the context does not contain the answer, say you could not find it in the
   uploaded documents. Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""
)

def save_index(store):
    store.save_local(INDEX_DIR) # Save the FAISS index to the local directory

def load_index():
    if not os.path.isdir(INDEX_DIR):
        return None
    return FAISS.load_local(INDEX_DIR, OpenAIEmbeddings(model=EMBED_MODEL), allow_dangerous_deserialization=True)

# ---------------------------------------------------------------- indexing
def build_index(uploaded_files, chunk_size, chunk_overlap):
    """Load every PDF, split into chunks, embed all into ONE FAISS store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks, warnings = [], []

    for f in uploaded_files:
        # PyPDFLoader needs a path on disk
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()  # one Document per page
        finally:
            os.unlink(tmp_path)

        if not any(p.page_content.strip() for p in pages):
            warnings.append(f"{f.name}: no extractable text (scanned PDF?). Skipped.")
            continue

        for p in pages:
            p.metadata["source"] = f.name                       # real name, not temp path
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1  # 1-indexed for humans

        # Splitting happens page by page, so a chunk never spans two pages
        all_chunks.extend(splitter.split_documents(pages))

    if not all_chunks:
        return None, [], warnings

    store = FAISS.from_documents(all_chunks, OpenAIEmbeddings(model=EMBED_MODEL))
    return store, all_chunks, warnings


# ---------------------------------------------------------------- answering
def format_context(docs):
    return "\n\n".join(
        f"[{i}] (file: {d.metadata['source']}, page: {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def answer_question(store, question, k):
    # MMR pulls relevant-but-diverse chunks, so one file doesn't crowd out the rest
    retriever = store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4}
    )
    docs = retriever.invoke(question)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = PROMPT | llm
    reply = chain.invoke({"context": format_context(docs), "question": question})
    return reply.content, docs


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF RAG with Sources")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("Set the OPENAI_API_KEY environment variable and restart the app.")
    st.stop()

for key, default in [("store", None), ("chunk_count", 0), ("per_file", {}), ("history", [])]:
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("1. Upload & index")
    files = st.file_uploader("Select PDFs (multiple allowed)", type="pdf",
                             accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
    k = st.slider("Passages retrieved (k)", 2, 12, 6)

    if st.button("Build index", type="primary", disabled=not files):
        with st.spinner(f"Reading, chunking and embedding {len(files)} PDF(s)..."):
            try:
                store, chunks, warns = build_index(files, chunk_size, chunk_overlap)
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                store, chunks, warns = None, [], []
        for w in warns:
            st.warning(w)
        if store:
            st.session_state.store = store
            st.session_state.chunk_count = len(chunks)
            st.session_state.per_file = dict(Counter(c.metadata["source"] for c in chunks))
            st.session_state.history = []
            save_index(store)
            st.success(f"Indexed {len(chunks)} chunks from "
                       f"{len(st.session_state.per_file)} file(s).")
            
        if os.path.isdir(INDEX_DIR) and st.button("Load saved index"):
            st.info("Loading saved index from disk.")
        with st.spinner("Loading saved index..."):
            store = load_index()
        # Rebuild the counts from the stored chunks
        docs = list(store.docstore._dict.values())
        st.session_state.store = store
        st.session_state.chunk_count = len(docs)
        st.session_state.per_file = dict(Counter(d.metadata["source"] for d in docs))
        st.session_state.history = []
        st.success(f"Loaded {len(docs)} chunks from {len(st.session_state.per_file)} file(s).")


    if st.session_state.store:
        st.divider()
        st.caption(f"Index ready: **{st.session_state.chunk_count}** chunks")
        for name, n in st.session_state.per_file.items():
            st.caption(f"• {name}: {n} chunks")

st.header("2. Ask a question")

# Replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        with st.expander("Sources"):
            for i, s in enumerate(turn["sources"], 1):
                st.markdown(f"**[{i}] {s['file']} — page {s['page']}**")
                st.caption(s["text"])

question = st.chat_input("Ask something across your documents",
                         disabled=st.session_state.store is None)
if st.session_state.store is None:
    saved = load_index()
    if saved:
        st.session_state.store = saved
        docs = list(saved.docstore._dict.values())
        st.session_state.chunk_count = len(docs)
        st.session_state.per_file = dict(Counter(d.metadata["source"] for d in docs))
        st.session_state.history = []
        st.success(f"Loaded {len(docs)} chunks from {len(st.session_state.per_file)} file(s).")

    st.info("Upload PDFs in the sidebar and click **Build index** to begin.")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, docs = answer_question(st.session_state.store, question, k)
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()
        st.markdown(answer)

        sources = [
            {"file": d.metadata["source"], "page": d.metadata["page"],
             "text": d.page_content.strip()}
            for d in docs
        ]
        # Compact source summary: file -> sorted unique pages
        by_file = {}
        for s in sources:
            by_file.setdefault(s["file"], set()).add(s["page"])
        st.markdown("**Sources:** " + "; ".join(
            f"{f} (p. {', '.join(map(str, sorted(p)))})" for f, p in by_file.items()
        ))
        # If an answer leans on many pages, the chunking is probably too coarse/fine
        if sum(len(p) for p in by_file.values()) >= 5:
            st.caption("⚠️ This answer draws on 5+ different pages. Consider tuning "
                       "chunk size or k if the citations feel scattered.")

        with st.expander("Sources (passages used)"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**[{i}] {s['file']} — page {s['page']}**")
                st.caption(s["text"])

    st.session_state.history.append({"q": question, "a": answer, "sources": sources})
