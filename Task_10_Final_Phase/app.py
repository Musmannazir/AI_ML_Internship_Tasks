from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_engine import ContextAwareRAGChatbot, LocalVectorStore, RAGConfig


st.set_page_config(page_title="Task 10 - Context-Aware RAG Chatbot", page_icon="chat", layout="wide")
st.title("Task 10: Context-Aware Chatbot (RAG)")

with st.sidebar:
    st.header("Index Settings")
    corpus_dir = st.text_input("Corpus folder", "Task_10_Final_Phase/knowledge_base")
    index_path = st.text_input("Vector index path", "Task_10_Final_Phase/vector_store/index.npz")
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=8, value=4)

    if st.button("Build / Rebuild Index"):
        cfg = RAGConfig(top_k=top_k)
        store = LocalVectorStore.from_corpus(Path(corpus_dir), cfg)
        store.save(Path(index_path))
        st.success("Vector index built successfully")


def get_chatbot(idx_path: str, top_k_value: int) -> ContextAwareRAGChatbot:
    if "chatbot" not in st.session_state or st.session_state.get("index_path") != idx_path:
        store = LocalVectorStore.load(Path(idx_path))
        cfg = RAGConfig(top_k=top_k_value)
        st.session_state.chatbot = ContextAwareRAGChatbot(store=store, config=cfg)
        st.session_state.index_path = idx_path
        st.session_state.messages = []
    else:
        st.session_state.chatbot.config.top_k = top_k_value
    return st.session_state.chatbot


chatbot = None
try:
    chatbot = get_chatbot(index_path, top_k)
except FileNotFoundError:
    st.info("No vector index found yet. Build it from the sidebar to start chatting.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the indexed documents")
if question and chatbot is not None:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    result = chatbot.chat(question)
    answer = result["answer"]
    sources = result["sources"]

    source_lines = []
    for src in sources:
        source_lines.append(
            f"- source: {src['source']} | chunk: {src['chunk_id']} | score: {src['score']}"
        )

    full_answer = answer + "\n\nSources:\n" + "\n".join(source_lines)

    with st.chat_message("assistant"):
        st.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
