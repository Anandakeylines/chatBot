import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---- Paths & Models ----
INDEX_PATH = os.path.join(os.path.dirname(__file__), "vectordb", "faiss_index")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

st.set_page_config(page_title="RAG: Chat with your PDFs", page_icon="📄")
st.title("📄🔎 RAG: Chat with your PDFs")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    st.caption("All local except OpenAI API calls.")
    st.write("**Vector store:** FAISS")
    st.write("**Embed model:**", EMBED_MODEL)
    st.write("**Chat model:**", CHAT_MODEL)
    st.info("If you add PDFs, run `python ingest.py` to rebuild the index.")

def load_retriever():
    """Load FAISS retriever (expects /vectordb/faiss_index/{index.faiss,index.pkl})."""
    index_dir = INDEX_PATH
    faiss_file = os.path.join(index_dir, "index.faiss")
    pkl_file = os.path.join(index_dir, "index.pkl")

    if not (os.path.isdir(index_dir) and os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        st.warning("Vector index not found. Put PDFs in `data/` and run `python ingest.py`.")
        return None

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="similarity", k=4)

retriever = load_retriever()
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "n/a")
        parts.append(f"[{i}] Source: {src} (page {page})\n{d.page_content}")
    return "\n\n".join(parts[:6])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
     "If the answer isn't in the context, say you don't know and suggest where to look in the documents.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

def make_chain():
    # Use itemgetter to route only the needed fields into each prompt slot
    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

# ---- Chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of tuples: ("user"/"assistant", text)

# Render past messages
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# ---- Chat input ----
user_q = st.chat_input("Ask a question about your PDFs...")
if user_q and retriever is not None:
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.chat_history.append(("user", user_q))

    chain = make_chain()
    # Convert history to MessagesPlaceholder format
    lc_history = []
    for role, content in st.session_state.chat_history[:-1]:
        lc_history.append(("human", content) if role == "user" else ("ai", content))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke({"question": user_q, "chat_history": lc_history})
            except Exception as e:
                answer = f"Error while generating answer: {e}"
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))

    # ---- Sources ----
    with st.expander("View retrieved chunks (context)"):
        try:
            # In LC 0.2, retrievers are Runnables; use invoke()
            docs = retriever.invoke(user_q)
            for i, d in enumerate(docs, 1):
                src = os.path.basename(d.metadata.get("source", "unknown"))
                page = d.metadata.get("page", "n/a")
                st.markdown(f"**[{i}] {src} — page {page}**")
                st.code(d.page_content[:1200])
        except Exception as e:
            st.warning(f"Could not display sources: {e}")
else:
    if retriever is None:
        st.info("Add PDFs to `data/` and run `python ingest.py` first.")
