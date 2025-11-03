import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "vectordb")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def load_pdfs(path: str):
    docs = []
    for fname in os.listdir(path):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, fname))
            docs.extend(loader.load())
    return docs

def main():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"data directory not found: {DATA_DIR}")

    print("Loading PDFs...")
    docs = load_pdfs(DATA_DIR)
    if not docs:
        print("No PDFs found in data/. Add some and run again.")
        return

    print(f"Loaded {len(docs)} pages. Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks. Embedding with {EMBED_MODEL}...")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = FAISS.from_documents(splits, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    save_path = os.path.join(INDEX_DIR, "faiss_index")
    vectordb.save_local(save_path)
    print(f"Saved FAISS index to {save_path}")

if __name__ == "__main__":
    main()
