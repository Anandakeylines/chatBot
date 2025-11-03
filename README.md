
# RAG (PDF) with LangChain + OpenAI

A minimal, local-first Retrieval-Augmented Generation (RAG) app that lets you:
1) Ingest your PDFs into a vector DB (FAISS)
2) Chat with them using OpenAI models via LangChain
3) Run a simple Streamlit UI

## 1) Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Put your PDF files into the `data/` folder.

## 2) Ingest your PDFs
```bash
python ingest.py
```
This will create/update a FAISS index under `vectordb/`.

## 3) Run the chat UI
```bash
streamlit run app.py
```
Open the URL Streamlit shows (usually http://localhost:8501).

## Notes
- Embeddings: `text-embedding-3-small` (cheap + good). Change in `ingest.py`.
- Chat model: `gpt-4o-mini`. Change in `app.py`.
- Vector store: FAISS (local, file-based). You can swap to Chroma or a hosted store.
- If you add/remove PDFs later, run `python ingest.py` again.
