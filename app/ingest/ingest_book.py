# ingest/ingest_book.py
import os
import uuid
import json
from typing import Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_ROOT = "storage/chroma_db"
UPLOADS_ROOT = "storage/uploads"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # MUST match retrieval

def ingest_book(pdf_path: str, title: str = None, book_id: str | None = None) -> Tuple[str, int]:
    """
    Ingest a PDF and persist embeddings to Chroma.

    Returns: (book_id, num_chunks)
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")

    # create ids and dirs
    if book_id is None:
        book_id = str(uuid.uuid4())

    persist_directory = os.path.join(CHROMA_ROOT, book_id)
    os.makedirs(persist_directory, exist_ok=True)
    os.makedirs(UPLOADS_ROOT, exist_ok=True)

    # optionally copy file to uploads folder for provenance (not required)
    filename = os.path.basename(pdf_path)
    saved_path = os.path.join(UPLOADS_ROOT, f"{book_id}--{filename}")
    if not os.path.exists(saved_path):
        # cheap copy
        with open(pdf_path, "rb") as src, open(saved_path, "wb") as dst:
            dst.write(src.read())

    # ---- load and split ----
    loader = PyPDFLoader(saved_path)
    documents = loader.load()  # list[Document]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)  # list[Document]

    if not chunks:
        raise RuntimeError("No chunks created from PDF. Check loader/splitter.")

    # ---- embeddings and vectorstore ----
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create / persist Chroma DB for this book
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist to disk (some Chroma wrappers require explicit persist)
    try:
        vector_store.persist()
    except Exception:
        # some versions persist automatically; ignore if method missing
        pass

    # ---- save metadata ----
    meta = {
        "book_id": book_id,
        "title": title or filename,
        "source_file": saved_path,
        "num_chunks": len(chunks),
        "embedding_model": EMBED_MODEL
    }
    with open(os.path.join(persist_directory, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"book_id": book_id, "num_chunks": len(chunks)}