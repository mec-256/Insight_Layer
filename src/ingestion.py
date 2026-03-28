import os
import sys
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_DIR, EMBEDDING_MODEL_NAME, DATABASE_URL


def load_documents():
    print(f"Loading documents from {DATA_DIR}...")
    documents = []
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    text_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents.extend(text_loader.load())
    print(f"Loaded {len(documents)} total documents.")
    return documents


def split_documents(documents):
    # Advanced Semantic Chunking
    # Prioritizes splitting by double newline (paragraphs),
    # then single newline, then periods (sentences), then spaces (words).
    for doc in documents:
        # Optimize metadata for cloud storage
        if "source" in doc.metadata:
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    print(f"Split into {len(chunks)} semantically-aware chunks.")
    return chunks


def save_to_supabase(chunks):
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Fix for SQLAlchemy/PGVector: postgres:// -> postgresql://
    conn_str = DATABASE_URL
    if conn_str and conn_str.startswith("postgres://"):
        conn_str = conn_str.replace("postgres://", "postgresql://", 1)

    print("Saving to Supabase (pgvector)...")

    # PGVector will create the tables automatically
    vector_store = PGVector(
        connection=conn_str,
        embeddings=embeddings,
        collection_name="insight_layer_docs",
        use_jsonb=True,
    )

    vector_store.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to Supabase cloud.")


def main():
    # This main() is for local testing/bulk ingestion if needed
    docs = load_documents()
    if not docs:
        print(f"No documents found to ingest locally.")
        return

    chunks = split_documents(docs)
    save_to_supabase(chunks)


if __name__ == "__main__":
    main()
