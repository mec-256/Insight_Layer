import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import DATA_DIR, CHROMA_PATH, EMBEDDING_MODEL_NAME

def load_documents():
    print(f"Loading documents from {DATA_DIR}...")
    documents=[]
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())
    
    text_loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
        )
    documents.extend(text_loader.load())
    print(f"Loaded {len(documents)} total documents.")
    return documents
    
def split_documents(documents):
    # Advanced Semantic Chunking
    # Prioritizes splitting by double newline (paragraphs), 
    # then single newline, then periods (sentences), then spaces (words).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]=f"chunk_{i}"
        
    print(f"Split into {len(chunks)} semantically-aware chunks.")
    return chunks
    
def save_to_chroma(chunks):
    print("initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("saving to chroma")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH 
        )    
   
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    docs=load_documents()
    if not docs:
         print(f"No documents found to ingest! Please add PDFs or TXT files to '{DATA_DIR}'.")
         return 
    
    chunks = split_documents(docs)
    save_to_chroma(chunks)
    
if __name__ == "__main__":
    main()
