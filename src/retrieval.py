from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import DATABASE_URL, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, TOP_K

# Load the re-ranker globally so it doesn't reload on every request
print("Loading Cross-Encoder re-ranker...")
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

# Fix for SQLAlchemy/PGVector: postgres:// -> postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = PGVector(
        connection=DATABASE_URL,
        embeddings=embeddings,
        collection_name="insight_layer_docs",
        use_jsonb=True,
    )
    return vector_store

def load_bm25_retriever(db):
    """BM25 is currently disabled in cloud mode to maintain statelessness and speed."""
    return None

def retrieve_context(db, bm25_retriever, question, user_id: int, filename_filter=None):
    """Performs Hybrid Search + Cross-Encoder Re-ranking"""
    fetch_k = TOP_K * 5 # Fetch 15 candidates
    
    # 1. Vector Search
    filter_dict = {"user_id": user_id}
    
    if filename_filter:
        # For PGVector, simple dictionary filters work best
        # Note: source should match the value saved in metadata
        filter_dict["source"] = filename_filter
        
    vector_results = db.similarity_search(question, k=fetch_k, filter=filter_dict)
        
    # 2. Keyword Search (BM25)
    keyword_results = []
    if bm25_retriever:
        try:
            # Filter BM25 results by user_id
            k_res = [doc for doc in bm25_retriever.invoke(question) if doc.metadata.get("user_id") == user_id]
            
            if filename_filter:
                keyword_results = [doc for doc in k_res if doc.metadata.get("source", "").endswith(filename_filter)]
            else:
                keyword_results = k_res
        except Exception as e:
            print("BM25 error:", e)
            
    # 3. Combine and Deduplicate
    combined = {}
    for doc in vector_results + keyword_results:
        combined[doc.page_content] = doc # deduplicate by content
        
    candidates = list(combined.values())
    if not candidates:
        return []
        
    # 4. Cross-Encoder Re-ranking
    # Create pairs of (Question, Document Content)
    pairs = [[question, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    # Sort candidates by score descending
    scored_candidates = list(zip(scores, candidates))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K
    results = [doc for score, doc in scored_candidates[:TOP_K]]
    return results
