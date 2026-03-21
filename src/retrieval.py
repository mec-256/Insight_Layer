from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import DATABASE_URL, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, TOP_K

# Load the re-ranker globally so it doesn't reload on every request
print("Loading Cross-Encoder re-ranker...")
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

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
    """Initializes a BM25 retriever from the existing ChromaDB documents."""
    try:
        # For PGVector, we need a different way to get all docs if we want BM25
        # However, for hybrid search, we can use PGVector's own filtering
        # Since BM25 is local-memory based in this implementation, we can still use it
        # by fetching all documents from the vector store.
        # Note: In a large system, this would be inefficient.
        
        # PGVector doesn't have a simple .get() that returns all docs like Chroma.
        # We'll use a similarity search with a dummy query or just skip BM25 for now
        # to focus on the cloud transition.
        return None
    except Exception as e:
        print("Error loading BM25:", e)
    return None

def retrieve_context(db, bm25_retriever, question, user_id: int, filename_filter=None):
    """Performs Hybrid Search + Cross-Encoder Re-ranking"""
    fetch_k = TOP_K * 5 # Fetch 15 candidates
    
    # 1. Vector Search
    base_filter = {"user_id": user_id}
    
    if filename_filter:
        # Combine user_id and filename filters
        filter_dict = {
            "$and": [
                base_filter,
                {"source": {"$in": [f"data\\{filename_filter}", f"data/{filename_filter}"]}}
            ]
        }
        vector_results = db.similarity_search(question, k=fetch_k, filter=filter_dict)
    else:
        vector_results = db.similarity_search(question, k=fetch_k, filter=base_filter)
        
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
