import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import DATABASE_URL, EMBEDDING_MODEL_NAME, TOP_K

db_url = DATABASE_URL
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

embeddings = None
cross_encoder = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings


def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            from src.config import RERANKER_MODEL_NAME

            print("Loading Cross-Encoder re-ranker...")
            cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as e:
            print(f"WARNING: Could not load cross-encoder: {e}")
            return None
    return cross_encoder


def load_db():
    from langchain_postgres.vectorstores import PGVector

    vector_store = PGVector(
        connection=db_url,
        embeddings=get_embeddings(),
        collection_name="insight_layer_docs",
        use_jsonb=True,
    )
    return vector_store


def load_bm25_retriever(db):
    """BM25 is disabled in cloud mode."""
    return None


def retrieve_context(db, bm25_retriever, question, user_id: int, filename_filter=None):
    fetch_k = TOP_K * 5

    filter_dict = {"user_id": user_id}
    if filename_filter:
        filter_dict["source"] = filename_filter

    vector_results = db.similarity_search(question, k=fetch_k, filter=filter_dict)

    keyword_results = []
    if bm25_retriever:
        try:
            k_res = [
                doc
                for doc in bm25_retriever.invoke(question)
                if doc.metadata.get("user_id") == user_id
            ]
            if filename_filter:
                keyword_results = [
                    d
                    for d in k_res
                    if d.metadata.get("source", "").endswith(filename_filter)
                ]
            else:
                keyword_results = k_res
        except Exception as e:
            print("BM25 error:", e)

    combined = {}
    for doc in vector_results + keyword_results:
        combined[doc.page_content] = doc

    candidates = list(combined.values())
    if not candidates:
        return []

    # Try re-ranking, fall back to vector search results if fails
    ce = get_cross_encoder()
    if ce:
        try:
            pairs = [[question, doc.page_content] for doc in candidates]
            scores = ce.predict(pairs)
            scored = list(zip(scores, candidates))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:TOP_K]]
        except Exception as e:
            print(f"Re-ranking failed, using vector results: {e}")

    return candidates[:TOP_K]
