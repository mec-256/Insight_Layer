import os
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def verify_cloud():
    print("--- ☁️ Cloud Verification Started ---")
    
    # 1. Test Supabase Storage
    try:
        print("1. Testing Supabase Storage...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to list buckets to verify key
        buckets = supabase.storage.list_buckets()
        print(f"✅ Supabase Storage Accessible. Buckets: {[b.name for b in buckets]}")
        
        # Verify 'documents' bucket exists
        if not any(b.name == 'documents' for b in buckets):
            print("❌ Error: 'documents' bucket not found. Please create it in Supabase.")
            return False
    except Exception as e:
        print(f"❌ Supabase Storage Error: {e}")
        return False

    # 2. Test pgvector
    try:
        print("\n2. Testing pgvector connection...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = PGVector(
            connection=DATABASE_URL,
            embeddings=embeddings,
            collection_name="test_collection",
            use_jsonb=True,
        )
        
        # Try a simple add/search
        test_doc = Document(page_content="The capital of France is Paris.", metadata={"test": True})
        vector_store.add_documents([test_doc])
        print("✅ Added test document to pgvector.")
        
        results = vector_store.similarity_search("What is the capital of France?", k=1)
        if results and "Paris" in results[0].page_content:
            print(f"✅ pgvector Search Success: {results[0].page_content}")
        else:
            print("❌ pgvector Search Failed to return correct answer.")
            return False
            
    except Exception as e:
        print(f"❌ pgvector Error: {e}")
        return False

    print("\n🎉 Cloud Migration Verified Successfully!")
    return True

if __name__ == "__main__":
    verify_cloud()
