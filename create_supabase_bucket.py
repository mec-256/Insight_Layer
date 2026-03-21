import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def create_bucket():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Creating 'documents' bucket...")
        # Create a public bucket
        supabase.storage.create_bucket('documents', options={'public': True})
        print("✅ Bucket 'documents' created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create bucket: {e}")
        return False

if __name__ == "__main__":
    create_bucket()
