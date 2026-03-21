import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def test_connection():
    if not DATABASE_URL or "[YOUR-PASSWORD]" in DATABASE_URL:
        print("❌ Error: Please replace '[YOUR-PASSWORD]' in your .env file with your actual Supabase database password.")
        return False
    
    try:
        print(f"Connecting to Supabase...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"✅ Success! Connected to: {db_version[0]}")
        
        # Test table creation
        print("Checking/Creating 'users' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                full_name TEXT
            )
        """)
        conn.commit()
        print("✅ 'users' table is ready in the cloud!")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
