import os
import psycopg2
from psycopg2 import extras
import jwt
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# Constants
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Use a warning for development, but in a real production app we'd raise an error
    SECRET_KEY = "INSIGHT_LAYER_DEV_SECRET"
    print("⚠️ WARNING: SECRET_KEY not found in environment. Using insecure default for development.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # 1 hour for better security

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# --- DB Utilities ---

def init_db():
    if not DATABASE_URL:
        print("Warning: DATABASE_URL not set. Skipping DB init.")
        return
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def get_user(username: str):
    if not DATABASE_URL:
        return None
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("SELECT id, username, hashed_password, full_name FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def create_user(username: str, password: str, full_name: Optional[str] = None):
    if not DATABASE_URL:
        return False
    hashed_password = pwd_context.hash(password)
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, hashed_password, full_name) VALUES (%s, %s, %s)", 
                       (username, hashed_password, full_name))
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# --- Auth Utilities ---

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except Exception:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Initialize the db on import
init_db()

# Security Check: Ensure SECRET_KEY is not the default in production environments
if SECRET_KEY == "INSIGHT_LAYER_DEV_SECRET":
    print("⚠️ WARNING: Using default development SECRET_KEY. Please set a strong SECRET_KEY in your .env for production.")
