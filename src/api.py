import sys
import os
import shutil
import re
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import DATA_DIR, SUPABASE_URL, SUPABASE_KEY, DATABASE_URL
from ingestion import split_documents
from retrieval import load_db, load_bm25_retriever, retrieve_context
from generation import build_prompt, ask_groq

import auth
from fastapi.security import OAuth2PasswordRequestForm

# --- Supabase Client (lazy initialization) ---
supabase: Client = None


def get_supabase():
    global supabase
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


# --- Preload DB once at startup ---
db = None
bm25 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, bm25
    os.makedirs(DATA_DIR, exist_ok=True)

    if not SUPABASE_URL or not DATABASE_URL:
        print("WARNING: Supabase/Database not configured. Some features may not work.")
    else:
        print("Loading Cloud Vector DB (pgvector)...")
        try:
            db = load_db()
            bm25 = load_bm25_retriever(db)
            print("Ready!")
        except Exception as e:
            print(f"WARNING: Failed to load vector DB: {e}")
            db = None
            bm25 = None

    yield
    print("Shutting down.")


# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Middleware ---
# In production, replace ["*"] with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (use absolute path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --- Pydantic models ---


class Message(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    question: str
    filename: Optional[str] = None
    chat_history: list[Message] = []


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


class UserSignup(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


# --- Auth Endpoints ---


@app.post("/auth/signup")
@limiter.limit("5/minute")
async def signup(request: Request, user: UserSignup):
    success = auth.create_user(user.username, user.password, user.full_name)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created successfully"}


@app.post("/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.get_user(form_data.username)
    if not user or not auth.verify_password(
        form_data.password, user["hashed_password"]
    ):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = auth.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/supabase-login")
async def supabase_login(data: dict):
    """
    Experimental: Endpoint to handle logins from 3rd party providers (Google/GitHub)
    after they are verified by Supabase on the frontend.
    """
    supabase_uid = data.get("uid")
    email = data.get("email")
    if not supabase_uid or not email:
        raise HTTPException(status_code=400, detail="Invalid Supabase user data")

    # Check if user exists, if not create them
    username = email.split("@")[0]
    user = auth.get_user(username)
    if not user:
        # Create a placeholder user for 3rd party auth
        auth.create_user(username, os.urandom(16).hex(), email)
        user = auth.get_user(username)

    access_token = auth.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me")
async def read_users_me(current_user: dict = Depends(auth.get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "full_name": current_user["full_name"],
    }


# --- Ask endpoint ---


@app.post("/ask", response_model=AnswerResponse)
async def ask(
    request: QuestionRequest, current_user: dict = Depends(auth.get_current_user)
):
    """Receives a question and returns an answer strictly from the specified user's documents."""

    print(f"Received question: {request.question[:100]}")

    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database not configured. Please contact support.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please provide a text question.")

    # Reject if question looks like an image filename
    lower_q = question.lower()
    if any(
        lower_q.endswith(ext)
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
    ):
        return AnswerResponse(
            answer="I can only process text questions, not images. Please type your question.",
            sources=[],
        )

    results = retrieve_context(
        db,
        bm25,
        question,
        user_id=current_user["id"],
        filename_filter=request.filename,
    )
    prompt = build_prompt(question, results, chat_history=request.chat_history)
    try:
        answer = ask_groq(prompt)
    except Exception as e:
        # Return friendly error instead of crashing
        return AnswerResponse(answer=f"Error: {str(e)}", sources=[])

    # Format sources for the frontend UI chips
    formatted_sources = set()
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source)  # Cross-platform safe
        page = doc.metadata.get("page")
        if page is not None:
            formatted_sources.add(f"{filename} (Page {int(page) + 1})")
        else:
            formatted_sources.add(filename)

    return AnswerResponse(answer=answer, sources=list(formatted_sources))


# --- Upload status tracking ---
upload_status = {}


def process_document(filename: str, user_id: int):
    global db, bm25
    print(f"\n--- BACKGROUND TASK: Started processing '{filename}' ---")
    try:
        # Download from Supabase Storage to a temporary local file for processing
        print(f"1/4. Downloading file '{filename}' from Supabase Storage...")
        res = (
            get_supabase().storage.from_("documents").download(f"{user_id}/{filename}")
        )

        # Save to a temporary path for the loaders to read
        temp_path = os.path.join(DATA_DIR, f"temp_{filename}")
        with open(temp_path, "wb") as f:
            f.write(res)

        print(f"     -> Loading file with right loader...")
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, autodetect_encoding=True)
        documents = loader.load()
        print(f"     -> Loaded {len(documents)} pages/documents.")

        # Split into chunks + add to the LIVE database
        print(f"2/4. Splitting document into chunks...")
        chunks = split_documents(documents)
        print(f"     -> Created {len(chunks)} chunks.")

        # Add user_id to metadata for isolation
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id

        print(f"3/4. Creating Vector Embeddings & saving to Supabase (pgvector)...")
        if db is None:
            raise Exception(
                "Database not initialized. Please check DATABASE_URL configuration."
            )
        db.add_documents(chunks)
        print(f"     -> Successfully saved to cloud database.")

        # Reload BM25 to include new documents
        print(f"4/4. Reloading Keyword Search (BM25) index...")
        bm25 = load_bm25_retriever(db)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        upload_status[filename] = {
            "status": "done",
            "chunks_added": len(chunks),
            "message": f"✅ '{filename}' uploaded and added to knowledge base.",
        }
        print(f"--- BACKGROUND TASK: Finished processing '{filename}' ! ---\n")
    except Exception as e:
        print(f"\n--- BACKGROUND TASK: Error processing '{filename}': {str(e)} ---\n")
        upload_status[filename] = {"status": "error", "detail": str(e)}


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(auth.get_current_user),
):
    """Upload a PDF or TXT file and queue it for async processing."""

    # Validate file type
    filename = file.filename
    # Sanitize filename: remove any path traversal characters or weird symbols
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(
            status_code=400, detail="Only PDF and TXT files are supported."
        )

    # Upload to Supabase Storage immediately
    # We use user-specific folders: documents/{user_id}/{filename}
    try:
        file_content = await file.read()
        get_supabase().storage.from_("documents").upload(
            path=f"{current_user['id']}/{filename}",
            file=file_content,
            file_options={"upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft upload failed: {str(e)}")

    # Mark as processing
    upload_status[filename] = {"status": "processing"}

    # Queue background task
    background_tasks.add_task(process_document, filename, current_user["id"])

    return {
        "message": f"⏳ '{filename}' is being uploaded to cloud and processed.",
        "status": "processing",
    }


@app.get("/upload/status/{filename}")
async def get_upload_status(
    filename: str, current_user: dict = Depends(auth.get_current_user)
):
    """Check the processing status of an uploaded file."""
    if filename not in upload_status:
        raise HTTPException(status_code=404, detail="File processing not found.")
    return upload_status[filename]
