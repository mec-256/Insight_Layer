import sys
import os
import shutil
from supabase import create_client, Client
# Ensure absolute imports work from src module
sys.path.append(os.path.dirname(__file__))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import DATA_DIR
from ingestion import split_documents
from retrieval import load_db, load_bm25_retriever, retrieve_context
from generation import build_prompt, ask_groq

import auth
from fastapi.security import OAuth2PasswordRequestForm
from config import SUPABASE_URL, SUPABASE_KEY

# --- Supabase Client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Preload DB once at startup ---
db = None
bm25 = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, bm25
    print("Loading ChromaDB and embedding model...")
    db = load_db()
    bm25 = load_bm25_retriever(db)
    print("Ready!")
    yield
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

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
async def signup(user: UserSignup):
    success = auth.create_user(user.username, user.password, user.full_name)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created successfully"}

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.get_user(form_data.username)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = auth.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me")
async def read_users_me(current_user: dict = Depends(auth.get_current_user)):
    return {"id": current_user["id"], "username": current_user["username"], "full_name": current_user["full_name"]}

# --- Ask endpoint ---

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest, current_user: dict = Depends(auth.get_current_user)):
    """Receives a question and returns an answer strictly from the specified user's documents."""
    results = retrieve_context(db, bm25, request.question, user_id=current_user["id"], filename_filter=request.filename)
    prompt = build_prompt(request.question, results, chat_history=request.chat_history)
    answer = ask_groq(prompt)
    
    # Format sources for the frontend UI chips
    formatted_sources = set()
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        filename = source.split('\\')[-1].split('/')[-1]
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
        res = supabase.storage.from_("documents").download(f"{user_id}/{filename}")
        
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
            "message": f"✅ '{filename}' uploaded and added to knowledge base."
        }
        print(f"--- BACKGROUND TASK: Finished processing '{filename}' ! ---\n")
    except Exception as e:
        print(f"\n--- BACKGROUND TASK: Error processing '{filename}': {str(e)} ---\n")
        upload_status[filename] = {"status": "error", "detail": str(e)}

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    current_user: dict = Depends(auth.get_current_user)
):
    """Upload a PDF or TXT file and queue it for async processing."""
    
    # Validate file type
    filename = file.filename
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Upload to Supabase Storage immediately
    # We use user-specific folders: documents/{user_id}/{filename}
    try:
        file_content = await file.read()
        supabase.storage.from_("documents").upload(
            path=f"{current_user['id']}/{filename}",
            file=file_content,
            file_options={"upsert": "true"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft upload failed: {str(e)}")
        
    # Mark as processing
    upload_status[filename] = {"status": "processing"}

    # Queue background task
    background_tasks.add_task(process_document, filename, current_user["id"])

    return {
        "message": f"⏳ '{filename}' is being uploaded to cloud and processed.",
        "status": "processing"
    }

@app.get("/upload/status/{filename}")
async def get_upload_status(filename: str, current_user: dict = Depends(auth.get_current_user)):
    """Check the processing status of an uploaded file."""
    if filename not in upload_status:
        raise HTTPException(status_code=404, detail="File processing not found.")
    return upload_status[filename]
