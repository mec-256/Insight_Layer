# InsightLayer - Enterprise RAG Application

An AI-powered document Q&A system that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded documents.

## Features

- **Document Upload** - PDF and TXT file support with automatic chunking
- **AI-Powered Q&A** - Ask questions in natural language, get answers with citations
- **Multi-User Auth** - Secure JWT-based authentication
- **Hybrid Search** - Vector similarity + keyword search + cross-encoder re-ranking
- **Cloud Storage** - Supabase pgvector for embeddings
- **Modern UI** - Dark theme with glassmorphism design

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11 |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Database | PostgreSQL + pgvector (Supabase) |
| Re-ranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| Frontend | HTML, CSS, JavaScript |

## Setup Instructions

### Prerequisites

- Python 3.11+
- Supabase account (free tier works)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone the repository

```bash
git clone https://github.com/mec-256/Insight_Layer.git
cd Insight_Layer
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
DATABASE_URL=your_supabase_database_url_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here
```

### 5. Set up Supabase

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Go to SQL Editor and run:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table (auto-created by LangChain)
-- The app will create the table automatically on first run
```

4. Get your connection string from Settings → Database → Connection String

### 6. Run the application

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open in browser

```
http://localhost:8000
```

## Project Structure

```
├── src/
│   ├── api.py          # FastAPI endpoints
│   ├── auth.py         # Authentication (JWT)
│   ├── config.py       # Configuration
│   ├── generation.py   # LLM integration (Groq)
│   ├── ingestion.py    # Document processing
│   └── retrieval.py    # Vector search & re-ranking
├── static/
│   └── index.html      # Frontend UI
├── data/               # Upload directory
├── requirements.txt    # Python dependencies
└── .env.example        # Environment template
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create new user |
| POST | `/auth/login` | User login |
| GET | `/auth/me` | Get current user |
| POST | `/upload` | Upload document |
| POST | `/ask` | Ask a question |

## Screenshots

### Login Page
![Login](screenshots/login.png)

### Chat Interface
![Chat](screenshots/chat.png)

> **Note:** To add screenshots, create a `screenshots/` folder in the repo and add your images.

## License

MIT

## Author

[Your Name] - [Your GitHub](https://github.com/mec-256)
