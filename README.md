# NeuralQuery — RAG Chatbot

An AI-powered knowledge assistant for Machine Learning, Deep Learning, RAG, and Context Engineering. Built with FastAPI, ChromaDB, Groq API, and a premium glassmorphic frontend.

## Features

- 🧠 **RAG Pipeline**: Retrieves relevant context from your documents before generating answers
- ⚡ **Groq API**: Ultra-fast LLM inference with Llama 3.3 70B
- 🔍 **Semantic Search**: ChromaDB with sentence-transformers embeddings
- 💬 **Streaming Responses**: Real-time token-by-token display
- 📝 **Multi-Turn Chat**: Maintains conversation history
- 📄 **Source Citations**: Shows which documents informed each answer
- 🎨 **Premium UI**: Dark glassmorphic design with animations

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd rag-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Groq API key
# Get one free at https://console.groq.com
```

### 3. Run Locally

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser. On first startup, the app will automatically ingest all documents from the `documents/` folder.

### 4. Add Your Own Documents

Place PDF, Markdown, or text files in the `documents/` folder, then restart the app or call:

```bash
curl -X POST http://localhost:8000/api/ingest
```

## Deploy to Railway

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Railway

1. Go to [railway.app](https://railway.app) and create a new project
2. Select "Deploy from GitHub repo"
3. Connect your repository
4. Add environment variable: `GROQ_API_KEY` = your key
5. Railway will auto-detect the Dockerfile and deploy

### 3. Access Your Chatbot

Railway will provide a public URL like `https://your-app.up.railway.app`

## API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/` | GET | Chat UI |
| `/api/chat` | POST | Streaming chat (SSE) |
| `/api/chat/sync` | POST | Non-streaming chat |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Knowledge base stats |
| `/api/ingest` | POST | Re-ingest documents |

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py           # FastAPI app
│   ├── config.py          # Configuration
│   ├── rag/
│   │   ├── embeddings.py  # Sentence-transformers
│   │   ├── vectorstore.py # ChromaDB operations
│   │   ├── ingestion.py   # Document loading & chunking
│   │   └── chain.py       # RAG query chain
│   └── routers/
│       └── chat.py        # API endpoints
├── static/
│   ├── index.html         # Chat UI
│   ├── style.css          # Premium styles
│   └── script.js          # Frontend logic
├── documents/             # Your training docs
├── Dockerfile             # Docker build
├── railway.toml           # Railway config
└── requirements.txt       # Dependencies
```

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **LLM**: Groq API (Llama 3.3 70B)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Railway (Docker)
