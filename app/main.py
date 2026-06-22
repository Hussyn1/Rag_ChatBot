import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import STATIC_DIR
from app.routers.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: auto-ingest documents if the vector store is empty
    print("=" * 60)
    print("RAG Chatbot Starting...")
    print("=" * 60)

    from app.rag.vectorstore import get_stats
    stats = get_stats()

    if stats["total_chunks"] == 0:
        print("Vector store is empty. Running initial document ingestion...")
        from app.rag.ingestion import ingest_all_documents
        result = ingest_all_documents()
        print(f"Initial ingestion: {result['files_processed']} files, {result['total_chunks']} chunks")
    else:
        print(f"Vector store already populated: {stats['total_chunks']} chunks")

    print("=" * 60)
    print("RAG Chatbot Ready!")
    print("=" * 60)

    yield

    # Shutdown
    print("RAG Chatbot shutting down...")


app = FastAPI(
    title="RAG Chatbot",
    description="A RAG-powered chatbot for ML, Deep Learning, RAG, and Context Engineering",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the main chat interface."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
