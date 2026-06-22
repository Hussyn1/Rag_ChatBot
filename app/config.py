import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
STATIC_DIR = str(BASE_DIR / "static")

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ORG_ID = os.getenv("GROQ_ORG_ID", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Chat
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))

# Collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
