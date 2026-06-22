import chromadb
from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from app.rag.embeddings import embed_texts, embed_query

_client = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    """Get or initialize the ChromaDB persistent client (singleton)."""
    global _client
    if _client is None:
        print(f"Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        print("ChromaDB initialized successfully.")
    return _client


def get_collection():
    """Get or create the knowledge base collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Collection '{COLLECTION_NAME}' ready. Document count: {_collection.count()}")
    return _collection


def add_documents(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    """Add document chunks to the vector store with their embeddings."""
    collection = get_collection()
    embeddings = embed_texts(documents)
    
    # ChromaDB has a batch limit, so we chunk the additions
    batch_size = 100
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_embeds = embeddings[i : i + batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
            embeddings=batch_embeds,
        )
        total_added += len(batch_docs)
    
    print(f"Added {total_added} chunks to the vector store.")
    return total_added


def query(text: str, n_results: int = 5) -> dict:
    """Query the vector store for the most relevant chunks."""
    collection = get_collection()
    
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    query_embedding = embed_query(text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    
    return results


def get_stats() -> dict:
    """Get statistics about the vector store."""
    collection = get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
    }


def clear_collection():
    """Delete and recreate the collection."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    return get_collection()
