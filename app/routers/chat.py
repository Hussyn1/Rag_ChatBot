import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.chain import chat
from app.rag.vectorstore import get_stats
from app.rag.ingestion import ingest_all_documents

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    content: str
    sources: list[dict]


@router.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "service": "rag-chatbot"}


@router.get("/stats")
async def stats():
    """Get statistics about the knowledge base."""
    try:
        store_stats = get_stats()
        return {
            "status": "ok",
            **store_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest():
    """Trigger document ingestion from the documents directory."""
    try:
        result = ingest_all_documents()
        return {
            "status": "ok",
            "message": "Ingestion complete",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with streaming response.
    Streams tokens as Server-Sent Events (SSE).
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    def event_stream():
        try:
            for chunk in chat(
                message=request.message,
                history=request.history,
                stream=True,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except ValueError as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
        except Exception as e:
            error_data = {
                "type": "error",
                "content": f"An error occurred: {str(e)}",
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/sync")
async def chat_sync_endpoint(request: ChatRequest):
    """Non-streaming chat endpoint for testing."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        result = chat(
            message=request.message,
            history=request.history,
            stream=False,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
