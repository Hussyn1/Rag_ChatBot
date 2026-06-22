import json
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_ORG_ID, GROQ_MODEL, TOP_K
from app.rag.vectorstore import query as vector_query


def get_groq_client() -> Groq:
    """Initialize the Groq client."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your environment variables or .env file."
        )
    return Groq(api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """You are an intelligent AI assistant specialized in Machine Learning, Deep Learning, RAG (Retrieval-Augmented Generation), and Context Engineering. You answer questions based on the provided context documents.

## Instructions:
1. Answer the user's question using ONLY the information from the provided context documents.
2. If the context contains relevant information, provide a detailed, well-structured answer.
3. If the context does not contain enough information to answer the question, say: "I don't have enough information in my knowledge base to answer this question accurately."
4. Always cite which source document(s) you used in your answer.
5. Use clear formatting with headings, bullet points, and code blocks when appropriate.
6. Be concise but thorough. Don't include unnecessary filler.
7. If the user asks a follow-up question, use the conversation history to maintain context.

## Response Format:
- Use markdown formatting for clear readability
- Include source citations at the end as: **Sources:** [document names]
- Use code blocks for any code examples
- Use bullet points for lists"""


def build_context(query_text: str, top_k: int = TOP_K) -> tuple[str, list[dict]]:
    """
    Retrieve relevant context from the vector store and format it.
    Returns the formatted context string and source metadata.
    """
    results = vector_query(query_text, n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        return "", []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # Cosine distance: lower = more similar (ChromaDB returns distance, not similarity)
        similarity = 1 - dist  # Convert distance to similarity
        source_name = meta.get("source", "Unknown")

        context_parts.append(
            f"[Document {i + 1}: {source_name} (relevance: {similarity:.2f})]\n{doc}"
        )
        sources.append(
            {
                "source": source_name,
                "chunk_index": meta.get("chunk_index", 0),
                "relevance": round(similarity, 3),
            }
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def format_history(history: list[dict]) -> list[dict]:
    """Format conversation history for the Groq API."""
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            formatted.append({"role": role, "content": content})
    return formatted


def chat(
    message: str,
    history: list[dict] | None = None,
    stream: bool = True,
):
    """
    Process a chat message through the RAG pipeline.
    1. Retrieve relevant context
    2. Build the prompt with context + history
    3. Call Groq API
    4. Return response (streaming or complete)
    """
    # Step 1: Retrieve context
    context, sources = build_context(message)

    # Step 2: Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    if history:
        messages.extend(format_history(history))

    # Build user message with context
    if context:
        user_content = f"""## Retrieved Context:
{context}

## User Question:
{message}"""
    else:
        user_content = f"""## Note: No relevant documents found in the knowledge base.

## User Question:
{message}"""

    messages.append({"role": "user", "content": user_content})

    # Step 3: Call Groq API
    client = get_groq_client()

    if stream:
        return _stream_response(client, messages, sources)
    else:
        return _complete_response(client, messages, sources)


def _stream_response(client: Groq, messages: list[dict], sources: list[dict]):
    """Generate a streaming response from Groq."""
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield {
                "type": "content",
                "content": delta.content,
            }

    # Send sources at the end
    yield {
        "type": "sources",
        "sources": sources,
    }

    yield {
        "type": "done",
    }


def _complete_response(client: Groq, messages: list[dict], sources: list[dict]) -> dict:
    """Generate a complete (non-streaming) response from Groq."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=False,
    )

    return {
        "content": response.choices[0].message.content,
        "sources": sources,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
