from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Incoming request payload for RAG query endpoint.
    
    query: user question
    k: how many chunks to retrieve from vector DB
    session_id: optional (future use: chat history / memory)
    """
    query: str
    k: int = 4
    session_id: str | None = None
