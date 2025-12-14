# models.py
from pydantic import BaseModel
from typing import List, Dict, Any

class EmbedRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class ContextItem(BaseModel):
    doc_id: str
    chunk_id: int
    text: str

class QueryResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, Any]]
