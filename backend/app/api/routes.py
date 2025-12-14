# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.core.embeddings import EmbeddingModel
from app.core.faiss_store import FaissStore
from app.core.rag import RAGPipeline

router = APIRouter()

# Initialize shared resources (singleton style)
embed_model = EmbeddingModel(settings.EMBEDDING_MODEL)
faiss_store = FaissStore(index_path=settings.FAISS_INDEX_PATH, metadata_path=settings.METADATA_PATH)
rag = RAGPipeline(embed_model, faiss_store)

# ---------- Request models ----------
class GenRequest(BaseModel):
    prompt: str
    model: Optional[str] = None

class ApproveRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class EmbedRequest(BaseModel):
    doc_id: str
    text: str

# ---------- Endpoints ----------
@router.post("/generate-doc")
def generate_doc(req: GenRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    model = req.model or settings.GEMINI_MODEL
    draft = rag.generate_draft(req.prompt, model=model)
    return {"draft": draft}

@router.post("/approve-doc")
def approve_doc(req: ApproveRequest):
    if not req.doc_id or not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="doc_id and text required")
    res = rag.approve_text(req.doc_id, req.text)
    return res

@router.post("/embed-docs")
def embed_docs(req: EmbedRequest):
    # backward compatible endpoint: index raw text
    if not req.doc_id or not req.text:
        raise HTTPException(status_code=400, detail="doc_id and text required")
    res = rag.approve_text(req.doc_id, req.text)
    return res

@router.post("/query")
def query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    answer, contexts, meta = rag.answer(req.query, top_k=req.top_k or 3)
    return {"answer": answer, "contexts": contexts, "meta": meta}
