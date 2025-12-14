# backend/app/core/rag.py
"""
Hybrid RAG pipeline with Generate -> Approve -> Query flow.

Key methods:
- generate_draft(prompt): ask Gemini to produce a structured doc (NOT indexed).
- answer(query, top_k): hybrid retrieval-first with LLM fallback and verification.
- approve_text(doc_id, text): helper used by routes.py to chunk/embed/index.

This file expects existing app.core.embeddings.EmbeddingModel and app.core.faiss_store.FaissStore.
"""
from typing import List, Tuple, Optional
from app.core.embeddings import EmbeddingModel
from app.core.faiss_store import FaissStore
from app.core.config import settings

import requests
import time
import traceback
import json

# Defaults
DEFAULT_MODEL = settings.GEMINI_MODEL or "gemini-flash-latest"


class RAGPipeline:
    def __init__(self, embed_model: EmbeddingModel, store: FaissStore):
        self.embed_model = embed_model
        self.store = store
        self._qa_pipeline = None  # lazy local QA fallback

    # --- Gemini low-level call (minimal compatible body) ---
    def _call_gemini_raw(self, prompt: str, model: str = DEFAULT_MODEL, timeout: int = None) -> Tuple[bool, str]:
        timeout = timeout or settings.GEMINI_TIMEOUT
        key = settings.GEMINI_API_KEY or ""
        if not key:
            print("[Gemini debug] GEMINI_API_KEY not set")
            return False, "GEMINI_API_KEY not set"

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        try:
            masked_key = f"{len(key)}-chars"
            print(f"[Gemini debug] Calling Gemini endpoint: {endpoint}")
            print(f"[Gemini debug] Model: {model} | API key length: {masked_key}")
            resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            print("[Gemini debug] Exception when calling Gemini:")
            traceback.print_exc()
            return False, f"HTTP exception: {e}"

        status = resp.status_code
        raw = "<no-body>"
        try:
            raw = resp.text
        except Exception:
            pass

        if status != 200:
            print(f"[Gemini debug] Non-200 status from Gemini: {status}")
            preview = raw if len(raw) < 2000 else raw[:2000] + "...(truncated)"
            print(f"[Gemini debug] Response body (preview): {preview}")
            return False, f"Gemini error {status}: {raw}"

        try:
            j = resp.json()
        except Exception as e:
            print("[Gemini debug] JSON parse failed")
            print(raw[:2000] if raw else "(empty)")
            traceback.print_exc()
            return False, f"Invalid JSON: {e}"

        # Defensive extractor
        def _extract_text(obj):
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (int, float, bool)):
                return str(obj)
            if isinstance(obj, list):
                parts = []
                for x in obj:
                    t = _extract_text(x)
                    if t:
                        parts.append(t)
                return "\n".join(parts)
            if isinstance(obj, dict):
                for k in ("text", "content", "message", "generated_text", "response", "output"):
                    if k in obj and obj[k] is not None:
                        return _extract_text(obj[k])
                # content list?
                if "content" in obj and isinstance(obj["content"], list):
                    return _extract_text(obj["content"])
                # else combine
                vals = []
                for v in obj.values():
                    t = _extract_text(v)
                    if t:
                        vals.append(t)
                return "\n".join(vals)
            return ""

        text_out = ""
        try:
            if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                text_out = _extract_text(j["candidates"][0])
        except Exception:
            text_out = ""

        if not text_out:
            for k in ("output", "responses", "response", "result"):
                if k in j and j[k] is not None:
                    text_out = _extract_text(j[k])
                    if text_out:
                        break

        if not text_out:
            for k in ("generated_text", "text"):
                if k in j and j[k] is not None:
                    text_out = _extract_text(j[k])
                    if text_out:
                        break

        text_out = (text_out or "").strip()
        preview = text_out if len(text_out) < 1000 else text_out[:1000] + "...(truncated)"
        print(f"[Gemini debug] Extracted text preview: {preview}")
        return True, text_out

    # --- High-level: generate a draft document from a prompt (not indexed) ---
    def generate_draft(self, prompt: str, model: str = DEFAULT_MODEL) -> str:
        """
        Ask Gemini to produce a structured cricket profile or document.
        We keep the prompt instructive and constrained to return a stand-alone document.
        """
        system = (
            "You are a helpful assistant that creates short, factual, structured player profiles.\n"
            "Produce a concise 120-250 word profile for the asked player. Use plain text only.\n"
            "Include a clear fact sentence for nickname or epithet if known (e.g., 'Sachin Tendulkar is widely known as the \"God of Cricket\".').\n"
            "Do not invent stats. If you don't know a stat, omit it.\n\n"
        )
        user = f"Generate a profile:\n\n{prompt}\n\nReturn only the profile text."
        full = system + user
        success, out = self._call_gemini_raw(full, model=model)
        if success:
            return out.strip() or ""
        # fallback text
        return f"[gemini error: {out}]"

    # --- Local QA load & helper (lazy) ---
    def _load_local_qa(self):
        if self._qa_pipeline is None:
            try:
                from transformers import pipeline
                self._qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2",
                    device=-1
                )
            except Exception as e:
                print("[RAG debug] Failed to load local QA:", e)
                self._qa_pipeline = None

    def _local_qa_answer(self, question: str, contexts: List[dict]) -> str:
        self._load_local_qa()
        if not self._qa_pipeline:
            return "I don't know."
        combined = "\n\n".join([c.get("text", "") for c in contexts[:4]])
        try:
            res = self._qa_pipeline({"question": question, "context": combined})
            ans = res.get("answer", "")
            return ans or "I don't know."
        except Exception as e:
            print("[RAG debug] local QA error:", e)
            return "I don't know."

    # --- Approve & index helper (used by /approve-doc) ---
    def approve_text(self, doc_id: str, text: str) -> dict:
        """
        Chunk, embed and save text into FAISS. Returns a small summary dict.
        This reuses embed_model chunking rules (we assume embed_model can embed_texts).
        """
        # Use chunker utils if you have them; else simple sentence-split chunks
        chunks = self._simple_chunk(text)
        # Embeddings for each chunk
        vectors = self.embed_model.embed_texts(chunks)
        # Add to FaissStore: store expects metadata list aligned with vectors
        metadatas = []
        for i, ch in enumerate(chunks):
            metadatas.append({"doc_id": doc_id, "chunk_id": i, "text": ch})
        self.store.add(vectors, metadatas)
        # Save index/metadata to disk
        try:
            self.store.save()
        except Exception as e:
            print("[RAG debug] Faiss save failed:", e)
        return {"status": "ok", "chunks_added": len(chunks)}

    def _simple_chunk(self, text: str, max_len: int = 800) -> List[str]:
        # naive chunker: split by sentences until concatenation reaches max_len chars
        import re
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        cur = ""
        for s in sents:
            if not s:
                continue
            if len(cur) + len(s) + 1 <= max_len:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = s
        if cur:
            chunks.append(cur.strip())
        return chunks

    # --- The hybrid answer pipeline ---
    def answer(self, query: str, top_k: int = 3) -> Tuple[str, List[dict], dict]:
        """
        Hybrid behavior:
        - Retrieve top_k contexts and their scores.
        - If top score >= RAG_THRESHOLD and RAG_MODE allows grounded answers:
            build a grounded prompt (contexts only) and ask Gemini to answer using only contexts.
            Return answer with verified=True if local QA confirms.
        - Else (no strong context):
            call Gemini raw (free-mode) to answer, then run verification pass (score vs contexts).
        - If Gemini fails, fallback to local QA or return snippets.
        Returns: (answer_text, contexts, meta)
        meta contains fields: {'mode': 'grounded'|'fallback', 'verified': True/False, 'scores':[...]}
        """
        contexts = self.store.search_by_query(query, self.embed_model, top_k=top_k) if hasattr(self.store, "search_by_query") else self.store.search(self.embed_model.embed_text(query), top_k=top_k)
        # contexts: list of dicts with keys: text, doc_id, score, chunk_id...
        scores = [c.get("score", 0.0) for c in contexts]
        top_score = max(scores) if scores else 0.0
        mode = "fallback"
        verified = False

        # If strong evidence, do grounded answer
        if settings.RAG_MODE == "strict" or (settings.RAG_MODE == "hybrid" and top_score >= settings.RAG_THRESHOLD):
            mode = "grounded"
            # build grounded prompt
            lines = [
                "You are a helpful assistant. Use ONLY the provided context to answer the user's question.",
                "If the answer is not contained in the context, say exactly: \"I don't know.\"",
                "",
                "Context:"
            ]
            for i, c in enumerate(contexts, start=1):
                txt = c.get("text", "")[:1400]
                lines.append(f"[{i}] {txt}")
            lines.append("")
            lines.append(f"User Question: {query}")
            lines.append("Answer concisely and cite the context numbers used (e.g., [1]).")
            lines.append("")
            lines.append("Answer:")
            prompt = "\n".join(lines)
            success, out = self._call_gemini_raw(prompt, model=DEFAULT_MODEL)
            if success and out:
                answer_text = self._clean_echo(prompt, out)
                # verification via local QA: check if local QA agrees with answer_text
                qa_ans = self._local_qa_answer(query, contexts)
                verified = (qa_ans.strip() != "" and qa_ans.strip().lower() in answer_text.strip().lower())
                return answer_text, contexts, {"mode": mode, "verified": verified, "scores": scores}
            # Gemini failed: try local QA
            answer_text = self._local_qa_answer(query, contexts)
            return answer_text, contexts, {"mode": mode, "verified": bool(answer_text != "I don't know."), "scores": scores}

        # Else fallback to LLM free-mode
        mode = "fallback"
        # simple user prompt (no context restriction)
        prompt = f"You are a helpful assistant. Answer the question concisely:\n\nUser Question: {query}\n\nAnswer:"
        success, out = self._call_gemini_raw(prompt, model=DEFAULT_MODEL)
        if success and out:
            answer_text = out.strip()
            # run a quick verification: use local QA on retrieved contexts and check token overlap
            qa_ans = self._local_qa_answer(query, contexts)
            # naive verification: if local QA returns non-empty and appears in answer_text --> verified
            verified = bool(qa_ans and qa_ans.strip() != "I don't know." and qa_ans.strip().lower() in answer_text.strip().lower())
            return answer_text, contexts, {"mode": mode, "verified": verified, "scores": scores}
        # Gemini failed -> local QA
        answer_text = self._local_qa_answer(query, contexts)
        return answer_text, contexts, {"mode": mode, "verified": bool(answer_text != "I don't know."), "scores": scores}

    # remove prompt echo like earlier
    def _clean_echo(self, prompt: str, generation: str) -> str:
        try:
            p = (prompt or "").strip()
            g = (generation or "").strip()
            if not g:
                return ""
            if p and g.startswith(p):
                marker = "Answer:"
                idx = g.find(marker)
                if idx != -1:
                    g = g[idx + len(marker):].strip()
                else:
                    g = g[len(p):].strip()
            if "Answer:" in g:
                idx = g.find("Answer:")
                tail = g[idx + len("Answer:"):].strip()
                if tail:
                    g = tail
            lines = [ln.strip() for ln in g.splitlines() if ln.strip()]
            cleaned = []
            for ln in lines:
                low = ln.lower()
                if low.startswith("context:") or low.startswith("user question:"):
                    continue
                if ln == p:
                    continue
                cleaned.append(ln)
            if cleaned:
                return cleaned[0] if len(cleaned[0]) < 800 else "\n".join(cleaned).strip()
            return g.split("\n", 1)[0].strip()
        except Exception:
            return (generation or "").strip()
