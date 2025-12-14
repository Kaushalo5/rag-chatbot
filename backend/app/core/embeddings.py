# backend/app/core/embeddings.py
"""
Robust embeddings wrapper using sentence-transformers.
If settings.EMBEDDING_MODEL is empty, fall back to 'all-MiniLM-L6-v2'.
"""

from typing import List
from app.core.config import settings

# Default model if none provided
DEFAULT_EMBEDDING_MODEL = settings.EMBEDDING_MODEL or "all-MiniLM-L6-v2"

class EmbeddingModel:
    def __init__(self, model_name: str = ""):
        # Choose model name: environment override or sensible default
        self.model_name = (model_name.strip() if model_name is not None else "").strip() or DEFAULT_EMBEDDING_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is not installed in this environment. "
                "Run: pip install sentence-transformers\nOriginal error: " + str(e)
            )

        try:
            # This will download model on first run if not cached
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            # Provide a clear actionable error
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{self.model_name}'. "
                f"Make sure the model name is correct and dependencies (transformers, torch) are installed. "
                f"Original error: {e}"
            )

    def embed_text(self, text: str):
        """
        Embed a single text -> numpy vector.
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded.")
        # sentence-transformers encode returns numpy array
        return self.model.encode([text], convert_to_numpy=True)[0]

    def embed_texts(self, texts: List[str]):
        """
        Embed a list of texts -> list/array of vectors.
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded.")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
