# faiss_store.py
import faiss
import numpy as np
import json
from app.core.config import settings
from app.utils import save_json, load_json, normalize_vectors

class FaissStore:
    def __init__(self, index_path: str = None, metadata_path: str = None):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.METADATA_PATH
        self.index = None
        self.metadatas = []  # list where position = faiss internal id
        self.dim = None
        # Try to load
        try:
            self.load()
        except Exception:
            self.index = None
            self.metadatas = []

    def create_index(self, dim: int):
        """
        We use IndexFlatIP on normalized vectors for cosine-similarity.
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, metadatas: list):
        """
        vectors: (n, dim) float32
        metadatas: list of dicts length n
        """
        if self.index is None:
            self.create_index(vectors.shape[1])
        # normalize for cosine with inner product
        vectors = normalize_vectors(vectors)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def search(self, q_vector: np.ndarray, top_k: int = 3):
        q = normalize_vectors(q_vector.reshape(1, -1))
        scores, ids = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            md = self.metadatas[idx].copy()
            md["score"] = float(score)
            results.append(md)
        return results

    def save(self):
    
     from pathlib import Path

    # Ensure the directory for the index exists
     idx_path = Path(self.index_path)
     idx_path.parent.mkdir(parents=True, exist_ok=True)

    # Write FAISS index if present
     if self.index is not None:
        # faiss.write_index expects a filesystem path string
        faiss.write_index(self.index, str(self.index_path))

    # save metadata JSON (utils.save_json already ensures parent dirs)
     save_json(self.metadatas, self.metadata_path)


    def load(self):
        # load metadata and index if exist
        try:
            self.metadatas = load_json(self.metadata_path)
        except FileNotFoundError:
            self.metadatas = []
        try:
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d
        except Exception:
            # index not available yet
            self.index = None
