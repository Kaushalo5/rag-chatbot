# backend/app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    # API keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    # Default model (we set to gemini-flash-latest as requested)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

    # FAISS / storage paths
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "app" / "storage")))
    INDEX_DIR: Path = STORAGE_DIR / "indexes"
    DOCS_DIR: Path = STORAGE_DIR / "docs"
    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss.index")
    METADATA_PATH: str = str(INDEX_DIR / "metadatas.json")

    # RAG / hybrid settings
    RAG_MODE: str = os.getenv("RAG_MODE", "hybrid")   # 'strict' or 'hybrid'
    RAG_THRESHOLD: float = float(os.getenv("RAG_THRESHOLD", 0.30))  # relevance score threshold

    # Embedding model name if using remote embeddings (not required for local SentenceTransformer)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # CORS (for dev)
    ALLOW_ORIGINS = ["*"]

    # Generation / Gemini options (kept minimal for compatibility)
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", 30))

# instantiate
settings = Settings()

# Ensure storage dirs exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
