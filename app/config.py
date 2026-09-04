from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    embedding_backend: str = "fake"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    chroma_persist_dir: str = "./data/chroma"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    collection_name: str = "rag_docs"
    upload_dir: str = "./data/uploads"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
