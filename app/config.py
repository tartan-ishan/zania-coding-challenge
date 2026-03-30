from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Document chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    confidence_threshold: float = 0.3
    retrieval_k: int = 15          # candidates per sub-query before MMR
    mmr_fetch_k: int = 30          # wider candidate pool for MMR diversity pass
    mmr_lambda: float = 0.5        # 0 = max diversity, 1 = max relevance
    bm25_weight: float = 0.4       # weight for BM25 in hybrid ensemble (semantic = 1 - bm25_weight)

    # Multi-query decomposition
    multi_query_count: int = 4     # sub-queries to generate per question

    # Retry settings for OpenAI rate limits
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Request limits
    max_document_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_questions_bytes: int = 100 * 1024        # 100 KB
    max_questions: int = 50
    max_concurrent_questions: int = 50
    llm_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
