from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    groq_api_key: str

    normalizer_model: str = "openai/gpt-oss-120b"
    answer_model: str = "openai/gpt-oss-120b"

    max_history_turns: int = 12

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "plant_chatbot"

    # Feedback RAG v5: local multilingual embedding + cross-encoder reranker.
    feedback_embedding_model: str = "intfloat/multilingual-e5-small"
    feedback_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    feedback_retrieve_k: int = 8
    feedback_rerank_k: int = 2
    feedback_similarity_threshold: float = 0.55
    feedback_reranker_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
