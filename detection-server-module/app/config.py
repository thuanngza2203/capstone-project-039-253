from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    groq_api_key: str

    normalizer_model: str = "openai/gpt-oss-120b"
    # Theo model: openai/gpt-oss-* nhận low/medium/high; qwen/qwen3-32b nhận none/default.
    # Để trống = không gửi tham số (model không hỗ trợ reasoning).
    normalizer_reasoning_effort: str = "low"
    answer_model: str = "openai/gpt-oss-120b"

    # Số message gần nhất đưa vào normalizer/answer LLM. 6 = giống trước refactor.
    max_history_turns: int = Field(default=6, ge=1)

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "plant_chatbot"

    # Feedback RAG v5: local multilingual embedding + cross-encoder reranker.
    feedback_embedding_model: str = "intfloat/multilingual-e5-small"
    feedback_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    feedback_retrieve_k: int = 8
    feedback_rerank_k: int = 2
    feedback_similarity_threshold: float = 0.55
    feedback_reranker_enabled: bool = True

    # Sinh câu trả lời: groq = rag/ nội bộ + Groq như trước; rag = gọi RAG-module server.
    answer_backend: Literal["groq", "rag"] = "groq"
    rag_api_url: str = "http://127.0.0.1:8010"
    rag_api_key: str = ""
    # Giây; phải lớn hơn timeout LLM bên RAG (VLLM_TIMEOUT=120).
    rag_api_timeout: float = 150.0
    # Gửi kèm câu gốc làm câu tìm phụ (extra_queries) bên cạnh câu Groq đã chuẩn hóa.
    # Tắt: đo 24/09 không thấy lợi (RAG-module/reports/2026-09-24-query-normalization).
    rag_search_original_query: bool = False

    # Đăng nhập HTTP Basic cho /admin và /admin/*. Mật khẩu trống = không bảo vệ (cảnh báo trong log).
    admin_username: str = "admin"
    admin_password: str = ""

    # Origin được gọi API từ trình duyệt, cách nhau bằng dấu phẩy. Trống = không bật CORS.
    cors_origins: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
