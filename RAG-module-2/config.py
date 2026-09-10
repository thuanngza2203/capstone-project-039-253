
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "plant_disease_vi"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding"
).strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().casefold() or "ollama"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b").strip() or "qwen3.5:9b"
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    or "http://127.0.0.1:11434"
)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "800"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


@lru_cache(maxsize=1)
def create_embeddings() -> Embeddings:
    """Tải embedding model local một lần trong mỗi Python process."""

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )


def create_chat_model() -> BaseChatModel:
    """Tạo chat model theo ``LLM_PROVIDER`` trong file ``.env``."""

    if LLM_PROVIDER == "ollama":
        if OLLAMA_NUM_CTX < 1 or OLLAMA_NUM_PREDICT < 1:
            raise RuntimeError(
                "OLLAMA_NUM_CTX và OLLAMA_NUM_PREDICT phải lớn hơn 0."
            )

        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu langchain-ollama. Hãy chạy "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            reasoning=OLLAMA_THINK,
            keep_alive=OLLAMA_KEEP_ALIVE,
            validate_model_on_init=True,
        )

    if LLM_PROVIDER != "gemini":
        raise RuntimeError(
            f"LLM_PROVIDER không hợp lệ: {LLM_PROVIDER!r}. "
            "Chỉ hỗ trợ 'ollama' hoặc 'gemini'."
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY. Hãy sao chép .env.example thành .env "
            "và điền API key trước khi chạy lệnh ask."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0,
        thinking_budget=0,
        max_tokens=800,
        max_retries=2,
    )
