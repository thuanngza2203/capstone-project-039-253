
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
    """Tạo Gemini chat model; đây là điểm thay thế khi thêm local LLM."""

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
