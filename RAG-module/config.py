
from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChunkingSettings:
    """Recursive dùng ký tự như cũ; structure giới hạn token của cả header + body."""

    strategy: str = "recursive"
    max_tokens: int = 400
    overlap_tokens: int = 40
    tokenizer_model: str = ""


def get_chunking_settings(*, strategy: str | None = None) -> ChunkingSettings:
    selected = (strategy or os.getenv("CHUNKING_STRATEGY", "recursive")).strip().lower()
    if selected not in {"recursive", "structure"}:
        raise ValueError("CHUNKING_STRATEGY phải là recursive hoặc structure.")
    # Các biến token không ảnh hưởng đường recursive, kể cả khi đang chỉnh thử.
    if selected == "recursive":
        return ChunkingSettings(strategy=selected)
    maximum = _positive_int_setting("CHUNK_MAX_TOKENS", 400)
    try:
        overlap = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
    except ValueError as exc:
        raise ValueError("CHUNK_OVERLAP_TOKENS phải là số nguyên không âm.") from exc
    if not 0 <= overlap < maximum:
        raise ValueError("CHUNK_OVERLAP_TOKENS phải nằm trong [0, CHUNK_MAX_TOKENS).")
    return ChunkingSettings(
        strategy=selected, max_tokens=maximum, overlap_tokens=overlap,
        tokenizer_model=os.getenv("CHUNK_TOKENIZER_MODEL", "").strip() or EMBEDDING_MODEL,
    )


def get_index_directory(
    directory: Path | str | None = None, *, strategy: str | None = None,
) -> Path:
    """CLI/path tường minh > CHROMA_DIR > thư mục mặc định theo strategy.

    Đường tương đối luôn tính từ RAG-module để chạy từ thư mục nào cũng giống nhau.
    """
    configured = directory if directory is not None else os.getenv("CHROMA_DIR", "").strip()
    if configured:
        path = Path(configured)
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    selected = get_chunking_settings(strategy=strategy).strategy
    return CHROMA_DIR if selected == "recursive" else PROJECT_ROOT / "chroma_db_structure"


@dataclass(frozen=True)
class RetrievalSettings:
    """Các nút chỉnh retrieval, tách riêng khỏi cấu hình LLM."""

    mode: str = "hybrid"
    candidate_k: int = 20
    rrf_k: int = 60
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_device: str = "cpu"


def _positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} phải là số nguyên lớn hơn 0.") from exc
    if value < 1:
        raise ValueError(f"{name} phải là số nguyên lớn hơn 0.")
    return value


@dataclass(frozen=True)
class ChatSettings:
    """Giới hạn lịch sử trong RAM; số lượt bằng 0 để tắt memory."""

    history_turns: int = 4
    history_max_chars: int = 6000


def get_chat_settings(*, history_turns: int | None = None) -> ChatSettings:
    try:
        turns = int(os.getenv("CHAT_HISTORY_TURNS", "4")) if history_turns is None else history_turns
    except ValueError as exc:
        raise ValueError("CHAT_HISTORY_TURNS phải là số nguyên không âm.") from exc
    if not isinstance(turns, int) or isinstance(turns, bool) or turns < 0:
        raise ValueError("CHAT_HISTORY_TURNS phải là số nguyên không âm.")
    max_chars = _positive_int_setting("CHAT_HISTORY_MAX_CHARS", 6000)
    if max_chars < 300:
        raise ValueError("CHAT_HISTORY_MAX_CHARS phải từ 300 trở lên.")
    return ChatSettings(history_turns=turns, history_max_chars=max_chars)


def get_retrieval_settings(
    *, mode: str | None = None, rerank: bool | None = None
) -> RetrievalSettings:
    """CLI/Python override được ưu tiên hơn các biến trong environment/.env."""

    selected_mode = (
        mode if mode is not None else os.getenv("RETRIEVAL_MODE", "hybrid")
    ).strip().casefold()
    if selected_mode not in {"semantic", "bm25", "hybrid"}:
        raise ValueError("RETRIEVAL_MODE phải là semantic, bm25 hoặc hybrid.")

    if rerank is None:
        value = os.getenv("RERANKER_ENABLED", "false").strip().casefold()
        if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError("RERANKER_ENABLED phải là true hoặc false.")
        rerank = value in {"true", "1", "yes", "on"}

    return RetrievalSettings(
        mode=selected_mode,
        candidate_k=_positive_int_setting("RETRIEVAL_CANDIDATE_K", 20),
        rrf_k=_positive_int_setting("RETRIEVAL_RRF_K", 60),
        reranker_enabled=rerank,
        reranker_model=(
            os.getenv("RERANKER_MODEL", "").strip()
            or RetrievalSettings.reranker_model
        ),
        reranker_device=os.getenv("RERANKER_DEVICE", "cpu").strip() or "cpu",
    )

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding"
).strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().casefold() or "ollama"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b").strip() or "qwen3.5:4b"
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

VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-0.6B").strip() or "Qwen/Qwen3-0.6B"
VLLM_BASE_URL = (
    os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").strip()
    or "http://127.0.0.1:8000/v1"
).rstrip("/")


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
    """Tạo Ollama, Gemini hoặc vLLM client theo config đọc khi khởi động."""

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

    if LLM_PROVIDER == "vllm":
        # Chỉ kiểm tra cấu hình vLLM khi provider này được chọn.
        try:
            max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "800"))
            timeout = int(os.getenv("VLLM_TIMEOUT", "120"))
        except ValueError as exc:
            raise RuntimeError(
                "VLLM_MAX_TOKENS và VLLM_TIMEOUT phải là số nguyên lớn hơn 0."
            ) from exc
        if max_tokens < 1 or timeout < 1:
            raise RuntimeError(
                "VLLM_MAX_TOKENS và VLLM_TIMEOUT phải là số nguyên lớn hơn 0."
            )

        # Tùy chọn cho chat template hỗ trợ enable_thinking (ví dụ Qwen3).
        # Để trống để dùng thiết lập mặc định của model/server.
        think = os.getenv("VLLM_THINK", "").strip().casefold()
        extra_body = None
        if think:
            if think not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                raise RuntimeError("VLLM_THINK phải là true, false hoặc để trống.")
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": think in {"1", "true", "yes", "on"}
                }
            }

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu langchain-openai. Hãy chạy "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        return ChatOpenAI(
            model=VLLM_MODEL,
            base_url=VLLM_BASE_URL,
            api_key=os.getenv("VLLM_API_KEY", "").strip() or "EMPTY",
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=2,
            use_responses_api=False,
            extra_body=extra_body,
        )

    if LLM_PROVIDER != "gemini":
        raise RuntimeError(
            f"LLM_PROVIDER không hợp lệ: {LLM_PROVIDER!r}. "
            "Chỉ hỗ trợ 'ollama', 'gemini' hoặc 'vllm'."
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
