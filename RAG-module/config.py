
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


def _int_setting(name: str, default: int, *, minimum: int = 1) -> int:
    """Đọc số nguyên từ environment; mọi cấu hình số đều đi qua đây."""

    limit = "lớn hơn 0" if minimum == 1 else f"từ {minimum} trở lên"
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} phải là số nguyên {limit}.") from exc
    if value < minimum:
        raise ValueError(f"{name} phải là số nguyên {limit}.")
    return value


# Tham số của đường recursive và số chunk trả về; mặc định giữ nguyên như cũ.
CHUNK_SIZE = _int_setting("RAG_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int_setting("RAG_CHUNK_OVERLAP", 150, minimum=0)
TOP_K = _int_setting("RAG_TOP_K", 4)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("RAG_CHUNK_OVERLAP phải nhỏ hơn RAG_CHUNK_SIZE.")


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
    maximum = _int_setting("CHUNK_MAX_TOKENS", 400)
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
    max_chars = _int_setting("CHAT_HISTORY_MAX_CHARS", 6000, minimum=300)
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
        candidate_k=_int_setting("RETRIEVAL_CANDIDATE_K", 20),
        rrf_k=_int_setting("RETRIEVAL_RRF_K", 60),
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
# Embedding vẫn đọc lúc import: nó là danh tính của index đã build, được manifest
# so khớp khi query, không phải nút chỉnh theo từng lần gọi như cấu hình LLM.


@dataclass(frozen=True)
class LLMSettings:
    """Cấu hình provider; đọc tại thời điểm gọi như các nhóm setting còn lại."""

    provider: str = "ollama"
    ollama_model: str = "qwen3.5:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 800
    ollama_keep_alive: str = "10m"
    ollama_think: bool = False
    gemini_model: str = "gemini-2.5-flash"
    vllm_model: str = "Qwen/Qwen3-0.6B"
    vllm_base_url: str = "http://127.0.0.1:8000/v1"


def get_llm_settings(*, provider: str | None = None) -> LLMSettings:
    """Đọc environment tại thời điểm gọi, không chốt cứng lúc import module.

    Nhờ vậy `provider=` và biến môi trường đổi trong process đều có hiệu lực.
    Riêng file `.env` vẫn chỉ được `load_dotenv` đọc một lần lúc import, nên sửa
    file thì vẫn phải khởi động lại chương trình.
    """

    selected = (
        provider if provider is not None else os.getenv("LLM_PROVIDER", "ollama")
    ).strip().casefold() or "ollama"
    return LLMSettings(
        provider=selected,
        ollama_model=os.getenv("OLLAMA_MODEL", "").strip() or LLMSettings.ollama_model,
        ollama_base_url=(
            os.getenv("OLLAMA_BASE_URL", "").strip() or LLMSettings.ollama_base_url
        ),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "800")),
        ollama_keep_alive=(
            os.getenv("OLLAMA_KEEP_ALIVE", "").strip() or LLMSettings.ollama_keep_alive
        ),
        ollama_think=os.getenv("OLLAMA_THINK", "false").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        },
        gemini_model=os.getenv("GEMINI_MODEL", "").strip() or LLMSettings.gemini_model,
        vllm_model=os.getenv("VLLM_MODEL", "").strip() or LLMSettings.vllm_model,
        vllm_base_url=(
            os.getenv("VLLM_BASE_URL", "").strip() or LLMSettings.vllm_base_url
        ).rstrip("/"),
    )


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


def create_chat_model(*, provider: str | None = None) -> BaseChatModel:
    """Tạo Ollama, Gemini hoặc vLLM client theo cấu hình đọc lúc gọi hàm."""

    settings = get_llm_settings(provider=provider)

    if settings.provider == "ollama":
        if settings.ollama_num_ctx < 1 or settings.ollama_num_predict < 1:
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
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
            num_predict=settings.ollama_num_predict,
            reasoning=settings.ollama_think,
            keep_alive=settings.ollama_keep_alive,
            validate_model_on_init=True,
        )

    if settings.provider == "vllm":
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
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key=os.getenv("VLLM_API_KEY", "").strip() or "EMPTY",
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=2,
            use_responses_api=False,
            extra_body=extra_body,
        )

    if settings.provider != "gemini":
        raise RuntimeError(
            f"LLM_PROVIDER không hợp lệ: {settings.provider!r}. "
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
        model=settings.gemini_model,
        api_key=api_key,
        temperature=0,
        thinking_budget=0,
        max_tokens=800,
        max_retries=2,
    )
