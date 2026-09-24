"""Tài nguyên dùng chung giữa các request và luồng retrieve/answer của API.

Server không giữ lịch sử hội thoại: mỗi request tự mang `history`. Store,
embedding và LLM client được mở một lần rồi dùng lại (giống RAGSession), có khóa
để nhiều request chạy song song trong threadpool của FastAPI không nạp trùng.

Để so sánh thí nghiệm, request chọn được index (cách chunk) và LLM provider.
Mỗi index/provider vẫn chỉ mở một lần; mọi index dùng chung một embedding model.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult

from config import (
    COLLECTION_NAME,
    INDEX_STRATEGIES,
    TOP_K,
    configured_model,
    create_chat_model,
    get_chat_settings,
    get_chunking_settings,
    get_index_directory,
    get_llm_settings,
    get_retrieval_settings,
    resolve_vllm_base_url,
    strategy_index_directory,
)
from conversation import rewrite_question
from index_manifest import corpus_fingerprint, read_manifest
from rag import (
    NO_CONTEXT_ANSWER,
    _load_vector_store,
    cited_source_numbers,
    generate_answer,
    invalid_citations,
    load_documents,
)
from retrieval import SearchResult, search_store
from taxonomy import ResolvedScope, resolve_scope

# finish_reason của câu bị cắt ở giới hạn token: OpenAI/vLLM/Ollama "length", Gemini "max_tokens".
TRUNCATED_REASONS = frozenset({"length", "max_tokens"})
PROBE_TIMEOUT_SECONDS = 10


class ServiceError(Exception):
    """Lỗi có mã HTTP; tầng API chuyển thành {"detail": ...}."""

    status_code = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class IndexUnavailable(ServiceError):
    status_code = 503


class RetrievalFailed(ServiceError):
    status_code = 503


class LLMUnavailable(ServiceError):
    status_code = 503


class LLMFailed(ServiceError):
    status_code = 502


@dataclass(frozen=True)
class IndexHandle:
    name: str
    directory: Path
    store: Any
    strategy: str | None  # Theo manifest; None = index cũ chưa có manifest.


@dataclass
class RunMeta:
    """Cấu hình thật đã dùng cho một request; công cụ thí nghiệm ghi lại nguyên khối."""

    index: str
    top_k: int
    chunking_strategy: str | None = None
    retrieval_mode: str | None = None  # None: không tìm (scope chặn).
    reranker_enabled: bool | None = None
    search_queries: list[str] = field(default_factory=list)  # Câu chính trước; rỗng: không tìm.
    llm: dict[str, Any] | None = None  # None: không gọi LLM.
    timing_ms: dict[str, int | None] = field(default_factory=lambda: {
        "rewrite": None, "retrieve": None, "generate": None, "total": None,
    })


@dataclass
class RetrievalOutcome:
    scope: ResolvedScope
    result: SearchResult | None  # None: phạm vi không cho phép tìm.
    meta: RunMeta


@dataclass
class AnswerOutcome:
    answer: str
    sources: list[str]
    grounded: bool
    retrieval_query: str
    scope: ResolvedScope
    result: SearchResult | None
    meta: RunMeta
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


class GenerationRecorder(BaseCallbackHandler):
    """Ghi model, token và finish_reason mà LLM báo về ở bước sinh câu trả lời."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.finish_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return
        message = getattr(generation, "message", None)
        metadata = getattr(message, "response_metadata", None) or {}
        info = generation.generation_info or {}
        self.model = metadata.get("model_name") or metadata.get("model") or self.model
        reason = metadata.get("finish_reason") or metadata.get("done_reason") or info.get("finish_reason")
        if reason is not None:
            # Gemini có thể trả enum (FinishReason.MAX_TOKENS): chỉ giữ tên.
            self.finish_reason = str(getattr(reason, "name", reason)).lower()
        usage = getattr(message, "usage_metadata", None) or {}
        if usage:
            self.input_tokens = usage.get("input_tokens")
            self.output_tokens = usage.get("output_tokens")

    def describe(self, provider: str) -> dict[str, Any]:
        return {
            "provider": provider,
            # LLM không báo tên (client giả trong test) thì lấy tên trong .env.
            "model": self.model or configured_model(get_llm_settings(provider=provider)),
            "finish_reason": self.finish_reason,
            "truncated": self.finish_reason in TRUNCATED_REASONS,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def history_messages(history: list[dict[str, str]]) -> list[BaseMessage]:
    """Giữ các lượt gần nhất trong CHAT_HISTORY_TURNS / CHAT_HISTORY_MAX_CHARS."""
    settings = get_chat_settings()
    if settings.history_turns == 0:
        return []
    recent = list(history[-2 * settings.history_turns:])
    while recent and sum(len(turn["content"]) for turn in recent) > settings.history_max_chars:
        recent.pop(0)
    return [
        HumanMessage(content=turn["content"]) if turn["role"] == "user"
        else AIMessage(content=turn["content"])
        for turn in recent
    ]


def subject_text(scope: ResolvedScope, subject_context: str | None) -> str:
    """Đối tượng cho LLM: nhãn đã chuẩn hóa + kết quả ảnh, không phải bằng chứng."""
    lines = []
    if scope.plant:
        lines.append(f"- Cây: {scope.plant}")
    if scope.disease:
        lines.append(f"- Bệnh: {scope.disease}")
    if subject_context and subject_context.strip():
        lines.append(f"- Kết quả nhận diện ảnh: {subject_context.strip()}")
    return "\n".join(lines)


def refusal_for(scope: ResolvedScope) -> str:
    """Câu trả lời dựng sẵn khi không được tìm; không gọi LLM."""
    if scope.status == "unsupported_disease":
        target = f"bệnh {scope.disease}" + (f" trên cây {scope.plant}" if scope.plant else "")
        return (
            f"Kho tài liệu hiện chưa có thông tin về {target}, nên mình chưa thể tư vấn "
            "chính xác. Bạn nên hỏi cán bộ kỹ thuật nông nghiệp tại địa phương."
        )
    return NO_CONTEXT_ANSWER


class RAGRuntime:
    def __init__(
        self,
        *,
        vector_store: Any | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Any | None = None,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        stores: Mapping[str, Any] | None = None,
        llms: Mapping[str, BaseChatModel] | None = None,
        index_directories: Mapping[str, Path | str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """`vector_store`/`llm` là index và LLM mặc định; `stores`/`llms` gắn theo tên
        (test inject). Thiếu thì mở từ đĩa / tạo client lúc cần."""
        self._embeddings = embeddings
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self.default_index = get_chunking_settings().strategy
        self._injected_stores = dict(stores or {})
        if vector_store is not None:
            self._injected_stores[self.default_index] = vector_store
        self._index_directories = {name: Path(path) for name, path in (index_directories or {}).items()}
        self._handles: dict[str, IndexHandle] = {}
        self._default_llm = llm
        self._llms: dict[str, BaseChatModel] = dict(llms or {})
        self._http_client = http_client
        self._store_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self.load_errors: dict[str, str] = {}

    # --- Index --------------------------------------------------------------

    def directory_for(self, name: str) -> Path:
        if name in self._index_directories:
            return self._index_directories[name]
        if name == self.default_index:
            return get_index_directory(self._persist_directory)
        # Index không mặc định luôn ở thư mục riêng của strategy, không theo CHROMA_DIR.
        return strategy_index_directory(name)

    @property
    def index_directory(self) -> Path:
        return self.directory_for(self.default_index)

    @property
    def load_error(self) -> str | None:
        return self.load_errors.get(self.default_index)

    @property
    def loaded(self) -> bool:
        return self.default_index in self._handles

    def load(self, index: str | None = None) -> IndexHandle:
        """Mở index + embedding một lần. Lỗi thì lần gọi sau thử lại (ví dụ vừa index xong)."""
        name = index or self.default_index
        if name not in INDEX_STRATEGIES:
            raise ValueError(f"Index không hợp lệ: {name!r}.")
        handle = self._handles.get(name)
        if handle is not None:
            return handle
        with self._store_lock:
            handle = self._handles.get(name)
            if handle is None:
                handle = self._open(name)
                self._handles[name] = handle
                self.load_errors.pop(name, None)
        return handle

    def _open(self, name: str) -> IndexHandle:
        directory = self.directory_for(name)
        injected = self._injected_stores.get(name)
        if injected is not None:
            return IndexHandle(name, directory, injected, name)
        try:
            manifest = read_manifest(directory, self._collection_name)
        except RuntimeError as exc:
            self.load_errors[name] = str(exc)
            raise IndexUnavailable(f"Index chưa sẵn sàng ({name}): {exc}") from exc
        strategy = manifest["chunking"]["strategy"] if manifest else None
        # Kiểm tra trước khi nạp embedding. build_index không cho ghi structure vào thư mục
        # chưa có manifest, nên index cũ không manifest là recursive.
        if manifest is not None or (directory / "chroma.sqlite3").exists():
            actual = strategy or "recursive"
            if actual != name:
                detail = (
                    f"Thư mục {directory} chứa index {actual}, không phải {name}. "
                    "Kiểm tra CHROMA_DIR và CHUNKING_STRATEGY, hoặc chạy "
                    f"`python main.py index --strategy {name}`."
                )
                self.load_errors[name] = detail
                raise IndexUnavailable(detail)
        try:
            store = _load_vector_store(
                directory, self._embeddings, self._collection_name, load_embeddings=True,
            )
        except RuntimeError as exc:
            self.load_errors[name] = str(exc)
            raise IndexUnavailable(f"Index chưa sẵn sàng ({name}): {exc}") from exc
        return IndexHandle(name, directory, store, strategy)

    def data_fingerprint(self) -> str | None:
        """Dấu vân tay của data/ hiện tại; None khi không đọc được (server không kèm data)."""
        try:
            return corpus_fingerprint(load_documents())
        except Exception:  # noqa: BLE001 - chỉ để báo index cũ, không được làm hỏng status.
            return None

    def index_status(self, name: str, fingerprint: str | None) -> dict[str, Any]:
        directory = self.directory_for(name)
        handle = self._handles.get(name)
        status: dict[str, Any] = {
            "name": name, "default": name == self.default_index, "directory": str(directory),
            "exists": directory.is_dir(), "loaded": handle is not None, "has_manifest": False,
            "strategy": handle.strategy if handle else None, "chunk_count": None,
            "built_at": None, "matches_data": None, "detail": self.load_errors.get(name),
        }
        if name in self._injected_stores:
            status["strategy"] = name
            return status
        try:
            manifest = read_manifest(directory, self._collection_name)
        except RuntimeError as exc:
            status["detail"] = str(exc)
            return status
        if manifest is None:
            if status["exists"] and status["detail"] is None:
                status["detail"] = (
                    "Index cũ chưa có manifest: không biết build từ dữ liệu nào. "
                    f"Chạy lại `python main.py index --strategy {name}`."
                )
            return status
        status.update(
            has_manifest=True,
            strategy=manifest["chunking"]["strategy"],
            chunk_count=manifest.get("chunk_count"),
            built_at=manifest.get("created_at"),
            matches_data=None if fingerprint is None else manifest.get("corpus_sha256") == fingerprint,
        )
        if status["matches_data"] is False and status["detail"] is None:
            status["detail"] = (
                "data/ đã đổi sau lần build này. Chạy lại "
                f"`python main.py index --strategy {name}` trước khi so sánh."
            )
        return status

    # --- LLM ----------------------------------------------------------------

    @staticmethod
    def provider_name(provider: str | None = None) -> str:
        return provider or get_llm_settings().provider

    def llm(self, provider: str | None = None) -> BaseChatModel:
        """Tạo client lúc cần: server vẫn phục vụ /v1/retrieve khi LLM chưa bật."""
        name = self.provider_name(provider)
        if self._default_llm is not None and name == self.provider_name():
            return self._default_llm
        client = self._llms.get(name)
        if client is not None:
            return client
        with self._llm_lock:
            client = self._llms.get(name)
            if client is None:
                try:
                    client = create_chat_model(provider=name)
                except (RuntimeError, ValueError) as exc:
                    raise LLMUnavailable(f"LLM chưa cấu hình được: {exc}") from exc
                self._llms[name] = client
        return client

    def describe_llm(self, provider: str | None = None, *, probe: bool = False) -> dict[str, Any]:
        """Cấu hình LLM (không kèm key). `probe` hỏi server LLM model nào đang thật sự chạy."""
        settings = get_llm_settings(provider=provider)
        name = settings.provider
        info: dict[str, Any] = {
            "provider": name, "default": name == self.provider_name(),
            "model": configured_model(settings), "endpoint": None, "probed": probe,
            "reachable": None, "served": None, "detail": None,
        }
        if name == "vllm":
            try:
                info["endpoint"] = resolve_vllm_base_url(settings)
            except RuntimeError as exc:
                info["detail"] = f"Cấu hình vLLM sai: {exc}"
                return info
        elif name == "ollama":
            info["endpoint"] = settings.ollama_base_url
        elif name != "gemini":
            info["detail"] = f"Provider không hợp lệ: {name!r}."
            return info
        if not probe:
            return info
        if name == "gemini":
            info["detail"] = "Không probe Gemini: Google phục vụ đúng model theo tên."
            return info

        client = self._http_client or httpx.Client(timeout=PROBE_TIMEOUT_SECONDS)
        try:
            served = (
                self._probe_vllm(client, info["endpoint"]) if name == "vllm"
                else self._probe_ollama(client, info["endpoint"], info["model"])
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            info.update(reachable=False, detail=f"Không hỏi được {info['endpoint']}: {type(exc).__name__}: {exc}")
            return info
        finally:
            if self._http_client is None:
                client.close()
        info.update(reachable=True, served=served)
        if name == "vllm" and all(model["id"] != info["model"] for model in served):
            info["detail"] = (
                f"VLLM_MODEL={info['model']!r} không khớp served-model-name của server "
                f"({', '.join(model['id'] for model in served) or 'không có'}): /v1/answer sẽ lỗi 404."
            )
        if name == "ollama" and not served:
            info["detail"] = f"Ollama chưa có model {info['model']!r}: chạy `ollama pull {info['model']}`."
        return info

    @staticmethod
    def _probe_vllm(client: httpx.Client, base_url: str) -> list[dict[str, Any]]:
        key = os.getenv("VLLM_API_KEY", "").strip()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        response = client.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        # `root` là model thật (LLM_MODEL_ID); `id` chỉ là served-model-name, thường là alias.
        return [
            {"id": model["id"], "root": model.get("root"), "max_model_len": model.get("max_model_len"),
             "digest": None, "details": None}
            for model in response.json()["data"]
        ]

    @staticmethod
    def _probe_ollama(client: httpx.Client, base_url: str, model: str) -> list[dict[str, Any]]:
        response = client.get(f"{base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        served = []
        for entry in response.json().get("models", []):
            if entry.get("name") != model and entry.get("model") != model:
                continue
            details = entry.get("details") or {}
            size = " ".join(filter(None, (details.get("parameter_size"), details.get("quantization_level"))))
            served.append({
                "id": entry.get("name") or model, "root": None, "max_model_len": None,
                # Digest phân biệt hai bản khác nhau của cùng một tag.
                "digest": str(entry.get("digest", "")).removeprefix("sha256:")[:12] or None,
                "details": size or None,
            })
        return served

    # --- Retrieve / answer --------------------------------------------------

    def _search(
        self, query: str, scope: ResolvedScope, meta: RunMeta, *,
        mode: str | None, rerank: bool | None, extra_queries: Sequence[str] = (),
    ) -> SearchResult:
        handle = self.load(meta.index)
        meta.chunking_strategy = handle.strategy
        started = time.perf_counter()
        try:
            settings = get_retrieval_settings(mode=mode, rerank=rerank)
            meta.retrieval_mode, meta.reranker_enabled = settings.mode, settings.reranker_enabled
            result = search_store(
                query, handle.store, k=meta.top_k, settings=settings,
                scope=scope.metadata_scope(), extra_queries=extra_queries,
            )
        except (RuntimeError, ValueError) as exc:
            raise RetrievalFailed(f"Retrieval lỗi: {exc}") from exc
        meta.timing_ms["retrieve"] = _elapsed_ms(started)
        meta.search_queries = list(result.queries)
        return result

    def _new_meta(self, index: str | None, top_k: int | None) -> RunMeta:
        name = index or self.default_index
        if name not in INDEX_STRATEGIES:
            raise ValueError(f"Index không hợp lệ: {name!r}.")
        return RunMeta(index=name, top_k=top_k or TOP_K)

    def retrieve(
        self, query: str, *, plant_type: str | None = None, disease: str | None = None,
        top_k: int | None = None, mode: str | None = None, rerank: bool | None = None,
        index: str | None = None, extra_queries: Sequence[str] = (),
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        meta = self._new_meta(index, top_k)
        scope = resolve_scope(plant_type, disease)
        result = None
        if scope.searchable:
            result = self._search(query, scope, meta, mode=mode, rerank=rerank,
                                  extra_queries=extra_queries)
        meta.timing_ms["total"] = _elapsed_ms(started)
        return RetrievalOutcome(scope, result, meta)

    def answer(
        self, query: str, *, plant_type: str | None = None, disease: str | None = None,
        history: list[dict[str, str]] | None = None, subject_context: str | None = None,
        rewrite_query: bool = False, retrieval_query: str | None = None,
        top_k: int | None = None, mode: str | None = None, rerank: bool | None = None,
        index: str | None = None, llm_provider: str | None = None,
        extra_queries: Sequence[str] = (),
    ) -> AnswerOutcome:
        started = time.perf_counter()
        meta = self._new_meta(index, top_k)
        scope = resolve_scope(plant_type, disease)
        messages = history_messages(history or [])

        def finish(answer: str, sources: list[str], grounded: bool, search_query: str,
                   result: SearchResult | None, **citations: list[int]) -> AnswerOutcome:
            meta.timing_ms["total"] = _elapsed_ms(started)
            return AnswerOutcome(answer, sources, grounded, search_query, scope, result, meta, **citations)

        if not scope.searchable:
            return finish(refusal_for(scope), [], False, query, None)

        # Câu tìm do phía gọi đưa (đã làm rõ) thắng; LLM vẫn thấy câu gốc `query`.
        if retrieval_query:
            rewrite_query = False
        retrieval_query = retrieval_query or query
        if rewrite_query and messages:
            rewrite_started = time.perf_counter()
            try:
                retrieval_query = rewrite_question(query, messages, self.llm(llm_provider))
            except ServiceError:
                raise
            except Exception as exc:  # noqa: BLE001 - lỗi provider nào cũng là lỗi upstream.
                raise LLMFailed(f"Không viết lại được câu hỏi: {exc}") from exc
            meta.timing_ms["rewrite"] = _elapsed_ms(rewrite_started)

        result = self._search(retrieval_query, scope, meta, mode=mode, rerank=rerank,
                              extra_queries=extra_queries)
        documents = result.documents
        if not documents:
            return finish(NO_CONTEXT_ANSWER, [], False, retrieval_query, result)

        llm = self.llm(llm_provider)
        recorder = GenerationRecorder()
        generate_started = time.perf_counter()
        try:
            answer, sources = generate_answer(
                query, documents, llm=llm, history=messages,
                retrieval_query=retrieval_query, subject=subject_text(scope, subject_context),
                callbacks=[recorder],
            )
        except Exception as exc:  # noqa: BLE001 - kết nối, 401 từ vLLM, timeout, output rỗng.
            raise LLMFailed(f"LLM lỗi hoặc không phản hồi: {type(exc).__name__}: {exc}") from exc
        meta.timing_ms["generate"] = _elapsed_ms(generate_started)
        meta.llm = recorder.describe(self.provider_name(llm_provider))
        return finish(
            answer, sources, True, retrieval_query, result,
            cited=cited_source_numbers(answer),
            invalid=invalid_citations(answer, len(documents)),
        )
