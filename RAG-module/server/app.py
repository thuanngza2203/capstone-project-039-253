"""FastAPI app cho RAG-module. Swagger: /docs, ReDoc: /redoc, schema: /openapi.json."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Security
from fastapi import Query as QueryParam
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import (
    EMBEDDING_MODEL,
    INDEX_STRATEGIES,
    get_api_settings,
    get_retrieval_settings,
)
from retrieval import SearchResult
from server.runtime import AnswerOutcome, RAGRuntime, RunMeta, ServiceError
from server.schemas import (
    SCOPE_STATUS_MEANING,
    AnswerRequest,
    AnswerResponse,
    Chunk,
    Citations,
    ErrorResponse,
    HealthResponse,
    IndexStatus,
    LLMInfo,
    ProviderName,
    ResponseMeta,
    RetrieveRequest,
    RetrieveResponse,
    Scope,
    StatusResponse,
    TaxonomyResponse,
)
from taxonomy import ResolvedScope, taxonomy_table

logger = logging.getLogger("rag.server")

API_VERSION = "1.2.0"

DESCRIPTION = """
API tra cứu kiến thức bệnh cây: tìm tài liệu (hybrid semantic + BM25) và sinh câu
trả lời có trích nguồn `[Nguồn n]`.

**Ai gọi API này.** detection-server-module là cổng duy nhất cho web/mobile; nó
chuẩn hóa câu hỏi, nhận diện ảnh rồi gửi `query`, `plant_type`, `disease` sang đây.
Server **không lưu lịch sử**: mỗi request tự mang `history` cần dùng.

**Xác thực.** Header `Authorization: Bearer <RAG_API_KEY>`. Khi `RAG_API_KEY` trống,
xác thực tắt và server chỉ được nghe `127.0.0.1`. Bấm **Authorize** để thử trên trang này.

**Phạm vi tìm kiếm** (trường `scope.status` trong response):

| Trạng thái | Nghĩa |
| --- | --- |
""" + "\n".join(f"| `{status}` | {meaning} |" for status, meaning in SCOPE_STATUS_MEANING.items()) + """

`unsupported_disease` và `unknown_disease` cố ý không tìm theo cây: tìm theo cây lúc
đó dễ lấy nhầm tài liệu của bệnh khác cùng cây và trả lời sai bệnh.

**Tìm bằng nhiều câu.** `extra_queries` (tối đa 3) được tìm y như câu chính; mọi bảng xếp
hạng gộp bằng RRF. `meta.search_queries` ghi lại các câu đã thật sự dùng. Detection gửi câu
đã chuẩn hóa trong `retrieval_query`; gửi kèm câu gốc ở đây là tùy chọn, tắt mặc định vì đo
24/09 không thấy lợi.

**So sánh thí nghiệm.** `index` chọn index `recursive`/`structure`, `llm_provider` chọn LLM
cho từng request; `meta` trong response ghi lại cấu hình thật đã dùng, token và thời gian.
LLM-server giữ tên API `rag-llm` khi đổi model, nên muốn biết model thật đang chạy thì gọi
`GET /v1/llm?probe=true` (trường `root`).
"""

TAGS = [
    {"name": "Hệ thống", "description": "Kiểm tra process, index và cấu hình đang chạy."},
    {"name": "Tra cứu", "description": "Tìm tài liệu và sinh câu trả lời."},
    {"name": "Danh mục", "description": "Cây, bệnh và độ phủ tài liệu mà API hiểu."},
]

AUTH_RESPONSES = {401: {"model": ErrorResponse, "description": "Thiếu hoặc sai API key."}}
INDEX_RESPONSES = {503: {"model": ErrorResponse, "description": "Index chưa build hoặc không mở được."}}

bearer = HTTPBearer(
    auto_error=False,
    scheme_name="RAG_API_KEY",
    description="Giá trị của RAG_API_KEY trong .env của RAG-module.",
)


def _scope(scope: ResolvedScope) -> Scope:
    return Scope(
        status=scope.status, searchable=scope.searchable, plant=scope.plant,
        disease=scope.disease, crop=scope.crop, sources=list(scope.sources),
        message=scope.message, received=scope.received,
    )


def _chunks(result: SearchResult | None) -> list[Chunk]:
    if result is None:
        return []
    return [
        Chunk(
            rank=rank,
            source=str(hit.document.metadata.get("source", "không rõ nguồn")),
            heading_path=hit.document.metadata.get("heading_path"),
            content=hit.document.page_content,
            semantic_rank=hit.semantic_rank,
            bm25_rank=hit.bm25_rank,
            rrf_score=hit.rrf_score,
            rerank_score=hit.rerank_score,
        )
        for rank, hit in enumerate(result.hits, start=1)
    ]


def _meta(meta: RunMeta) -> ResponseMeta:
    return ResponseMeta.model_validate(asdict(meta))


def _answer_response(outcome: AnswerOutcome, debug: bool) -> AnswerResponse:
    return AnswerResponse(
        answer=outcome.answer,
        sources=outcome.sources,
        grounded=outcome.grounded,
        retrieval_query=outcome.retrieval_query,
        scope=_scope(outcome.scope),
        citations=Citations(cited=outcome.cited, invalid=outcome.invalid),
        chunks=_chunks(outcome.result) if debug else None,
        debug=outcome.result.to_debug_dict() if debug and outcome.result else None,
        meta=_meta(outcome.meta),
    )


def create_app(runtime: RAGRuntime | None = None, *, api_key: str | None = None) -> FastAPI:
    """`runtime`/`api_key` để test inject; bỏ trống thì đọc từ .env như khi chạy thật."""

    expected_key = get_api_settings().api_key if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Nạp index + embedding lúc khởi động để request đầu không chờ. Lỗi thì vẫn
        # chạy: /health và /v1/status báo lý do, request sau tự thử mở lại.
        try:
            await run_in_threadpool(app.state.runtime.load)
        except ServiceError as exc:
            logger.warning("RAG chưa sẵn sàng: %s", exc.detail)
        yield

    app = FastAPI(
        title="RAG bệnh cây API",
        version=API_VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.runtime = runtime or RAGRuntime()
    app.state.api_key = expected_key

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    def require_api_key(
        credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    ) -> None:
        if not app.state.api_key:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode(), app.state.api_key.encode()
        ):
            raise HTTPException(401, "Thiếu hoặc sai API key.",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", tags=["Hệ thống"], response_model=HealthResponse,
             summary="Process còn sống")
    def health() -> HealthResponse:
        """Không cần key, không chạm index hay LLM. Dùng cho healthcheck của container."""
        return HealthResponse()

    router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)],
                       responses=AUTH_RESPONSES)

    @router.get("/status", tags=["Hệ thống"], response_model=StatusResponse,
                summary="Index, cấu hình retrieval và LLM đang dùng")
    def status() -> StatusResponse:
        """Không gọi LLM. `ready=false` kèm lý do khi index chưa mở được."""
        runtime_ = app.state.runtime
        detail = handle = None
        try:
            handle = runtime_.load()
        except ServiceError as exc:
            detail = exc.detail
        chunk_count = None
        if handle is not None:
            try:
                chunk_count = len(handle.store.get(include=[]).get("ids") or [])
            except Exception:  # noqa: BLE001 - store giả/legacy vẫn cho xem status.
                chunk_count = None
        llm = runtime_.describe_llm()
        retrieval = get_retrieval_settings()
        fingerprint = runtime_.data_fingerprint()
        return StatusResponse(
            ready=handle is not None, detail=detail,
            index_directory=str(runtime_.index_directory), chunk_count=chunk_count,
            chunking_strategy=(handle.strategy if handle and handle.strategy else runtime_.default_index),
            embedding_model=EMBEDDING_MODEL,
            retrieval_mode=retrieval.mode, reranker_enabled=retrieval.reranker_enabled,
            llm_provider=llm["provider"], llm_model=llm["model"],
            llm_endpoint=llm["endpoint"] or llm["detail"],
            auth_enabled=bool(app.state.api_key),
            default_index=runtime_.default_index,
            indexes=[IndexStatus(**runtime_.index_status(name, fingerprint)) for name in INDEX_STRATEGIES],
        )

    @router.get("/llm", tags=["Hệ thống"], response_model=LLMInfo,
                summary="LLM của một provider và model thật đang chạy")
    def llm_info(
        provider: ProviderName | None = QueryParam(
            None, description="Bỏ trống = LLM_PROVIDER."),
        probe: bool = QueryParam(False, description=(
            "Hỏi server LLM: vLLM `GET /v1/models` (trả `root` là model thật), "
            "Ollama `/api/tags` (trả digest). Không sinh token nào.")),
    ) -> LLMInfo:
        """Dùng trước mỗi đợt chạy thí nghiệm để ghi lại đúng model đang host trên Vast."""
        return LLMInfo.model_validate(app.state.runtime.describe_llm(provider, probe=probe))

    @router.get("/taxonomy", tags=["Danh mục"], response_model=TaxonomyResponse,
                summary="Cây, bệnh, alias và bệnh nào có tài liệu")
    def taxonomy() -> TaxonomyResponse:
        """App dùng để biết trước câu hỏi nào trả lời được (`has_document`)."""
        return TaxonomyResponse(plants=taxonomy_table(), statuses=SCOPE_STATUS_MEANING)

    @router.post("/retrieve", tags=["Tra cứu"], response_model=RetrieveResponse,
                 responses=INDEX_RESPONSES, summary="Tìm tài liệu, không gọi LLM")
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        """Dùng để debug retrieval, hoặc khi phía gọi tự sinh câu trả lời (phương án D)."""
        outcome = app.state.runtime.retrieve(
            request.query, plant_type=request.plant_type, disease=request.disease,
            top_k=request.top_k, mode=request.mode, rerank=request.rerank, index=request.index,
            extra_queries=request.extra_queries,
        )
        debug = outcome.result.to_debug_dict() if request.debug and outcome.result else None
        return RetrieveResponse(scope=_scope(outcome.scope), chunks=_chunks(outcome.result),
                                debug=debug, meta=_meta(outcome.meta))

    @router.post(
        "/answer", tags=["Tra cứu"], response_model=AnswerResponse,
        summary="Tìm tài liệu rồi sinh câu trả lời có trích nguồn",
        responses={
            **INDEX_RESPONSES,
            502: {"model": ErrorResponse, "description": "LLM lỗi, sai key, hoặc không tới được."},
        },
    )
    def answer(request: AnswerRequest) -> AnswerResponse:
        """Gọi LLM theo LLM_PROVIDER, trừ khi không có tài liệu hoặc scope không cho tìm
        (`grounded=false`, câu từ chối dựng sẵn). `citations.invalid` khác rỗng là dấu
        hiệu LLM dẫn nguồn không có trong ngữ cảnh."""
        outcome = app.state.runtime.answer(
            request.query, plant_type=request.plant_type, disease=request.disease,
            history=[turn.model_dump() for turn in request.history],
            subject_context=request.subject_context, rewrite_query=request.rewrite_query,
            retrieval_query=request.retrieval_query,
            top_k=request.top_k, mode=request.mode, rerank=request.rerank,
            index=request.index, llm_provider=request.llm_provider,
            extra_queries=request.extra_queries,
        )
        return _answer_response(outcome, request.debug)

    app.include_router(router)
    return app
