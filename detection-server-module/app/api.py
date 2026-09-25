import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes.admin import router as admin_router
from app.routes.auth import router as auth_router
from app.routes.feedback import router as feedback_router

from app.config import Settings, get_settings

from app.schemas import (
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    HealthResponse,
    ModelOptions,
    OkResponse,
    SessionSnapshot,
)

from app.answer.base import AnswerBackend, AnswerBackendError
from app.answer.rag_http import RagHttpBackend
from app.answer.web_search import GroqWebSearch
from app.chat.pipeline import ChatService
from app.chat.context import ContextResolver
from app.chat.detector import PlantAIDetector
from app.chat.query import RetrievalQueryBuilder
from app.chat.routing import QueryRouter
from app.chat.session import MongoSessionStore
from app.llm.groq import GroqQueryNormalizer

from app.security import admin_auth_enabled, require_admin

from app.plant_ai.model_manager import (
    IEVIT_MODELS,
    download_ievit_model,
    download_plant_classifier_model,
    download_yolo_model,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"


# =====================================================
# STARTUP LOAD AI MODELS
# =====================================================


def check_ai_models() -> None:
    """Tải checkpoint còn thiếu; lỗi từng model chỉ ghi log, không chặn khởi động."""
    logger.info("Starting plant AI model check")

    for name, download in (
        ("YOLO", download_yolo_model),
        ("Plant classifier", download_plant_classifier_model),
    ):
        try:
            logger.info("%s ready: %s", name, download())
        except Exception:
            logger.exception("%s download failed", name)

    for plant in IEVIT_MODELS:
        try:
            logger.info("%s ready: %s", plant, download_ievit_model(plant))
        except Exception:
            logger.exception("%s download failed", plant)

    logger.info("Plant AI models ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Answer backend: %s", settings.answer_backend)
    if not admin_auth_enabled():
        logger.warning(
            "ADMIN_PASSWORD trống: /admin và /admin/* không có xác thực. "
            "Đặt ADMIN_PASSWORD trong .env trước khi mở ra ngoài."
        )
    await asyncio.to_thread(check_ai_models)
    yield


app = FastAPI(
    title="Plant Disease Chatbot",
    version="1.4.0",
    description=(
        "Plant AI chatbot with YOLO/IEViT detection + "
        "MongoDB memory + Feedback RAG. Answers come from the internal "
        "Chroma RAG + Groq (ANSWER_BACKEND=groq) or the RAG-module server "
        "(ANSWER_BACKEND=rag)."
    ),
    lifespan=lifespan,
)


# =====================================================
# CORS: chỉ các origin trong CORS_ORIGINS, không dùng "*"
# =====================================================

_cors_origins = get_settings().cors_origin_list

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )


# =====================================================
# ROUTERS
# =====================================================


app.include_router(feedback_router)

app.include_router(auth_router)

app.include_router(admin_router)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# =====================================================
# CACHE CONTROL
# =====================================================


@app.middleware("http")
async def disable_dev_cache(request, call_next):

    response = await call_next(request)

    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


# =====================================================
# SERVICES
# =====================================================


@lru_cache
def get_session_store() -> MongoSessionStore:

    settings = get_settings()

    return MongoSessionStore(max_turns=settings.max_history_turns)


def build_answer_backend(settings: Settings) -> AnswerBackend:
    """ANSWER_BACKEND=groq: như trước refactor; rag: gọi RAG-module server."""

    if settings.answer_backend == "rag":
        from app.feedback.rag import get_feedback_rag_service

        return RagHttpBackend(
            base_url=settings.rag_api_url,
            api_key=settings.rag_api_key,
            timeout=settings.rag_api_timeout,
            # Câu trả lời mẫu admin đã duyệt được gửi kèm sang RAG.
            feedback_rag=get_feedback_rag_service(),
        )

    # Import muộn: chế độ rag không cần nạp Chroma/embedding của rag/ nội bộ.
    from app.answer.groq import GroqAnswerBackend
    from app.feedback.rag import get_feedback_rag_service
    from app.llm.groq import GroqAnswerLLM
    from rag.service import RAGService

    return GroqAnswerBackend(
        rag=RAGService(),
        feedback_rag=get_feedback_rag_service(),
        answer_llm=GroqAnswerLLM(
            api_key=settings.groq_api_key,
            model=settings.answer_model,
        ),
    )


@lru_cache
def get_chat_service() -> ChatService:

    settings = get_settings()

    return ChatService(
        # Query understanding
        normalizer=GroqQueryNormalizer(
            api_key=settings.groq_api_key,
            model=settings.normalizer_model,
            reasoning_effort=settings.normalizer_reasoning_effort.strip() or None,
        ),
        # Retrieval + final answer generation
        answer_backend=build_answer_backend(settings),
        # Vision pipeline
        detector=PlantAIDetector(),
        sessions=get_session_store(),
        resolver=ContextResolver(),
        router=QueryRouter(),
        query_builder=RetrievalQueryBuilder(
            search_original_query=settings.rag_search_original_query,
        ),
        # Nút "Tìm trên web": Groq tìm web và trả lời thẳng.
        web_search=GroqWebSearch(
            api_key=settings.groq_api_key,
            model=settings.web_search_model.strip(),
        ) if settings.web_search_enabled else None,
    )


# =====================================================
# FRONTEND
# =====================================================


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False, dependencies=[Depends(require_admin)])
async def admin_page():
    return FileResponse(STATIC_DIR / "admin_review.html")


@app.get("/health", response_model=HealthResponse, tags=["Hệ thống"])
async def health():
    return HealthResponse()


# =====================================================
# CHAT API
# =====================================================


@app.get("/api/models", response_model=ModelOptions, tags=["Chat"])
async def models():
    """Model trả lời người dùng chọn được (CHAT_LLM_PROVIDERS; rỗng khi ANSWER_BACKEND=groq)
    và có nút "Tìm trên web" hay không (WEB_SEARCH_MODEL)."""
    settings = get_settings()
    providers = settings.chat_llm_provider_list
    return ModelOptions(
        answer_backend=settings.answer_backend,
        providers=providers,
        default=providers[0] if providers else None,
        web_search=settings.web_search_enabled,
    )


def resolve_llm_provider(settings: Settings, requested: str) -> str | None:
    """Model gửi sang RAG; bỏ trống = model đầu của CHAT_LLM_PROVIDERS (hoặc mặc định của RAG)."""
    allowed = settings.chat_llm_provider_list
    if settings.answer_backend != "rag":
        return None  # Groq trả lời: không có model nào để chọn.
    requested = requested.strip().lower()
    if not requested:
        return allowed[0] if allowed else None
    if requested not in allowed:
        raise HTTPException(status_code=400, detail=f"Model '{requested}' không được phép dùng.")
    return requested


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    session_id: Annotated[str, Form(min_length=1)],
    message: Annotated[str, Form()] = "",
    image: Annotated[UploadFile | None, File()] = None,
    llm_provider: Annotated[str, Form(description="Một giá trị của GET /api/models; bỏ trống = mặc định.")] = "",
    web_search: Annotated[bool, Form(description="true: Groq tìm web và trả lời, không qua RAG.")] = False,
    service: ChatService = Depends(get_chat_service),
):

    try:

        settings = get_settings()
        if web_search and not settings.web_search_enabled:
            raise HTTPException(status_code=400, detail="Tìm trên web chưa được bật trên máy chủ.")
        # Tìm trên web luôn trả lời bằng Groq: model đã chọn không dùng tới.
        provider = None if web_search else resolve_llm_provider(settings, llm_provider)

        image_bytes = None
        image_filename = None

        if image is not None:
            image_bytes = await image.read()
            image_filename = image.filename

            if not image_bytes:
                raise HTTPException(status_code=400, detail="Ảnh upload bị rỗng.")

        return await service.chat(
            session_id=session_id,
            raw_query=message,
            image_bytes=image_bytes,
            image_filename=image_filename,
            llm_provider=provider,
            web_search=web_search,
        )

    except HTTPException:
        raise

    except AnswerBackendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        # Chi tiết chỉ nằm trong log server, không trả nguyên văn exception cho client.
        logger.exception("Chat pipeline failed for session=%s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Hệ thống gặp lỗi khi xử lý câu hỏi. Bạn thử lại sau nhé.",
        ) from exc


# =====================================================
# CONVERSATION API
# =====================================================


@app.get("/api/conversations", response_model=list[ConversationSummary], tags=["Hội thoại"])
async def list_conversations(
    store: MongoSessionStore = Depends(get_session_store),
):
    return await store.list_conversations()


@app.get("/api/conversations/{session_id}", response_model=ConversationDetail, tags=["Hội thoại"])
async def get_conversation(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):

    conversation = await store.conversation_detail(session_id)

    if conversation is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện")

    return conversation


@app.delete("/api/conversations/{session_id}", response_model=OkResponse, tags=["Hội thoại"])
async def delete_conversation(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):
    await store.clear(session_id)
    return OkResponse()


@app.get("/api/session/{session_id}", response_model=SessionSnapshot, tags=["Hội thoại"])
async def get_session(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):
    return await store.snapshot(session_id)


@app.delete("/api/session/{session_id}", response_model=OkResponse, tags=["Hội thoại"])
async def clear_session(
    session_id: str,
    store: MongoSessionStore = Depends(get_session_store),
):
    await store.clear(session_id)
    return OkResponse()
