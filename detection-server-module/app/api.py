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
from app.routes.feedback import router as feedback_router

from app.config import Settings, get_settings

from app.schemas import (
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    HealthResponse,
    OkResponse,
    SessionSnapshot,
)

from app.answer.base import AnswerBackend, AnswerBackendError
from app.answer.rag_http import RagHttpBackend
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
        return RagHttpBackend(
            base_url=settings.rag_api_url,
            api_key=settings.rag_api_key,
            timeout=settings.rag_api_timeout,
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
        ),
        # Retrieval + final answer generation
        answer_backend=build_answer_backend(settings),
        # Vision pipeline
        detector=PlantAIDetector(),
        sessions=get_session_store(),
        resolver=ContextResolver(),
        router=QueryRouter(),
        query_builder=RetrievalQueryBuilder(),
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


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    session_id: Annotated[str, Form(min_length=1)],
    message: Annotated[str, Form()] = "",
    image: Annotated[UploadFile | None, File()] = None,
    service: ChatService = Depends(get_chat_service),
):

    try:

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
