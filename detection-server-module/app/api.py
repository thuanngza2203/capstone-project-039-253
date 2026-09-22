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

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


from app.routes.admin import router as admin_router
from app.routes.feedback import router as feedback_router

from app.config import get_settings

from app.schemas import (
    ChatResponse,
    SessionSnapshot,
)

from app.chat.pipeline import ChatService
from app.chat.context import ContextResolver
from app.chat.detector import PlantAIDetector

from app.feedback.rag import (
    get_feedback_rag_service
)

from app.llm.groq import (
    GroqAnswerLLM,
    GroqQueryNormalizer,
)

from app.chat.query import (
    RetrievalQueryBuilder
)

from app.chat.routing import QueryRouter

from app.chat.session import (
    MongoSessionStore
)


# ==========================
# RAG REAL SERVICE
# ==========================

from rag.service import RAGService



from app.plant_ai.model_manager import (
    download_yolo_model,
    download_plant_classifier_model,
    download_ievit_model,
)



BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"



app = FastAPI(

    title="Plant Disease Chatbot",

    version="1.3.0",

    description=(

        "Plant AI chatbot with YOLO/IEViT detection + "

        "Chroma Vector RAG retrieval + "

        "MongoDB memory + Feedback RAG + "

        "Groq Answer LLM."

    ),

)



# =====================================================
# STARTUP LOAD AI MODELS
# =====================================================


@app.on_event("startup")
async def startup_ai_models():

    print("=" * 60)
    print("STARTING PLANT AI MODEL CHECK")
    print("=" * 60)


    # YOLO

    try:

        path = download_yolo_model()

        print(
            "YOLO ready:",
            path
        )

    except Exception as e:

        print(
            "YOLO error:",
            e
        )



    # Plant classifier

    try:

        path = download_plant_classifier_model()

        print(
            "Plant classifier ready:",
            path
        )

    except Exception as e:

        print(
            "Plant classifier error:",
            e
        )



    # IEViT

    plants = [

        "Apple",
        "Cherry",
        "Corn",
        "Grape",
        "Peach",
        "Pepper",
        "Potato",
        "Strawberry",
        "Tomato",

    ]


    for plant in plants:

        try:

            path = download_ievit_model(
                plant
            )

            print(
                plant,
                "ready:",
                path
            )


        except Exception as e:

            print(
                plant,
                "error:",
                e
            )



    print("=" * 60)
    print("PLANT AI MODELS READY")
    print("=" * 60)





# =====================================================
# ROUTERS
# =====================================================


app.include_router(
    feedback_router
)


app.include_router(
    admin_router
)



app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)





# =====================================================
# CACHE CONTROL
# =====================================================


@app.middleware("http")
async def disable_dev_cache(
    request,
    call_next,
):

    response = await call_next(
        request
    )


    if (

        request.url.path == "/"

        or

        request.url.path.startswith(
            "/static/"
        )

    ):

        response.headers[
            "Cache-Control"
        ] = (
            "no-store, no-cache, "
            "must-revalidate, max-age=0"
        )

        response.headers[
            "Pragma"
        ] = "no-cache"


        response.headers[
            "Expires"
        ] = "0"



    return response





# =====================================================
# SERVICES
# =====================================================


@lru_cache
def get_session_store() -> MongoSessionStore:

    settings = get_settings()


    return MongoSessionStore(

        max_turns=
            settings.max_history_turns

    )




@lru_cache
def get_chat_service() -> ChatService:


    settings = get_settings()



    return ChatService(


        # Query understanding

        normalizer=

            GroqQueryNormalizer(

                api_key=
                    settings.groq_api_key,

                model=
                    settings.normalizer_model,

            ),



        # Final answer generation

        answer_llm=

            GroqAnswerLLM(

                api_key=
                    settings.groq_api_key,

                model=
                    settings.answer_model,

            ),




        # Vision pipeline

        detector=

            PlantAIDetector(),




        # =========================
        # RAG REAL
        # =========================

        rag=

            RAGService(),




        sessions=

            get_session_store(),




        resolver=

            ContextResolver(),




        router=

            QueryRouter(),




        query_builder=

            RetrievalQueryBuilder(),




        feedback_rag=

            get_feedback_rag_service(),


    )





# =====================================================
# FRONTEND
# =====================================================


@app.get(
    "/",
    include_in_schema=False
)
async def index():

    return FileResponse(
        STATIC_DIR / "index.html"
    )





@app.get(
    "/admin",
    include_in_schema=False
)
async def admin_page():

    return FileResponse(
        STATIC_DIR / "admin_review.html"
    )





@app.get("/health")
async def health():

    return {
        "status":
            "ok"
    }





# =====================================================
# CHAT API
# =====================================================


@app.post(
    "/api/chat",
    response_model=ChatResponse
)
async def chat(

    session_id:
        Annotated[
            str,
            Form(min_length=1)
        ],


    message:
        Annotated[
            str,
            Form()
        ] = "",


    image:
        Annotated[
            UploadFile | None,
            File()
        ] = None,



    service:
        ChatService =
            Depends(
                get_chat_service
            ),

):


    try:


        image_bytes = None

        image_filename = None



        if image is not None:


            image_bytes = await image.read()


            image_filename = (
                image.filename
            )


            if not image_bytes:

                raise HTTPException(

                    status_code=400,

                    detail=
                    "Ảnh upload bị rỗng."

                )



        return await service.chat(

            session_id=session_id,

            raw_query=message,

            image_bytes=image_bytes,

            image_filename=image_filename,

        )



    except HTTPException:

        raise



    except ValueError as exc:


        raise HTTPException(

            status_code=400,

            detail=str(exc)

        )



    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {exc}"
        )





# =====================================================
# CONVERSATION API
# =====================================================


@app.get("/api/conversations")
async def list_conversations(

    store:
        MongoSessionStore =
        Depends(get_session_store),

):

    return await store.list_conversations()





@app.get(
    "/api/conversations/{session_id}"
)
async def get_conversation(

    session_id: str,

    store:
        MongoSessionStore =
        Depends(get_session_store),

):


    conversation = await (
        store.conversation_detail(
            session_id
        )
    )


    if conversation is None:

        raise HTTPException(

            status_code=404,

            detail=
            "Không tìm thấy cuộc trò chuyện"

        )


    return conversation





@app.delete(
    "/api/conversations/{session_id}"
)
async def delete_conversation(

    session_id: str,

    store:
        MongoSessionStore =
        Depends(get_session_store),

):

    await store.clear(
        session_id
    )


    return {
        "ok": True
    }





@app.get(
    "/api/session/{session_id}",
    response_model=SessionSnapshot
)
async def get_session(

    session_id: str,

    store:
        MongoSessionStore =
        Depends(get_session_store),

):

    return await store.snapshot(
        session_id
    )





@app.delete(
    "/api/session/{session_id}"
)
async def clear_session(

    session_id: str,

    store:
        MongoSessionStore =
        Depends(get_session_store),

):

    await store.clear(
        session_id
    )


    return {
        "ok": True
    }
