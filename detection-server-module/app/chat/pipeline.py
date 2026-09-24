import logging
from collections.abc import Awaitable, Callable

from app.schemas import (
    Action,
    ChatResponse,
    ChatTurn,
    PipelineDebug,
    RAG_HISTORY_CONTENT_MAX_CHARS,
    RAG_HISTORY_MAX_MESSAGES,
    RAG_QUERY_MAX_CHARS,
    RAG_SUBJECT_CONTEXT_MAX_CHARS,
    RagAnswer,
    RagAnswerRequest,
    RagChatMessage,
    ResolvedQuery,
)

from app.answer.base import AnswerBackend, AnswerContext
from app.chat.context import ContextResolver
from app.chat.detector import DiseaseDetector
from app.chat.query import RetrievalQueryBuilder
from app.chat.routing import QueryRouter
from app.chat.session import MongoSessionStore
from app.feedback.service import save_feedback_record
from app.llm.base import QueryNormalizerLLM


logger = logging.getLogger(__name__)

FeedbackRecorder = Callable[..., Awaitable[str]]


class ChatService:

    def __init__(
        self,
        *,
        normalizer: QueryNormalizerLLM,
        answer_backend: AnswerBackend,
        detector: DiseaseDetector,
        sessions: MongoSessionStore,
        resolver: ContextResolver,
        router: QueryRouter,
        query_builder: RetrievalQueryBuilder,
        feedback_recorder: FeedbackRecorder = save_feedback_record,
    ):
        self.normalizer = normalizer
        self.answer_backend = answer_backend
        self.detector = detector
        self.sessions = sessions
        self.resolver = resolver
        self.router = router
        self.query_builder = query_builder
        # Tách ra để test offline không cần MongoDB.
        self.feedback_recorder = feedback_recorder

    async def chat(
        self,
        *,
        session_id: str,
        raw_query: str,
        image_bytes: bytes | None = None,
        image_filename: str | None = None,
    ) -> ChatResponse:

        raw_query = raw_query.strip()

        if not raw_query and image_bytes is None:
            raise ValueError("Cần nhập câu hỏi hoặc upload ảnh.")

        effective_query = raw_query or "Ảnh này đang bị bệnh gì?"

        logger.info("Chat session=%s query=%r", session_id, effective_query)

        # ==================================================
        # 1. LOAD SESSION CONTEXT
        # ==================================================

        previous_detection = await self.sessions.get_last_detection(session_id)

        # MAX_HISTORY_TURNS: số message gần nhất đưa vào normalizer và answer LLM.
        recent_history = await self.sessions.recent_history_text(
            session_id,
            limit=self.sessions.max_turns,
        )

        session_context = self._build_normalizer_context(
            previous_detection=previous_detection,
            recent_history=recent_history,
        )

        # ==================================================
        # 2. IMAGE DETECTION
        # Chạy trước chuẩn hóa: ảnh không phải lá (InvalidImageError -> HTTP 400)
        # bị trả về ngay, không gọi Groq và không lưu gì của lượt này.
        # ==================================================

        current_detection = None

        if image_bytes is not None:
            current_detection = await self.detector.detect(
                image_bytes=image_bytes,
                filename=image_filename,
            )
            # Chưa lưu last_detection ở đây: nếu Groq/RAG lỗi phía sau thì Mongo
            # không được còn lại hội thoại rỗng. Lưu ở bước 9 khi đã có câu trả lời.

        # ==================================================
        # 3. QUERY NORMALIZATION
        # ==================================================

        analysis = await self.normalizer.analyze(
            raw_query=effective_query,
            session_context=session_context,
        )

        logger.info(
            "Normalized query=%r intent=%s plant=%r disease=%r",
            analysis.normalized_query, analysis.intent.value, analysis.plant, analysis.disease,
        )

        active_detection = (
            current_detection
            if current_detection is not None
            else previous_detection
        )

        # ==================================================
        # 4. RESOLVE CONTEXT
        # ==================================================

        resolved = self.resolver.resolve(
            analysis=analysis,
            current_detection=current_detection,
            session_detection=active_detection,
        )

        # ==================================================
        # 5. ROUTER
        # ==================================================

        decision = self.router.decide(
            analysis=analysis,
            resolved=resolved,
            has_current_image=current_detection is not None,
        )

        # ==================================================
        # 6. RETRIEVAL QUERY
        # ==================================================

        retrieval_query = self.query_builder.build(
            resolved=resolved,
            decision=decision,
            fallback_normalized_query=analysis.normalized_query,
        )

        logger.info(
            "Route action=%s resolved_plant=%r resolved_disease=%r retrieval_query=%r",
            decision.action.value, resolved.plant, resolved.disease, retrieval_query,
        )

        feedback_payload = {
            "query": analysis.normalized_query,
            "planttype": resolved.plant,
            "disease": resolved.disease,
        }

        detector_context = self._detector_text(
            current_detection
            or (
                active_detection
                if analysis.refers_to_previous_context
                else None
            )
        )

        # ==================================================
        # 7–8. RETRIEVAL + ANSWER (AnswerBackend)
        # Chỉ ACCEPT_QUERY mới gọi backend; các nhánh khác trả câu dựng sẵn.
        # Backend lỗi thì exception đi thẳng lên route: chưa lưu gì của lượt này.
        # ==================================================

        rag_request: RagAnswerRequest | None = None
        result: RagAnswer | None = None

        if decision.action == Action.ACCEPT_QUERY and retrieval_query:
            history = await self.sessions.recent_messages(
                session_id,
                limit=RAG_HISTORY_MAX_MESSAGES,
            )
            rag_request = self.build_rag_request(
                query=effective_query,
                retrieval_query=retrieval_query,
                resolved=resolved,
                history=history,
                subject_context=detector_context,
            )
            result = await self.answer_backend.answer(
                rag_request,
                AnswerContext(
                    resolved=resolved,
                    normalized_query=analysis.normalized_query,
                    recent_history=recent_history,
                    detector_context=detector_context,
                ),
            )
            answer = result.answer
        else:
            answer = decision.message or "Bạn có thể cung cấp thêm thông tin không?"

        # ==================================================
        # 9. SAVE DETECTION + USER MESSAGE
        # ==================================================

        if current_detection is not None:
            await self.sessions.set_last_detection(session_id, current_detection)

        await self.sessions.add_turn(
            session_id,
            ChatTurn(
                role="user",
                content=effective_query,
                image_uploaded=image_bytes is not None,
                image_filename=image_filename,
            ),
        )

        # ==================================================
        # 10. FEEDBACK RECORD
        # ==================================================

        feedback_id = await self.feedback_recorder(
            session_id=session_id,
            question=effective_query,
            answer=answer,
            metadata={
                "model": self.answer_backend.name,
                "action": decision.action.value,
                "normalized_query": analysis.normalized_query,
                "plant": resolved.plant,
                "disease": resolved.disease,
                # Để admin biết câu trả lời dựa trên tài liệu nào.
                "sources": result.sources if result else [],
                "scope_status": result.scope_status if result else None,
                "grounded": result.grounded if result else None,
            },
        )

        # ==================================================
        # 11. DEBUG OBJECT
        # ==================================================

        debug = PipelineDebug(
            raw_query=effective_query,
            normalized_query=analysis.normalized_query,
            intent=analysis.intent,
            explicit_plant=analysis.plant,
            explicit_disease=analysis.disease,
            symptoms=analysis.symptoms,
            focus=analysis.focus,
            refers_to_previous_context=analysis.refers_to_previous_context,
            detection=current_detection,
            resolved_plant=resolved.plant,
            resolved_disease=resolved.disease,
            context_source=resolved.context_source,
            action=decision.action,
            retrieval_query=retrieval_query,
            rag_documents=result.rag_documents if result else [],
            feedback_rag_payload=feedback_payload,
            feedback_rag_examples=result.feedback_examples if result else [],
            answer_backend=self.answer_backend.name,
            rag_request=rag_request.model_dump(mode="json", exclude_none=True) if rag_request else None,
            rag_sources=result.sources if result else [],
            rag_grounded=result.grounded if result else None,
            rag_scope_status=result.scope_status if result else None,
        )

        # ==================================================
        # 12. SAVE ASSISTANT MESSAGE
        # ==================================================

        await self.sessions.add_turn(
            session_id,
            ChatTurn(
                role="assistant",
                content=answer,
                feedback_id=feedback_id,
                action=decision.action,
                debug=debug.model_dump(mode="json"),
            ),
        )

        memory = await self.sessions.get_last_detection(session_id)

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            action=decision.action,
            memory=memory,
            debug=debug,
            feedback_id=feedback_id,
            sources=result.sources if result else [],
        )

    # ==================================================
    # HELPERS
    # ==================================================

    @staticmethod
    def build_rag_request(
        *,
        query: str,
        retrieval_query: str,
        resolved: ResolvedQuery,
        history: list[dict],
        subject_context: str,
    ) -> RagAnswerRequest:
        """Payload đúng `AnswerRequest` của RAG; cắt theo giới hạn của hợp đồng.

        plant_type/disease gửi nguyên nhãn (RAG tự nhận `Apple___Apple_scab`...).
        Không gửi rewrite_query: normalizer ở đây đã giải quyết "bệnh này", "nó".
        """
        messages = [
            RagChatMessage(
                role=message["role"],
                content=message["content"][:RAG_HISTORY_CONTENT_MAX_CHARS],
            )
            for message in history[-RAG_HISTORY_MAX_MESSAGES:]
        ]
        return RagAnswerRequest(
            query=query[:RAG_QUERY_MAX_CHARS],
            retrieval_query=retrieval_query[:RAG_QUERY_MAX_CHARS],
            plant_type=resolved.plant[:100] if resolved.plant else None,
            disease=resolved.disease[:100] if resolved.disease else None,
            history=messages,
            subject_context=subject_context[:RAG_SUBJECT_CONTEXT_MAX_CHARS] or None,
        )

    @staticmethod
    def _build_normalizer_context(
        *,
        previous_detection,
        recent_history: str,
    ) -> str:

        if previous_detection:
            detection_text = (
                f"last_detection: "
                f"plant={previous_detection.plant}, "
                f"disease={previous_detection.disease}, "
                f"confidence={previous_detection.confidence}"
            )
        else:
            detection_text = "last_detection: none"

        return f"""

{detection_text}

recent_history:

{recent_history or "(empty)"}

""".strip()

    @staticmethod
    def _detector_text(detection) -> str:

        if detection is None:
            return ""

        return (
            f"plant={detection.plant}; "
            f"disease={detection.disease}; "
            f"confidence={detection.confidence:.2f}; "
            f"source={detection.source}"
        )
