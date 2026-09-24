import logging
from collections.abc import Awaitable, Callable

from app.schemas import (
    Action,
    ChatResponse,
    ChatTurn,
    Intent,
    PipelineDebug,
    QueryAnalysis,
    RAG_EXTRA_QUERIES_MAX,
    RAG_HISTORY_CONTENT_MAX_CHARS,
    RAG_HISTORY_MAX_MESSAGES,
    RAG_QUERY_MAX_CHARS,
    RAG_SUBJECT_CONTEXT_MAX_CHARS,
    RagAnswer,
    RagAnswerRequest,
    RagChatMessage,
    ResolvedQuery,
    RouteDecision,
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
        # Groq lỗi/hết quota thì không trả 500: câu gốc đi thẳng sang RAG (bước 5–6).
        # ==================================================

        normalizer_failed = False
        try:
            analysis = await self.normalizer.analyze(
                raw_query=effective_query,
                session_context=session_context,
            )
        except Exception:  # noqa: BLE001 - lỗi mạng, quota, JSON sai đều xử lý như nhau.
            logger.exception("Normalizer failed; sending the raw query to RAG")
            analysis = self.fallback_analysis(effective_query)
            normalizer_failed = True

        if not analysis.normalized_query.strip():
            analysis = analysis.model_copy(update={"normalized_query": effective_query})

        logger.info(
            "Normalized query=%r intent=%s plant=%r disease=%r named=%s",
            analysis.normalized_query, analysis.intent.value, analysis.plant, analysis.disease,
            analysis.disease_named,
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

        if normalizer_failed:
            # Không có intent/cây/bệnh để route: để RAG tự tìm bằng câu gốc.
            decision = RouteDecision(action=Action.ACCEPT_QUERY)
        else:
            decision = self.router.decide(
                analysis=analysis,
                resolved=resolved,
                has_current_image=current_detection is not None,
            )

        # ==================================================
        # 6. RETRIEVAL QUERY
        # Câu chuẩn hóa là câu tìm; cây/bệnh (kể cả từ ảnh/session) đi riêng trong
        # plant_type/disease để RAG khoanh phạm vi. Groq lỗi: không có câu chuẩn
        # hóa, RAG tìm bằng câu gốc và tự viết lại câu nối tiếp (rewrite_query).
        # ==================================================

        retrieval_query = None
        extra_queries: list[str] = []
        if not normalizer_failed:
            retrieval_query = self.query_builder.build(
                analysis=analysis,
                decision=decision,
            )
            extra_queries = self.query_builder.extra_queries(
                raw_query=effective_query,
                analysis=analysis,
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

        if decision.action == Action.ACCEPT_QUERY and (retrieval_query or normalizer_failed):
            history = await self.sessions.recent_messages(
                session_id,
                limit=RAG_HISTORY_MAX_MESSAGES,
            )
            rag_request = self.build_rag_request(
                query=effective_query,
                retrieval_query=retrieval_query,
                extra_queries=extra_queries,
                rewrite_query=normalizer_failed,
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
            disease_named=analysis.disease_named,
            suspected_disease=resolved.suspected_disease,
            normalizer_failed=normalizer_failed,
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
        retrieval_query: str | None,
        resolved: ResolvedQuery,
        history: list[dict],
        subject_context: str,
        extra_queries: list[str] | None = None,
        rewrite_query: bool = False,
    ) -> RagAnswerRequest:
        """Payload đúng `AnswerRequest` của RAG; cắt theo giới hạn của hợp đồng.

        plant_type/disease gửi nguyên nhãn (RAG tự nhận `Apple___Apple_scab`...).
        rewrite_query chỉ bật khi Groq lỗi; bình thường normalizer ở đây đã giải
        quyết "bệnh này", "nó" nên RAG không cần viết lại.
        """
        messages = [
            RagChatMessage(
                role=message["role"],
                content=message["content"][:RAG_HISTORY_CONTENT_MAX_CHARS],
            )
            for message in history[-RAG_HISTORY_MAX_MESSAGES:]
        ]
        extras = [text[:RAG_QUERY_MAX_CHARS] for text in extra_queries or []][:RAG_EXTRA_QUERIES_MAX]
        return RagAnswerRequest(
            query=query[:RAG_QUERY_MAX_CHARS],
            retrieval_query=retrieval_query[:RAG_QUERY_MAX_CHARS] if retrieval_query else None,
            extra_queries=extras or None,
            rewrite_query=True if rewrite_query else None,
            plant_type=resolved.plant[:100] if resolved.plant else None,
            disease=resolved.disease[:100] if resolved.disease else None,
            history=messages,
            subject_context=subject_context[:RAG_SUBJECT_CONTEXT_MAX_CHARS] or None,
        )

    @staticmethod
    def fallback_analysis(query: str) -> QueryAnalysis:
        """Thay kết quả Groq khi Groq lỗi: không đoán cây/bệnh, không dùng session.

        Cây/bệnh chỉ còn từ ảnh của lượt này (ContextResolver); RAG tìm bằng câu gốc.
        """
        return QueryAnalysis(
            normalized_query=query,
            plant=None,
            disease=None,
            disease_named=False,
            symptoms=[],
            intent=Intent.GENERAL_INFO,
            focus=None,
            refers_to_previous_context=False,
            is_plant_related=True,
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
