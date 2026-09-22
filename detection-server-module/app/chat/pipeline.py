from app.schemas import (
    Action,
    ChatResponse,
    ChatTurn,
    PipelineDebug,
)

from app.chat.context import ContextResolver
from app.chat.detector import DiseaseDetector

from app.feedback.service import (
    save_feedback_record
)

from app.feedback.rag import (
    FeedbackRAGService
)

from app.llm.base import (
    AnswerLLM,
    QueryNormalizerLLM,
)

from app.chat.query import (
    RetrievalQueryBuilder
)

from app.chat.routing import (
    QueryRouter
)

from app.chat.session import (
    MongoSessionStore
)

from rag.service import (
    RAGService
)



class ChatService:


    def __init__(
        self,
        *,
        normalizer: QueryNormalizerLLM,
        answer_llm: AnswerLLM,
        detector: DiseaseDetector,
        rag: RAGService,
        sessions: MongoSessionStore,
        resolver: ContextResolver,
        router: QueryRouter,
        query_builder: RetrievalQueryBuilder,
        feedback_rag: FeedbackRAGService,
    ):

        self.normalizer = normalizer

        self.answer_llm = answer_llm

        self.detector = detector

        self.rag = rag

        self.sessions = sessions

        self.resolver = resolver

        self.router = router

        self.query_builder = query_builder

        self.feedback_rag = feedback_rag



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

            raise ValueError(
                "Cần nhập câu hỏi hoặc upload ảnh."
            )



        effective_query = (

            raw_query

            or

            "Ảnh này đang bị bệnh gì?"

        )



        print(f"\n[CHAT] session={session_id} query={effective_query}")



        # ==================================================
        # 1. LOAD SESSION CONTEXT
        # ==================================================


        previous_detection = await (

            self.sessions
            .get_last_detection(
                session_id
            )

        )


        recent_history = await (

            self.sessions
            .recent_history_text(
                session_id,
                limit=6
            )

        )



        session_context = (

            self._build_normalizer_context(

                previous_detection=
                    previous_detection,

                recent_history=
                    recent_history,

            )

        )



        # ==================================================
        # 2. QUERY NORMALIZATION
        # ==================================================


        analysis = await (

            self.normalizer.analyze(

                raw_query=
                    effective_query,

                session_context=
                    session_context,

            )

        )



        print(
            f"[NORMALIZED] query={analysis.normalized_query!r} "
            f"intent={analysis.intent.value} plant={analysis.plant!r} "
            f"disease={analysis.disease!r}"
        )



        # ==================================================
        # 3. IMAGE DETECTION
        # ==================================================


        current_detection = None



        if image_bytes is not None:


            current_detection = await (

                self.detector.detect(

                    image_bytes=
                        image_bytes,

                    filename=
                        image_filename,

                )

            )



            await self.sessions.set_last_detection(

                session_id,

                current_detection

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

            analysis=
                analysis,

            current_detection=
                current_detection,

            session_detection=
                active_detection,

        )



        # ==================================================
        # 5. ROUTER
        # ==================================================


        decision = self.router.decide(

            analysis=
                analysis,

            resolved=
                resolved,

            has_current_image=
                current_detection is not None,

        )


        # ==================================================
        # 6. RETRIEVAL QUERY
        # ==================================================


        retrieval_query = (

            self.query_builder.build(

                resolved=
                    resolved,

                decision=
                    decision,

                fallback_normalized_query=
                    analysis.normalized_query,

            )

        )


        print(
            f"[ROUTE] action={decision.action.value} "
            f"resolved_plant={resolved.plant!r} "
            f"resolved_disease={resolved.disease!r}"
        )
        print(f"[RAG QUERY] {retrieval_query or '(not requested)'}")



        rag_documents = []

        feedback_examples = []
        feedback_payload = {

            "query":
                analysis.normalized_query,

            "planttype":
                resolved.plant,

            "disease":
                resolved.disease,

        }



        detector_context = self._detector_text(

            current_detection

            or

            (
                active_detection
                if analysis.refers_to_previous_context
                else None
            )

        )



        answer = None



        # ==================================================
        # 7. REAL RAG RETRIEVAL
        # ==================================================


        if (

            decision.action == Action.ACCEPT_QUERY

            and

            retrieval_query

        ):


            rag_documents = await (

                self.rag.retrieve(

                    retrieval_query=
                        retrieval_query,

                    resolved=
                        resolved,

                )

            )



            print(
                f"[RAG] documents={len(rag_documents)} "
                "score=Chroma distance (lower is better)"
            )
            for index, doc in enumerate(rag_documents, start=1):
                score = "n/a" if doc.score is None else f"{doc.score:.6f}"
                print(
                    f"[RAG DOC {index}] score={score} "
                    f"source={doc.source} title={doc.title}"
                )
                print(doc.content)



            # ==================================================
            # 8. FEEDBACK RAG
            # ==================================================


            feedback_examples = await (

                self.feedback_rag.retrieve(

                    query=
                        analysis.normalized_query,

                    planttype=
                        resolved.plant,

                    disease=
                        resolved.disease,

                )

            )



            # ==================================================
            # FINAL INPUT BEFORE LLM
            # ==================================================


            print("[ANSWER LLM INPUT]")
            print(f"original_query={effective_query!r}")
            print(f"normalized_query={analysis.normalized_query!r}")
            print(f"retrieval_query={retrieval_query!r}")
            print(
                "resolved="
                f"{{plant: {resolved.plant!r}, disease: {resolved.disease!r}, "
                f"intent: {resolved.intent.value!r}, focus: {resolved.focus!r}}}"
            )
            print(f"detector_context={detector_context!r}")
            print(f"recent_history={recent_history!r}")
            print(f"rag_documents={rag_documents!r}")
            print(f"feedback_examples={feedback_examples!r}")
            print("[CALL ANSWER LLM]")



            answer = await (

                self.answer_llm.generate(

                    original_query=
                        effective_query,


                    retrieval_query=
                        retrieval_query,


                    resolved=
                        resolved,


                    rag_documents=
                        rag_documents,


                    recent_history=
                        recent_history,


                    detector_context=
                        detector_context,


                    feedback_examples=
                        feedback_examples,

                )

            )


        else:


            answer = (

                decision.message

                or

                "Bạn có thể cung cấp thêm thông tin không?"

            )



        # ==================================================
        # 9. SAVE USER MESSAGE
        # ==================================================


        await self.sessions.add_turn(

            session_id,

            ChatTurn(

                role="user",

                content=
                    effective_query,

                image_uploaded=
                    image_bytes is not None,

                image_filename=
                    image_filename,

            )

        )



        # ==================================================
        # 10. FEEDBACK RECORD
        # ==================================================


        feedback_id = await save_feedback_record(

            session_id=
                session_id,

            question=
                effective_query,

            answer=
                answer,

            metadata={

                "model":
                    "groq",


                "action":
                    decision.action.value,


                "normalized_query":
                    analysis.normalized_query,


                "plant":
                    resolved.plant,


                "disease":
                    resolved.disease,


            }

        )



        # ==================================================
        # 11. DEBUG OBJECT
        # ==================================================


        debug = PipelineDebug(

            raw_query=
                effective_query,


            normalized_query=
                analysis.normalized_query,


            intent=
                analysis.intent,


            explicit_plant=
                analysis.plant,


            explicit_disease=
                analysis.disease,


            symptoms=
                analysis.symptoms,


            focus=
                analysis.focus,


            refers_to_previous_context=
                analysis.refers_to_previous_context,


            detection=
                current_detection,


            resolved_plant=
                resolved.plant,


            resolved_disease=
                resolved.disease,


            context_source=
                resolved.context_source,


            action=
                decision.action,


            retrieval_query=
                retrieval_query,


            rag_documents=
                rag_documents,


            feedback_rag_payload=
                feedback_payload,


            feedback_rag_examples=
                feedback_examples,

        )



        # ==================================================
        # 12. SAVE ASSISTANT MESSAGE
        # ==================================================


        await self.sessions.add_turn(

            session_id,

            ChatTurn(

                role="assistant",

                content=
                    answer,


                feedback_id=
                    feedback_id,


                action=
                    decision.action,


                debug=
                    debug.model_dump(
                        mode="json"
                    ),

            )

        )



        memory = await (

            self.sessions.get_last_detection(

                session_id

            )

        )



        return ChatResponse(

            session_id=
                session_id,


            answer=
                answer,


            action=
                decision.action,


            memory=
                memory,


            debug=
                debug,


            feedback_id=
                feedback_id,

        )



    # ==================================================
    # HELPERS
    # ==================================================


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

            detection_text = (
                "last_detection: none"
            )



        return f"""

{detection_text}

recent_history:

{recent_history or "(empty)"}

""".strip()



    @staticmethod
    def _detector_text(
        detection
    ) -> str:


        if detection is None:

            return ""



        return (

            f"plant={detection.plant}; "

            f"disease={detection.disease}; "

            f"confidence={detection.confidence:.2f}; "

            f"source={detection.source}"

        )
