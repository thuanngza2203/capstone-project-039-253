import logging

from app.answer.base import AnswerBackend, AnswerContext
from app.feedback.rag import FeedbackRAGService
from app.llm.base import AnswerLLM
from app.schemas import RagAnswer, RagAnswerRequest
from rag.service import RAGService


logger = logging.getLogger(__name__)


class GroqAnswerBackend(AnswerBackend):
    """Luồng trước refactor: rag/ nội bộ + Feedback RAG + GroqAnswerLLM."""

    name = "groq"

    def __init__(
        self,
        *,
        rag: RAGService,
        feedback_rag: FeedbackRAGService,
        answer_llm: AnswerLLM,
    ):
        self.rag = rag
        self.feedback_rag = feedback_rag
        self.answer_llm = answer_llm

    async def answer(
        self,
        request: RagAnswerRequest,
        context: AnswerContext | None = None,
    ) -> RagAnswer:
        if context is None:
            raise ValueError("GroqAnswerBackend cần AnswerContext từ pipeline.")

        resolved = context.resolved
        retrieval_query = request.retrieval_query or request.query

        rag_documents = await self.rag.retrieve(
            retrieval_query=retrieval_query,
            resolved=resolved,
        )
        logger.info(
            "RAG documents=%d score=Chroma distance (lower is better)",
            len(rag_documents),
        )
        for index, doc in enumerate(rag_documents, start=1):
            score = "n/a" if doc.score is None else f"{doc.score:.6f}"
            logger.info("RAG doc %d score=%s source=%s", index, score, doc.source)
            logger.debug("RAG doc %d content=%s", index, doc.content)

        feedback_examples = await self.feedback_rag.retrieve(
            query=context.normalized_query,
            planttype=resolved.plant,
            disease=resolved.disease,
        )
        logger.info("Feedback RAG examples=%d", len(feedback_examples))
        logger.debug("Answer LLM recent_history=%r", context.recent_history)

        answer = await self.answer_llm.generate(
            original_query=request.query,
            retrieval_query=retrieval_query,
            resolved=resolved,
            rag_documents=rag_documents,
            recent_history=context.recent_history,
            detector_context=context.detector_context,
            feedback_examples=feedback_examples,
        )

        return RagAnswer(
            answer=answer,
            sources=list(dict.fromkeys(doc.source for doc in rag_documents)),
            grounded=bool(rag_documents),
            scope_status="internal",
            rag_documents=rag_documents,
            feedback_examples=feedback_examples,
        )
