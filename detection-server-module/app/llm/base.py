from abc import ABC, abstractmethod

from app.schemas import QueryAnalysis, RagDocument, ResolvedQuery


class QueryNormalizerLLM(ABC):
    @abstractmethod
    async def analyze(
        self,
        raw_query: str,
        session_context: str,
    ) -> QueryAnalysis:
        raise NotImplementedError


class AnswerLLM(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        original_query: str,
        retrieval_query: str,
        resolved: ResolvedQuery,
        rag_documents: list[RagDocument],
        recent_history: str,
        detector_context: str,
        feedback_examples: list[dict] | None = None,
    ) -> str:
        raise NotImplementedError
