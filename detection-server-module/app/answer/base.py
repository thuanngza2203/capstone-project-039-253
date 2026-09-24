from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import RagAnswer, RagAnswerRequest, ResolvedQuery


@dataclass(frozen=True)
class AnswerContext:
    """Dữ liệu pipeline có sẵn nhưng không nằm trong hợp đồng RAG.

    Chỉ GroqAnswerBackend dùng, để giữ y hệt prompt trước refactor.
    RagHttpBackend bỏ qua: RAG server chỉ nhận RagAnswerRequest.
    """

    resolved: ResolvedQuery
    normalized_query: str
    recent_history: str
    detector_context: str


class AnswerBackendError(Exception):
    """Lỗi đã được ánh xạ sang HTTP status trả cho client của detection."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class AnswerBackend(ABC):
    name: str

    @abstractmethod
    async def answer(
        self,
        request: RagAnswerRequest,
        context: AnswerContext | None = None,
    ) -> RagAnswer:
        raise NotImplementedError
