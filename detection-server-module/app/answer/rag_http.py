import logging

import httpx
from pydantic import ValidationError

from app.answer.base import AnswerBackend, AnswerBackendError, AnswerContext
from app.schemas import RagAnswer, RagAnswerRequest


logger = logging.getLogger(__name__)

ANSWER_PATH = "/v1/answer"

MISCONFIGURED = "Dịch vụ tra cứu chưa cấu hình đúng."
LLM_DOWN = "Máy chủ trả lời tạm thời không phản hồi. Bạn thử lại sau ít phút nhé."
NOT_READY = "Dịch vụ tra cứu chưa sẵn sàng. Bạn thử lại sau ít phút nhé."
UNREACHABLE = "Không kết nối được dịch vụ tra cứu. Bạn thử lại sau ít phút nhé."
CONTRACT_ERROR = "Lỗi nội bộ khi gọi dịch vụ tra cứu."


class RagHttpBackend(AnswerBackend):
    """Gọi `POST {RAG_API_URL}/v1/answer` của RAG-module (xem file OpenAPI)."""

    name = "rag"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 150.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        # Test truyền httpx.MockTransport; chạy thật dùng transport mặc định.
        self._transport = transport

    async def answer(
        self,
        request: RagAnswerRequest,
        context: AnswerContext | None = None,
    ) -> RagAnswer:
        del context  # RAG server chỉ nhận đúng RagAnswerRequest.

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = request.model_dump(mode="json", exclude_none=True)

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(ANSWER_PATH, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("RAG timeout after %ss: %s", self.timeout, exc)
            raise AnswerBackendError(503, UNREACHABLE) from exc
        except httpx.HTTPError as exc:
            logger.warning("RAG unreachable at %s: %s", self.base_url, exc)
            raise AnswerBackendError(503, UNREACHABLE) from exc

        if response.status_code != 200:
            raise self._error(response)

        try:
            data = response.json()
            result = RagAnswer(
                answer=data["answer"],
                sources=data.get("sources") or [],
                grounded=data["grounded"],
                scope_status=data["scope"]["status"],
            )
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            logger.error("RAG returned an invalid body: %s", response.text[:2000])
            raise AnswerBackendError(503, NOT_READY) from exc

        logger.info(
            "RAG answer grounded=%s scope=%s sources=%s",
            result.grounded, result.scope_status, result.sources,
        )
        return result

    @staticmethod
    def _error(response: httpx.Response) -> AnswerBackendError:
        status = response.status_code
        body = response.text[:2000]
        if status == 401:
            logger.error("RAG rejected RAG_API_KEY (401)")
            return AnswerBackendError(503, MISCONFIGURED)
        if status == 422:
            # Payload sai hợp đồng: lỗi code bên detection.
            logger.error("RAG rejected payload (422): %s", body)
            return AnswerBackendError(500, CONTRACT_ERROR)
        if status == 502:
            logger.warning("RAG LLM error (502): %s", body)
            return AnswerBackendError(503, LLM_DOWN)
        if status == 503:
            logger.warning("RAG index not ready (503): %s", body)
            return AnswerBackendError(503, NOT_READY)
        logger.error("RAG unexpected status %s: %s", status, body)
        return AnswerBackendError(503 if status >= 500 else 500, CONTRACT_ERROR)
