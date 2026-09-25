import logging
import re
import unicodedata
from typing import TYPE_CHECKING

import httpx
from pydantic import ValidationError

from app.answer.base import AnswerBackend, AnswerBackendError, AnswerContext
from app.schemas import (
    RAG_FEEDBACK_ANSWER_MAX_CHARS,
    RAG_FEEDBACK_EXAMPLES_MAX,
    RAG_FEEDBACK_QUESTION_MAX_CHARS,
    RagAnswer,
    RagAnswerRequest,
    RagFeedbackExample,
)

if TYPE_CHECKING:  # Chỉ để gợi ý kiểu: module này kéo theo sentence-transformers.
    from app.feedback.rag import FeedbackRAGService


logger = logging.getLogger(__name__)

ANSWER_PATH = "/v1/answer"

MISCONFIGURED = "Dịch vụ tra cứu chưa cấu hình đúng."
LLM_DOWN = "Model trả lời đang không phản hồi. Bạn chọn model khác hoặc thử lại sau ít phút nhé."
LLM_NOT_CONFIGURED = "Model này chưa được cấu hình trên máy chủ. Bạn chọn model khác nhé."
NOT_READY = "Dịch vụ tra cứu chưa sẵn sàng. Bạn thử lại sau ít phút nhé."
UNREACHABLE = "Không kết nối được dịch vụ tra cứu. Bạn thử lại sau ít phút nhé."
CONTRACT_ERROR = "Lỗi nội bộ khi gọi dịch vụ tra cứu."

# Nhãn [Nguồn n] RAG bắt LLM ghi (để đo trích dẫn) và cả chuỗi liền nhau: "[Nguồn 1], [Nguồn 2]",
# "[Nguồn 1][Nguồn 3]", "[Nguồn 2: apple/apple_scab.txt]". Chỉ ăn khoảng trắng cùng dòng phía trước.
_CITATIONS = re.compile(
    r"[ \t]*\[\s*Nguồn\s+\d[^\]]*\](?:[ \t]*(?:,|;|và|and)?[ \t]*\[\s*Nguồn\s+\d[^\]]*\])*",
    re.IGNORECASE,
)
# Dòng chỉ còn "Nguồn:" hoặc gạch đầu dòng rỗng sau khi bỏ nhãn.
_LEFTOVER = {"", "nguồn", "nguồn tham khảo", "tài liệu tham khảo"}


def strip_citations(answer: str) -> str:
    """Bỏ nhãn [Nguồn n] khỏi câu trả lời cho người dùng; web liệt kê tài liệu và link riêng."""
    lines = []
    for line in unicodedata.normalize("NFC", answer).splitlines():
        cleaned = _CITATIONS.sub("", line)
        if cleaned != line and cleaned.strip(" \t-*•_:.,;").casefold() in _LEFTOVER:
            continue
        lines.append(cleaned.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text or answer.strip()


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
        feedback_rag: "FeedbackRAGService | None" = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        # Test truyền httpx.MockTransport; chạy thật dùng transport mặc định.
        self._transport = transport
        # Câu trả lời mẫu admin đã duyệt, gửi kèm để RAG đưa vào prompt (như backend groq).
        self.feedback_rag = feedback_rag

    async def _feedback_examples(self, context: AnswerContext | None) -> list[dict]:
        """Lỗi Feedback RAG (Mongo, tải model) không được làm hỏng câu trả lời: bỏ qua ví dụ."""
        if self.feedback_rag is None or context is None:
            return []
        try:
            return await self.feedback_rag.retrieve(
                query=context.normalized_query,
                planttype=context.resolved.plant,
                disease=context.resolved.disease,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Feedback RAG failed; answering without admin examples")
            return []

    async def answer(
        self,
        request: RagAnswerRequest,
        context: AnswerContext | None = None,
    ) -> RagAnswer:
        examples = [
            example for example in await self._feedback_examples(context)
            if (example.get("question") or "").strip() and (example.get("preferred_answer") or "").strip()
        ][:RAG_FEEDBACK_EXAMPLES_MAX]
        logger.info("Feedback RAG examples sent to RAG=%d", len(examples))
        if examples:
            request = request.model_copy(update={"feedback_examples": [
                RagFeedbackExample(
                    question=example["question"].strip()[:RAG_FEEDBACK_QUESTION_MAX_CHARS],
                    answer=example["preferred_answer"].strip()[:RAG_FEEDBACK_ANSWER_MAX_CHARS],
                )
                for example in examples
            ]})

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
            llm = (data.get("meta") or {}).get("llm") or {}
            result = RagAnswer(
                answer=strip_citations(data["answer"]),
                sources=data.get("sources") or [],
                # RAG trước 1.4.0 chưa có `documents`: web tự hiện tên file từ `sources`.
                documents=data.get("documents") or [],
                grounded=data["grounded"],
                scope_status=data["scope"]["status"],
                llm_provider=llm.get("provider"),
                llm_model=llm.get("model"),
                # Để debug/admin thấy mẫu nào đã được gửi (như backend groq).
                feedback_examples=examples,
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
            # RAG dùng 503 cho cả index chưa mở được lẫn LLM chưa cấu hình (ví dụ thiếu
            # GEMINI_API_KEY); detail của lỗi LLM bắt đầu bằng "LLM".
            try:
                detail = str(response.json().get("detail", ""))
            except (ValueError, AttributeError):
                detail = ""
            if detail.startswith("LLM"):
                logger.warning("RAG LLM not configured (503): %s", body)
                return AnswerBackendError(503, LLM_NOT_CONFIGURED)
            logger.warning("RAG index not ready (503): %s", body)
            return AnswerBackendError(503, NOT_READY)
        logger.error("RAG unexpected status %s: %s", status, body)
        return AnswerBackendError(503 if status >= 500 else 500, CONTRACT_ERROR)
