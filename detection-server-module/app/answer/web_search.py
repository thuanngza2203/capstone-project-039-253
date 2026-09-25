"""Nút "Tìm trên web": Groq tự tìm web (tool browser_search) rồi trả lời, không qua RAG."""

import logging
import re
from urllib.parse import urlparse

from groq import AsyncGroq, GroqError

from app.answer.base import AnswerBackendError
from app.prompts.web_search_prompt import WEB_SEARCH_SYSTEM_PROMPT
from app.schemas import RagAnswer, SourceLink


logger = logging.getLogger(__name__)

UNAVAILABLE = 'Tìm trên web tạm thời không dùng được. Bạn tắt "Tìm trên web" hoặc thử lại sau nhé.'
# Dấu trích dẫn nội bộ của browser_search, ví dụ 【1†L9-L13】: người dùng không đọc được.
_CITATION = re.compile(r"[ \t]*【[^】]*】")
# Model không mở trang nào: hiện vài kết quả tìm kiếm đầu thay vì cả 10.
_FALLBACK_SOURCES = 3


def web_sources(executed_tools: list[dict] | None) -> list[SourceLink]:
    """Các trang model đã mở đọc (browser.open), tiêu đề lấy từ kết quả tìm kiếm."""
    titles: dict[str, str] = {}
    searched: list[str] = []
    opened: list[str] = []
    for tool in executed_tools or []:
        kind = tool.get("type") or ""
        for item in (tool.get("search_results") or {}).get("results") or []:
            url = item.get("url") or ""
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            if kind == "browser_search":
                titles.setdefault(url, (item.get("title") or "").strip())
                searched.append(url)
            elif kind == "browser.open":
                opened.append(url)
    chosen = list(dict.fromkeys(opened)) or list(dict.fromkeys(searched))[:_FALLBACK_SOURCES]
    return [
        SourceLink(label=titles.get(url) or urlparse(url).netloc.removeprefix("www."), url=url)
        for url in chosen
    ]


def clean_answer(text: str) -> str:
    text = _CITATION.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _skipped_tool(exc: GroqError) -> bool:
    """400 `tool_use_failed`: tool_choice="required" nhưng model trả lời luôn, không tìm web."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        return isinstance(error, dict) and error.get("code") == "tool_use_failed"
    return "tool_use_failed" in str(exc)


class GroqWebSearch:
    name = "groq_web_search"

    def __init__(self, *, api_key: str, model: str, client: AsyncGroq | None = None):
        self.model = model
        # Test truyền client giả; chạy thật dùng AsyncGroq.
        self.client = client or AsyncGroq(api_key=api_key)

    async def _complete(self, messages: list[dict], *, tool_choice: str):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[{"type": "browser_search"}],
            tool_choice=tool_choice,
            reasoning_effort="low",
            temperature=0.2,
            max_tokens=2000,
        )

    async def answer(self, *, question: str, history: list[dict], detector_context: str = "") -> RagAnswer:
        """`history`: các message {role, content} gần nhất, cũ trước mới sau."""
        user = f"{question}\n\n{detector_context}" if detector_context else question
        messages = [
            {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user},
        ]
        try:
            response = await self._complete(messages, tool_choice="required")
        except GroqError as exc:
            if not _skipped_tool(exc):
                logger.warning("Groq web search failed: %s", exc)
                raise AnswerBackendError(503, UNAVAILABLE) from exc
            # Đo 25/09: thỉnh thoảng model bỏ qua tool dù "required" và Groq trả 400. Thử lại một
            # lần với "auto" (prompt vẫn yêu cầu tìm web) thay vì báo lỗi cho người dùng.
            logger.info("Model skipped browser_search; retrying with tool_choice=auto")
            try:
                response = await self._complete(messages, tool_choice="auto")
            except GroqError as retry_exc:
                logger.warning("Groq web search failed: %s", retry_exc)
                raise AnswerBackendError(503, UNAVAILABLE) from retry_exc

        message = response.choices[0].message.model_dump()
        answer = clean_answer(message.get("content") or "")
        if not answer:
            logger.warning("Groq web search returned no content: finish=%s", response.choices[0].finish_reason)
            raise AnswerBackendError(503, UNAVAILABLE)

        sources = web_sources(message.get("executed_tools"))
        logger.info("Web search answer sources=%s", [link.url for link in sources])
        return RagAnswer(
            answer=answer,
            web_sources=sources,
            grounded=bool(sources),
            scope_status="web_search",
            llm_provider="groq",
            llm_model=self.model,
        )
