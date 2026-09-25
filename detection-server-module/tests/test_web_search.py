"""Nút "Tìm trên web": Groq browser_search trả lời thẳng, không qua normalizer/router/RAG."""

import asyncio
from types import SimpleNamespace

import groq
import httpx
import pytest

from app.answer.base import AnswerBackendError
from app.answer.web_search import UNAVAILABLE, GroqWebSearch, clean_answer, web_sources
from app.schemas import Action, Intent

# Rút gọn từ phản hồi thật của Groq (openai/gpt-oss-120b + browser_search), 25/09/2026.
EXECUTED_TOOLS = [
    {"type": "browser_search", "search_results": {"results": [
        {"title": "Bệnh úa sớm cà chua (Alternaria solani)", "url": "https://thuocbvtv.com/benh-ua-som/"},
        {"title": "Tomato early blight", "url": "https://extension.umn.edu/early-blight"},
        {"title": "Trang khác", "url": "https://example.com/a"},
        {"title": "Trang khác 2", "url": "https://example.com/b"},
    ]}},
    {"type": "browser.open", "search_results": {"results": [
        {"title": "extension.umn.edu - viewing lines [0 - 125]", "url": "https://extension.umn.edu/early-blight"},
    ]}},
]


class FakeCompletions:
    def __init__(self, content="Do nấm **Alternaria solani**【1†L9-L13】【1†L40-L46】.", tools=EXECUTED_TOOLS, error=None):
        self.content, self.tools, self.error = content, tools, error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(model_dump=lambda: {"content": self.content, "executed_tools": self.tools})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def searcher(**kwargs) -> tuple[GroqWebSearch, FakeCompletions]:
    completions = FakeCompletions(**kwargs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return GroqWebSearch(api_key="x", model="openai/gpt-oss-120b", client=client), completions


def test_sources_are_opened_pages_with_search_titles():
    assert [(link.label, link.url) for link in web_sources(EXECUTED_TOOLS)] == [
        ("Tomato early blight", "https://extension.umn.edu/early-blight"),
    ]
    # Không mở trang nào: 3 kết quả tìm kiếm đầu.
    assert len(web_sources(EXECUTED_TOOLS[:1])) == 3


def test_citation_markers_are_removed():
    assert clean_answer("Do nấm Alternaria【1†L9-L13】【2†L1-L4】.\n\n\n\nHết.") == "Do nấm Alternaria.\n\nHết."


def test_request_uses_browser_search_and_carries_history_and_detection():
    search, completions = searcher()
    history = [{"role": "user", "content": "Lá cà chua có đốm"}, {"role": "assistant", "content": "Có thể là cháy sớm."}]
    result = asyncio.run(search.answer(question="Do tác nhân nào?", history=history,
                                       detector_context="Kết quả nhận diện ảnh lượt này: plant=Tomato"))
    [call] = completions.calls
    assert call["tools"] == [{"type": "browser_search"}] and call["tool_choice"] == "required"
    assert call["messages"][1:3] == history
    assert call["messages"][-1]["content"].endswith("Kết quả nhận diện ảnh lượt này: plant=Tomato")
    assert result.answer == "Do nấm **Alternaria solani**."
    assert result.scope_status == "web_search" and result.llm_provider == "groq"
    assert result.web_sources[0].url == "https://extension.umn.edu/early-blight"


def tool_use_failed() -> groq.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "Tool choice is required, but model did not call a tool", "code": "tool_use_failed"}}
    return groq.BadRequestError("Error code: 400", response=httpx.Response(400, request=request), body=body)


def test_model_that_skips_the_tool_is_retried_with_auto():
    """Đo 25/09: Groq trả 400 tool_use_failed khi model trả lời luôn dù tool_choice=required."""
    search, completions = searcher()
    original = completions.create

    async def flaky(**kwargs):
        if kwargs["tool_choice"] == "required":
            completions.calls.append(kwargs)
            raise tool_use_failed()
        return await original(**kwargs)

    completions.create = flaky
    result = asyncio.run(search.answer(question="x", history=[]))
    assert [call["tool_choice"] for call in completions.calls] == ["required", "auto"]
    assert result.answer == "Do nấm **Alternaria solani**."


@pytest.mark.parametrize("kwargs", [
    {"error": groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))},
    {"content": "【1†L1-L2】"},
])
def test_groq_failure_or_empty_answer_is_503(kwargs):
    search, _ = searcher(**kwargs)
    with pytest.raises(AnswerBackendError) as error:
        asyncio.run(search.answer(question="x", history=[]))
    assert error.value.status_code == 503 and error.value.detail == UNAVAILABLE


def test_pipeline_web_search_skips_normalizer_router_and_rag():
    from test_pipeline import APPLE_SCAB, Harness, analysis

    h = Harness()
    search, completions = searcher()
    h.service.web_search = search

    async def must_not_normalize(raw_query, session_context):
        raise AssertionError("Tìm trên web không gọi normalizer")

    # Lượt 1 bình thường (có ảnh), lượt 2 tìm web: dùng kết quả ảnh gần nhất làm ngữ cảnh.
    h.chat("Lá táo bị gì?", analysis(plant="apple", disease="apple_scab", intent=Intent.TREATMENT), image=b"img")
    h.normalizer.analyze = must_not_normalize
    response = asyncio.run(h.service.chat(session_id="s1", raw_query="Bệnh này có lây không?", web_search=True))

    assert len(h.backend.requests) == 1  # RAG chỉ được gọi ở lượt 1
    assert response.action == Action.WEB_SEARCH
    assert response.web_sources[0].url == "https://extension.umn.edu/early-blight"
    assert response.debug.web_search is True and response.debug.intent is None
    assert "ảnh gần nhất trong cuộc trò chuyện" in completions.calls[0]["messages"][-1]["content"]
    assert APPLE_SCAB.disease in completions.calls[0]["messages"][-1]["content"]
    feedback = h.feedback[-1]["metadata"]
    assert feedback["action"] == "WEB_SEARCH" and feedback["grounded"] is None
    assert feedback["web_sources"] == ["https://extension.umn.edu/early-blight"]
    turns = asyncio.run(h.sessions.snapshot("s1")).turns
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[-1].debug["web_sources"][0]["label"] == "Tomato early blight"


def test_pipeline_web_search_error_saves_nothing():
    from test_pipeline import Harness

    h = Harness()
    h.service.web_search, _ = searcher(error=groq.APIConnectionError(request=httpx.Request("POST", "https://x")))
    with pytest.raises(AnswerBackendError):
        asyncio.run(h.service.chat(session_id="s1", raw_query="Giá phân bón?", web_search=True))
    assert asyncio.run(h.sessions.snapshot("s1")).turns == [] and h.feedback == []
