import asyncio
import json

import httpx
import pytest

from app.answer.base import AnswerBackendError
from app.answer.rag_http import LLM_DOWN, LLM_NOT_CONFIGURED, NOT_READY, RagHttpBackend, strip_citations
from app.schemas import RagAnswerRequest

REQUEST = RagAnswerRequest(
    query="Bệnh ghẻ táo xử lý thế nào?",
    retrieval_query="Cách điều trị bệnh apple_scab trên cây apple",
    plant_type="apple",
    disease="apple_scab",
)

OK_BODY = {
    "answer": "Thu gom lá bệnh [Nguồn 1].",
    "sources": ["apple/apple_scab.txt"],
    "grounded": True,
    "retrieval_query": "Cách điều trị bệnh apple_scab trên cây apple",
    "scope": {"status": "document", "searchable": True, "message": "..."},
    "citations": {"cited": [1], "invalid": []},
}


def call(handler, api_key="secret", request=REQUEST):
    backend = RagHttpBackend(
        base_url="http://rag.test", api_key=api_key, timeout=3,
        transport=httpx.MockTransport(handler),
    )
    return asyncio.run(backend.answer(request))


def test_sends_bearer_key_and_exact_fields():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    result = call(handler)
    assert seen["url"] == "http://rag.test/v1/answer"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"] == {
        "query": "Bệnh ghẻ táo xử lý thế nào?",
        "retrieval_query": "Cách điều trị bệnh apple_scab trên cây apple",
        "plant_type": "apple",
        "disease": "apple_scab",
        "history": [],
    }
    assert result.answer == "Thu gom lá bệnh."  # bỏ nhãn [Nguồn 1]
    assert result.sources == ["apple/apple_scab.txt"]
    assert result.grounded is True
    assert result.scope_status == "document"
    # RAG cũ (trước 1.4.0) không có documents/meta: vẫn đọc được.
    assert result.documents == [] and result.llm_provider is None


def test_selected_model_is_sent_and_answering_model_is_read():
    seen = {}
    body = {
        **OK_BODY,
        "documents": [{"source": "apple/apple_scab.txt", "title": "Bệnh ghẻ táo",
                       "links": [{"label": "UMN Extension", "url": "https://extension.umn.edu/x"}]}],
        "meta": {"llm": {"provider": "gemini", "model": "gemini-2.5-flash"}},
    }

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    result = call(handler, request=REQUEST.model_copy(update={"llm_provider": "gemini"}))
    assert seen["body"]["llm_provider"] == "gemini"
    assert result.llm_provider == "gemini" and result.llm_model == "gemini-2.5-flash"
    assert result.documents[0].title == "Bệnh ghẻ táo"
    assert result.documents[0].links[0].url == "https://extension.umn.edu/x"


class FakeFeedbackRAG:
    def __init__(self, examples=None, error=None):
        self.examples, self.error, self.calls = examples or [], error, []

    async def retrieve(self, *, query, planttype, disease):
        self.calls.append((query, planttype, disease))
        if self.error:
            raise self.error
        return self.examples


def context():
    from app.answer.base import AnswerContext
    from app.schemas import Intent, ResolvedQuery

    return AnswerContext(resolved=ResolvedQuery(plant="apple", disease="apple_scab", intent=Intent.TREATMENT),
                         normalized_query="Bệnh ghẻ táo xử lý thế nào?", recent_history="", detector_context="")


def answer_with(feedback_rag):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    backend = RagHttpBackend(base_url="http://rag.test", timeout=3, transport=httpx.MockTransport(handler),
                             feedback_rag=feedback_rag)
    return asyncio.run(backend.answer(REQUEST, context())), seen["body"]


def test_admin_examples_are_sent_to_rag():
    rag = FakeFeedbackRAG([
        {"question": "Ghẻ táo có cần nhổ cây?", "preferred_answer": "Không cần nhổ.", "similarity": 0.9},
        {"question": "  ", "preferred_answer": "bỏ qua: câu hỏi rỗng"},
        {"question": "Q", "preferred_answer": "x" * 5000},
    ])
    result, body = answer_with(rag)
    assert rag.calls == [("Bệnh ghẻ táo xử lý thế nào?", "apple", "apple_scab")]
    assert body["feedback_examples"] == [
        {"question": "Ghẻ táo có cần nhổ cây?", "answer": "Không cần nhổ."},
        {"question": "Q", "answer": "x" * 3000},
    ]
    assert [example["question"] for example in result.feedback_examples] == ["Ghẻ táo có cần nhổ cây?", "Q"]


@pytest.mark.parametrize("rag", [FakeFeedbackRAG([]), FakeFeedbackRAG(error=RuntimeError("Mongo down"))])
def test_no_examples_or_feedback_failure_keeps_the_payload(rag):
    result, body = answer_with(rag)
    assert "feedback_examples" not in body
    assert result.answer == "Thu gom lá bệnh."


@pytest.mark.parametrize(("raw", "clean"), [
    ("Thu gom lá bệnh [Nguồn 1].", "Thu gom lá bệnh."),
    ("Phun đồng [Nguồn 1], [Nguồn 2] và tỉa cành [Nguồn 3][Nguồn 4].", "Phun đồng và tỉa cành."),
    ("Theo [Nguồn 2: apple/apple_scab.txt] thì nên tỉa.", "Theo thì nên tỉa."),
    ("- Tỉa cành [Nguồn 1, 2]\n- Bón phân [nguồn 3]", "- Tỉa cành\n- Bón phân"),
    ("Nên tỉa cành.\n\n**Nguồn:** [Nguồn 1], [Nguồn 2]", "Nên tỉa cành."),
    ("Nguồn bệnh là lá rụng [Nguồn 1].", "Nguồn bệnh là lá rụng."),
    ("Không có nhãn nào.", "Không có nhãn nào."),
])
def test_strip_citations(raw, clean):
    assert strip_citations(raw) == clean


def test_no_header_without_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=OK_BODY)

    call(handler, api_key="")
    assert seen["auth"] is None


def test_refusal_is_not_an_error():
    body = {**OK_BODY, "answer": "Kho tài liệu hiện tại chưa có thông tin...", "sources": [],
            "grounded": False, "scope": {"status": "unsupported_disease", "searchable": False,
                                         "message": "..."}}
    result = call(lambda request: httpx.Response(200, json=body))
    assert result.grounded is False
    assert result.scope_status == "unsupported_disease"


@pytest.mark.parametrize(("rag_status", "client_status"), [
    (401, 503), (422, 500), (502, 503), (503, 503), (500, 503),
])
def test_error_mapping(rag_status, client_status):
    with pytest.raises(AnswerBackendError) as error:
        call(lambda request: httpx.Response(rag_status, json={"detail": "x"}))
    assert error.value.status_code == client_status
    assert error.value.detail  # thông báo tiếng Việt, không lộ thân lỗi RAG
    assert "x" != error.value.detail


@pytest.mark.parametrize(("status", "detail", "message"), [
    (502, "LLM lỗi hoặc không phản hồi: ConnectError", LLM_DOWN),
    (503, "LLM chưa cấu hình được: Thiếu GEMINI_API_KEY.", LLM_NOT_CONFIGURED),
    (503, "Index chưa sẵn sàng (structure): ...", NOT_READY),
])
def test_llm_errors_suggest_another_model(status, detail, message):
    with pytest.raises(AnswerBackendError) as error:
        call(lambda request: httpx.Response(status, json={"detail": detail}))
    assert error.value.status_code == 503
    assert error.value.detail == message


def test_401_message():
    with pytest.raises(AnswerBackendError) as error:
        call(lambda request: httpx.Response(401, json={"detail": "Thiếu hoặc sai API key."}))
    assert error.value.detail == "Dịch vụ tra cứu chưa cấu hình đúng."


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_unreachable_or_timeout_is_503(exc):
    def handler(request):
        raise exc

    with pytest.raises(AnswerBackendError) as error:
        call(handler)
    assert error.value.status_code == 503
