import asyncio
import json

import httpx
import pytest

from app.answer.base import AnswerBackendError
from app.answer.rag_http import RagHttpBackend
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


def call(handler, api_key="secret"):
    backend = RagHttpBackend(
        base_url="http://rag.test", api_key=api_key, timeout=3,
        transport=httpx.MockTransport(handler),
    )
    return asyncio.run(backend.answer(REQUEST))


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
    assert result.answer == OK_BODY["answer"]
    assert result.sources == ["apple/apple_scab.txt"]
    assert result.grounded is True
    assert result.scope_status == "document"


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
