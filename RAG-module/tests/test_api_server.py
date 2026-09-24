"""RAG API: xác thực, phạm vi, từ chối không gọi LLM, ánh xạ lỗi, Swagger. Không mạng, không model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import rag
from server import __main__ as server_main
from server.app import create_app
from server.runtime import RAGRuntime

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def doc(text: str, source: str, crop: str, heading: str) -> Document:
    return Document(page_content=text,
                    metadata={"source": source, "crop": crop, "heading_path": heading})


DOCS = [
    doc("Xử lý bệnh ghẻ táo: thu gom lá bệnh, phun thuốc gốc đồng.",
        "apple/apple_scab.txt", "apple", "Bệnh ghẻ táo > XỬ LÝ"),
    doc("Triệu chứng bệnh ghẻ táo: đốm xanh ô liu trên lá.",
        "apple/apple_scab.txt", "apple", "Bệnh ghẻ táo > TRIỆU CHỨNG"),
    # Đúng ý định "xử lý", đúng cây táo, SAI bệnh: đúng loại chunk nguy hiểm đo được 21/09.
    doc("Xử lý khi cây đã bị bệnh gỉ sắt: cắt bỏ cành bệnh.",
        "apple/apple_cedar_rust.txt", "apple", "Gỉ sắt táo > XỬ LÝ KHI CÂY ĐÃ BỊ BỆNH"),
    doc("Xử lý cháy sớm khoai tây bằng luân canh.",
        "potato/potato_early_blight.txt", "potato", "Cháy sớm khoai tây > XỬ LÝ"),
    doc("Xử lý mốc sương cà chua: tiêu hủy cây bệnh.",
        "tomato/tomato_late_blight.txt", "tomato", "Mốc sương cà chua > XỬ LÝ"),
]


class FilterStore:
    """Store giả hiểu đúng cú pháp filter của Chroma mà MetadataScope sinh ra."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def _match(self, document: Document, where: dict | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(self._match(document, part) for part in where["$and"])
        (key, condition), = where.items()
        value = document.metadata.get(key)
        return value in condition["$in"] if isinstance(condition, dict) else value == condition

    def similarity_search(self, question: str, *, k: int, filter: dict | None = None):
        return [d for d in self.documents if self._match(d, filter)][:k]

    def get(self, *, include, limit=None):
        records = {"ids": [f"id-{i}" for i in range(len(self.documents))]}
        if include:
            records["documents"] = [d.page_content for d in self.documents]
            records["metadatas"] = [d.metadata for d in self.documents]
        return records


class RecordingLLM:
    def __init__(self, answer: str = "Thu gom lá bệnh [Nguồn 1].",
                 rewrite: str = '{"query": "Bệnh ghẻ táo xử lý thế nào?"}',
                 error: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.answer, self.rewrite, self.error = answer, rewrite, error
        self.runnable = RunnableLambda(self._call)

    def _call(self, prompt_value) -> AIMessage:
        text = prompt_value.to_string()
        self.prompts.append(text)
        if self.error:
            raise self.error
        is_rewrite = "bộ viết lại câu hỏi" in text
        return AIMessage(content=self.rewrite if is_rewrite else self.answer)


@pytest.fixture
def llm() -> RecordingLLM:
    return RecordingLLM()


@pytest.fixture
def client(llm: RecordingLLM) -> TestClient:
    runtime = RAGRuntime(vector_store=FilterStore(DOCS), llm=llm.runnable)
    return TestClient(create_app(runtime, api_key=KEY))


def sources(response) -> set[str]:
    return {chunk["source"] for chunk in response.json()["chunks"]}


# --- Xác thực ---------------------------------------------------------------

def test_health_needs_no_key(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}])
def test_v1_rejects_missing_or_wrong_key(client: TestClient, headers: dict) -> None:
    response = client.post("/v1/retrieve", json={"query": "x"}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Thiếu hoặc sai API key."}


def test_empty_key_disables_auth() -> None:
    open_client = TestClient(create_app(RAGRuntime(vector_store=FilterStore(DOCS)), api_key=""))
    assert open_client.post("/v1/retrieve", json={"query": "ghẻ táo"}).status_code == 200


# --- Retrieve ---------------------------------------------------------------

def test_scope_keeps_out_same_crop_wrong_disease_chunk(client: TestClient) -> None:
    body = {"query": "bệnh ghẻ trên cây táo xử lý như thế nào"}
    unscoped = client.post("/v1/retrieve", json=body, headers=AUTH)
    assert "apple/apple_cedar_rust.txt" in sources(unscoped)

    scoped = client.post("/v1/retrieve", headers=AUTH,
                         json={**body, "plant_type": "apple", "disease": "apple_scab"})
    assert scoped.status_code == 200
    assert sources(scoped) == {"apple/apple_scab.txt"}
    scope = scoped.json()["scope"]
    assert scope["status"] == "document"
    assert scope["sources"] == ["apple/apple_scab.txt"]


def test_old_planttype_field_and_detector_label_are_accepted(client: TestClient) -> None:
    response = client.post("/v1/retrieve", headers=AUTH,
                           json={"query": "xử lý", "planttype": "Apple", "disease": "Scab"})
    assert response.json()["scope"]["disease"] == "apple_scab"
    assert sources(response) == {"apple/apple_scab.txt"}


def test_crop_scope_uses_corpus_folder(client: TestClient) -> None:
    response = client.post("/v1/retrieve", headers=AUTH, json={"query": "xử lý", "plant_type": "tomato"})
    assert sources(response) == {"tomato/tomato_late_blight.txt"}


def test_debug_returns_candidates_and_scope(client: TestClient) -> None:
    response = client.post("/v1/retrieve", headers=AUTH, json={
        "query": "xử lý", "plant_type": "apple", "disease": "apple_scab", "debug": True})
    debug = response.json()["debug"]
    assert debug["scope"] == {"sources": ["apple/apple_scab.txt"], "crop": None}
    assert debug["candidates"]


@pytest.mark.parametrize("body", [
    {"query": "   "},
    {"query": "x", "top_k": 0},
    {"query": "x", "top_k": 21},
    {"query": "x", "mode": "fuzzy"},
    {"query": "x", "plant": "apple"},  # gõ nhầm tên trường thì báo, không lặng lẽ bỏ qua
])
def test_invalid_request_is_422(client: TestClient, body: dict) -> None:
    assert client.post("/v1/retrieve", json=body, headers=AUTH).status_code == 422


# --- Answer -----------------------------------------------------------------

def test_answer_prompt_has_subject_scoped_context_and_history(client: TestClient, llm: RecordingLLM) -> None:
    response = client.post("/v1/answer", headers=AUTH, json={
        "query": "Cách xử lý bệnh ghẻ trên cây táo?",
        "plant_type": "Apple", "disease": "Apple_scab",
        "subject_context": "plant=Apple; disease=Scab; confidence=0.93",
        "history": [{"role": "user", "content": "Lá táo có đốm xanh ô liu"},
                    {"role": "assistant", "content": "Có thể là ghẻ táo."}],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["sources"] == ["apple/apple_scab.txt"]
    assert data["citations"] == {"cited": [1], "invalid": []}
    assert data["chunks"] is None  # chỉ có khi debug

    [prompt] = llm.prompts
    assert rag.SUBJECT_HEADER in prompt
    assert "- Cây: apple" in prompt and "- Bệnh: apple_scab" in prompt
    assert "Kết quả nhận diện ảnh: plant=Apple; disease=Scab; confidence=0.93" in prompt
    assert "Lá táo có đốm xanh ô liu" in prompt
    assert "gỉ sắt" not in prompt  # chunk sai bệnh không lọt vào ngữ cảnh


@pytest.mark.parametrize(("plant", "disease", "status", "phrase"), [
    ("potato", "late_blight", "unsupported_disease", "late_blight"),
    ("apple", "Apple___Weird_Class", "unknown_disease", "chưa có đủ thông tin"),
])
def test_unsearchable_scope_answers_without_retrieval_or_llm(
    client: TestClient, llm: RecordingLLM, plant: str, disease: str, status: str, phrase: str,
) -> None:
    response = client.post("/v1/answer", headers=AUTH,
                           json={"query": "Phòng bệnh thế nào?", "plant_type": plant, "disease": disease})
    data = response.json()
    assert response.status_code == 200
    assert data["grounded"] is False
    assert data["sources"] == []
    assert data["scope"]["status"] == status
    assert data["scope"]["searchable"] is False
    assert phrase in data["answer"]
    assert llm.prompts == []


def test_no_matching_documents_refuses_without_llm(client: TestClient, llm: RecordingLLM) -> None:
    response = client.post("/v1/answer", headers=AUTH, json={"query": "xử lý", "plant_type": "cherry"})
    assert response.json()["grounded"] is False
    assert response.json()["answer"] == rag.NO_CONTEXT_ANSWER
    assert llm.prompts == []


def test_invalid_citation_is_reported(llm: RecordingLLM) -> None:
    llm.answer = "Theo [Nguồn 3] thì cắt tỉa."
    client = TestClient(create_app(RAGRuntime(vector_store=FilterStore(DOCS), llm=llm.runnable), api_key=KEY))
    with pytest.warns(UserWarning, match="dẫn nguồn không có trong NGỮ CẢNH"):
        data = client.post("/v1/answer", headers=AUTH, json={
            "query": "xử lý", "plant_type": "apple", "disease": "apple_scab", "top_k": 2}).json()
    assert data["citations"]["invalid"] == [3]


def test_rewrite_query_runs_only_when_asked(client: TestClient, llm: RecordingLLM) -> None:
    history = [{"role": "user", "content": "Bệnh ghẻ táo là gì?"},
               {"role": "assistant", "content": "Là bệnh nấm."}]
    plain = client.post("/v1/answer", headers=AUTH, json={"query": "Vậy xử lý sao?", "history": history})
    assert plain.json()["retrieval_query"] == "Vậy xử lý sao?"
    assert len(llm.prompts) == 1

    rewritten = client.post("/v1/answer", headers=AUTH,
                            json={"query": "Vậy xử lý sao?", "history": history, "rewrite_query": True})
    assert rewritten.json()["retrieval_query"] == "Bệnh ghẻ táo xử lý thế nào?"
    assert len(llm.prompts) == 3  # rewrite + answer


def test_history_is_trimmed_to_chat_settings(client: TestClient, llm: RecordingLLM,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "1")
    history = [{"role": "user", "content": f"cũ-{i}"} for i in range(4)] + [
        {"role": "user", "content": "gần-nhất-hỏi"}, {"role": "assistant", "content": "gần-nhất-đáp"}]
    client.post("/v1/answer", headers=AUTH, json={"query": "xử lý ghẻ táo", "history": history})
    assert "gần-nhất-hỏi" in llm.prompts[0] and "gần-nhất-đáp" in llm.prompts[0]
    assert "cũ-3" not in llm.prompts[0]


def test_cli_prompt_is_unchanged_without_subject(llm: RecordingLLM) -> None:
    """Server thêm khối ĐỐI TƯỢNG; đường CLI không có đối tượng thì prompt như cũ."""
    rag.ask("Xử lý ghẻ táo?", llm=llm.runnable, vector_store=FilterStore(DOCS), mode="semantic")
    assert rag.SUBJECT_HEADER not in llm.prompts[0]
    assert "Human: NGỮ CẢNH:" in llm.prompts[0]


# --- Lỗi --------------------------------------------------------------------

def test_llm_failure_is_502(llm: RecordingLLM) -> None:
    llm.error = ConnectionError("tunnel đóng")
    client = TestClient(create_app(RAGRuntime(vector_store=FilterStore(DOCS), llm=llm.runnable), api_key=KEY))
    response = client.post("/v1/answer", headers=AUTH, json={"query": "xử lý ghẻ táo"})
    assert response.status_code == 502
    assert "LLM lỗi hoặc không phản hồi" in response.json()["detail"]
    assert "tunnel đóng" in response.json()["detail"]


def test_llm_misconfiguration_is_503_but_retrieve_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_HOST", "http://203.0.113.10")  # lỗi chép địa chỉ hay gặp
    client = TestClient(create_app(RAGRuntime(vector_store=FilterStore(DOCS)), api_key=KEY))
    answer = client.post("/v1/answer", headers=AUTH, json={"query": "xử lý ghẻ táo"})
    assert answer.status_code == 503
    assert "VLLM_HOST" in answer.json()["detail"]
    assert client.post("/v1/retrieve", headers=AUTH, json={"query": "xử lý ghẻ táo"}).status_code == 200


def test_missing_index_is_503_and_status_explains(tmp_path: Path) -> None:
    runtime = RAGRuntime(persist_directory=tmp_path / "chua-index")
    with TestClient(create_app(runtime, api_key=KEY)) as client:  # chạy lifespan: không crash
        assert client.get("/health").status_code == 200
        response = client.post("/v1/retrieve", headers=AUTH, json={"query": "x"})
        assert response.status_code == 503
        assert "Index chưa sẵn sàng" in response.json()["detail"]
        status = client.get("/v1/status", headers=AUTH).json()
        assert status["ready"] is False and "chưa được tạo" in status["detail"]


def test_status_reports_config_without_secrets(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_PORT", "8001")
    monkeypatch.setenv("VLLM_API_KEY", "must-not-leak")
    response = client.get("/v1/status", headers=AUTH)
    data = response.json()
    assert data["ready"] is True and data["chunk_count"] == len(DOCS)
    assert data["llm_provider"] == "vllm"
    assert data["llm_endpoint"] == "http://127.0.0.1:8001/v1"
    assert data["auth_enabled"] is True
    assert "must-not-leak" not in response.text and KEY not in response.text


# --- Danh mục, Swagger, CLI -------------------------------------------------

def test_taxonomy_lists_coverage(client: TestClient) -> None:
    plants = {p["plant"]: p for p in client.get("/v1/taxonomy", headers=AUTH).json()["plants"]}
    assert plants["pepper"]["crop"] == "pepper_bell"
    potato = {d["disease"]: d for d in plants["potato"]["diseases"]}
    assert potato["late_blight"]["has_document"] is False
    assert potato["early_blight"]["source"] == "potato/potato_early_blight.txt"


def test_openapi_documents_auth_errors_and_fields(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "RAG_API_KEY" in schema["components"]["securitySchemes"]
    answer = schema["paths"]["/v1/answer"]["post"]
    assert {"200", "401", "422", "502", "503"} <= set(answer["responses"])
    fields = schema["components"]["schemas"]["AnswerRequest"]["properties"]
    assert {"query", "plant_type", "disease", "history", "subject_context"} <= set(fields)
    assert "security" not in schema["paths"]["/health"]["get"]


def test_cli_warns_but_allows_public_bind_without_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """API public là lựa chọn đã chốt: không key vẫn chạy trên 0.0.0.0, chỉ cảnh báo."""
    import uvicorn

    started = {}
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(kwargs))
    assert server_main.main(["--host", "0.0.0.0"]) == 0
    assert started["host"] == "0.0.0.0"
    assert "RAG_API_KEY trống" in capsys.readouterr().err


def test_cli_exports_openapi(tmp_path: Path) -> None:
    target = tmp_path / "openapi.json"
    assert server_main.main(["--export-openapi", str(target)]) == 0
    assert "/v1/answer" in json.loads(target.read_text(encoding="utf-8"))["paths"]


def test_retrieval_query_searches_while_llm_sees_the_user_question(client: TestClient, llm: RecordingLLM) -> None:
    """Detection gửi câu đã làm rõ để tìm, nhưng LLM phải trả lời câu thật của người dùng."""
    response = client.post("/v1/answer", headers=AUTH, json={
        "query": "Lá táo nhà tôi bị vậy có cần nhổ cây không?",
        "retrieval_query": "Cách điều trị bệnh apple_scab trên cây apple",
        "plant_type": "apple", "disease": "apple_scab",
    })
    assert response.json()["retrieval_query"] == "Cách điều trị bệnh apple_scab trên cây apple"
    prompt = llm.prompts[0]
    assert "CÂU HỎI:\nLá táo nhà tôi bị vậy có cần nhổ cây không?" in prompt
    assert "CÂU HỎI ĐÃ LÀM RÕ:\nCách điều trị bệnh apple_scab trên cây apple" in prompt


def test_retrieval_query_and_rewrite_are_mutually_exclusive(client: TestClient) -> None:
    response = client.post("/v1/answer", headers=AUTH, json={
        "query": "x", "retrieval_query": "y", "rewrite_query": True})
    assert response.status_code == 422
