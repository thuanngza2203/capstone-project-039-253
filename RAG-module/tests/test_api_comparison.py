"""API cho thí nghiệm: chọn index/LLM theo request, `meta`, trạng thái index, probe LLM.

Không mạng, không model: store và LLM giả, server LLM giả qua httpx.MockTransport.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from index_manifest import manifest_path
from server.app import create_app
from server.runtime import RAGRuntime
from test_api_server import AUTH, DOCS, KEY, FilterStore, RecordingLLM, doc

COLLECTION = "plant_disease_vi"

RECURSIVE_DOCS = [doc("Chunk recursive: xử lý bệnh ghẻ táo.", "apple/apple_scab.txt", "apple", "")]


def make_client(runtime: RAGRuntime) -> TestClient:
    return TestClient(create_app(runtime, api_key=KEY))


def write_manifest(directory: Path, *, strategy: str, corpus: str, status: str = "ready") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path(directory, COLLECTION).write_text(json.dumps({
        "collection": COLLECTION, "status": status, "chunking": {"strategy": strategy},
        "corpus_sha256": corpus, "chunk_count": 7, "created_at": "2026-09-24T00:00:00+00:00",
        "embedding_model": "AITeamVN/Vietnamese_Embedding",
    }), encoding="utf-8")


@pytest.fixture
def two_indexes() -> RAGRuntime:
    return RAGRuntime(
        stores={"recursive": FilterStore(RECURSIVE_DOCS), "structure": FilterStore(DOCS)},
        llm=RecordingLLM().runnable,
    )


# --- Chọn index ---------------------------------------------------------------

def test_request_picks_index_and_meta_says_which(two_indexes: RAGRuntime) -> None:
    client = make_client(two_indexes)
    body = {"query": "xử lý ghẻ táo"}
    default = client.post("/v1/retrieve", headers=AUTH, json=body).json()
    assert default["meta"]["index"] == "recursive"  # conftest: CHUNKING_STRATEGY=recursive
    assert [c["content"] for c in default["chunks"]] == ["Chunk recursive: xử lý bệnh ghẻ táo."]

    structure = client.post("/v1/retrieve", headers=AUTH, json={**body, "index": "structure"}).json()
    assert structure["meta"]["index"] == "structure"
    assert structure["meta"]["chunking_strategy"] == "structure"
    assert len(structure["chunks"]) == 4


def test_unknown_index_name_is_422(two_indexes: RAGRuntime) -> None:
    response = make_client(two_indexes).post(
        "/v1/answer", headers=AUTH, json={"query": "x", "index": "semantic"})
    assert response.status_code == 422


def test_index_folder_of_other_strategy_is_refused_before_loading(tmp_path: Path) -> None:
    """CHROMA_DIR trỏ nhầm: hỏi recursive mà thư mục là structure thì báo lỗi, không so sánh nhầm."""
    wrong = tmp_path / "structure_db"
    write_manifest(wrong, strategy="structure", corpus="x")
    runtime = RAGRuntime(index_directories={"recursive": wrong})
    response = make_client(runtime).post(
        "/v1/retrieve", headers=AUTH, json={"query": "x", "index": "recursive"})
    assert response.status_code == 503
    assert "chứa index structure, không phải recursive" in response.json()["detail"]


# --- Chọn LLM và meta ---------------------------------------------------------

def test_request_picks_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    vllm, gemini = RecordingLLM("Từ vLLM [Nguồn 1]."), RecordingLLM("Từ Gemini [Nguồn 1].")
    runtime = RAGRuntime(vector_store=FilterStore(DOCS), llms={"vllm": vllm.runnable, "gemini": gemini.runnable})
    client = make_client(runtime)

    default = client.post("/v1/answer", headers=AUTH, json={"query": "xử lý ghẻ táo"}).json()
    assert default["answer"] == "Từ vLLM [Nguồn 1]." and default["meta"]["llm"]["provider"] == "vllm"

    other = client.post("/v1/answer", headers=AUTH,
                        json={"query": "xử lý ghẻ táo", "llm_provider": "gemini"}).json()
    assert other["answer"] == "Từ Gemini [Nguồn 1]." and other["meta"]["llm"]["provider"] == "gemini"
    assert len(vllm.prompts) == len(gemini.prompts) == 1


def test_meta_records_what_the_llm_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "trong-env")
    reply = AIMessage(
        content="Thu gom lá bệnh [Nguồn 1]",
        response_metadata={"model_name": "rag-llm", "finish_reason": "length"},
        usage_metadata={"input_tokens": 1200, "output_tokens": 800, "total_tokens": 2000},
    )
    runtime = RAGRuntime(vector_store=FilterStore(DOCS), llm=GenericFakeChatModel(messages=iter([reply])))
    meta = make_client(runtime).post("/v1/answer", headers=AUTH, json={"query": "xử lý ghẻ táo"}).json()["meta"]
    assert meta["llm"] == {
        "provider": "vllm", "model": "rag-llm", "finish_reason": "length", "truncated": True,
        "input_tokens": 1200, "output_tokens": 800,
    }
    assert meta["retrieval_mode"] == "hybrid" and meta["top_k"] == 4
    timing = meta["timing_ms"]
    assert timing["rewrite"] is None
    assert all(isinstance(timing[key], int) for key in ("retrieve", "generate", "total"))


def test_refusal_meta_has_no_llm_and_no_retrieval(two_indexes: RAGRuntime) -> None:
    body = make_client(two_indexes).post("/v1/answer", headers=AUTH, json={
        "query": "Cháy muộn khoai tây xử lý sao?", "plant_type": "potato", "disease": "late_blight",
        "index": "structure",
    }).json()
    assert body["grounded"] is False
    meta = body["meta"]
    assert meta["index"] == "structure" and meta["llm"] is None
    assert meta["retrieval_mode"] is None and meta["timing_ms"]["retrieve"] is None


# --- Trạng thái index ---------------------------------------------------------

def test_status_lists_indexes_and_flags_stale_or_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_embeddings,
) -> None:
    monkeypatch.setenv("CHUNKING_STRATEGY", "structure")
    fingerprint = RAGRuntime(vector_store=FilterStore(DOCS)).data_fingerprint()
    assert fingerprint  # data/ thật của repo đọc được

    fresh, legacy = tmp_path / "structure_db", tmp_path / "recursive_db"
    write_manifest(fresh, strategy="structure", corpus=fingerprint)
    legacy.mkdir()
    (legacy / "chroma.sqlite3").write_bytes(b"")
    runtime = RAGRuntime(embeddings=fake_embeddings,
                         index_directories={"structure": fresh, "recursive": legacy})
    status = make_client(runtime).get("/v1/status", headers=AUTH).json()
    indexes = {i["name"]: i for i in status["indexes"]}

    assert status["default_index"] == "structure"
    assert indexes["structure"]["default"] is True and indexes["structure"]["matches_data"] is True
    assert indexes["structure"]["chunk_count"] == 7
    # Index legacy không mặc định: chưa mở, chỉ báo thiếu manifest.
    assert indexes["recursive"]["default"] is False and indexes["recursive"]["loaded"] is False
    assert indexes["recursive"]["has_manifest"] is False and indexes["recursive"]["matches_data"] is None
    assert "chưa có manifest" in indexes["recursive"]["detail"]

    # Build từ data/ trước khi đổi tên file (đúng tình trạng chroma_db ngày 24/09).
    write_manifest(legacy, strategy="recursive", corpus="du-lieu-cu")
    stale = make_client(runtime).get("/v1/status", headers=AUTH).json()["indexes"]
    stale = next(i for i in stale if i["name"] == "recursive")
    assert stale["matches_data"] is False and "data/ đã đổi" in stale["detail"]


# --- Probe LLM ----------------------------------------------------------------

def vllm_env(monkeypatch: pytest.MonkeyPatch, model: str = "rag-llm") -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_PORT", "8001")
    monkeypatch.setenv("VLLM_MODEL", model)
    monkeypatch.setenv("VLLM_API_KEY", "khoa-bi-mat")


def runtime_with_llm_server(handler) -> RAGRuntime:
    return RAGRuntime(vector_store=FilterStore(DOCS),
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_llm_info_without_probe_does_not_call_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("không được gọi mạng khi probe=false")

    response = make_client(runtime_with_llm_server(handler)).get("/v1/llm", headers=AUTH)
    data = response.json()
    assert data["provider"] == "vllm" and data["default"] is True
    assert data["endpoint"] == "http://127.0.0.1:8001/v1" and data["served"] is None
    assert "khoa-bi-mat" not in response.text


def test_probe_reveals_real_model_behind_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_env(monkeypatch)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "rag-llm", "object": "model", "root": "Qwen/Qwen3-32B-FP8", "max_model_len": 32768},
        ]})

    response = make_client(runtime_with_llm_server(handler)).get("/v1/llm?probe=true", headers=AUTH)
    data = response.json()
    assert seen == {"url": "http://127.0.0.1:8001/v1/models", "auth": "Bearer khoa-bi-mat"}
    assert data["reachable"] is True and data["detail"] is None
    assert data["served"][0]["root"] == "Qwen/Qwen3-32B-FP8"
    assert "khoa-bi-mat" not in response.text


def test_probe_warns_when_vllm_model_does_not_match(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_env(monkeypatch, model="Qwen/Qwen3-0.6B")  # đúng dòng cũ còn sót trong .env

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "rag-llm", "root": "Qwen/Qwen3.5-4B"}]})

    data = make_client(runtime_with_llm_server(handler)).get("/v1/llm?probe=true", headers=AUTH).json()
    assert data["reachable"] is True
    assert "không khớp served-model-name" in data["detail"] and "rag-llm" in data["detail"]


def test_probe_reports_unreachable_or_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    response = make_client(runtime_with_llm_server(handler)).get("/v1/llm?probe=true", headers=AUTH)
    data = response.json()
    assert response.status_code == 200
    assert data["reachable"] is False and "401" in data["detail"]
    assert "khoa-bi-mat" not in response.text


def test_probe_ollama_returns_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "qwen3.5:4b", "digest": "sha256:0123456789abcdef",
             "details": {"parameter_size": "4.0B", "quantization_level": "Q4_K_M"}},
            {"name": "llama3:8b", "digest": "ffff"},
        ]})

    data = make_client(runtime_with_llm_server(handler)).get(
        "/v1/llm?provider=ollama&probe=true", headers=AUTH).json()
    assert data["served"] == [{"id": "qwen3.5:4b", "root": None, "max_model_len": None,
                               "digest": "0123456789ab", "details": "4.0B Q4_K_M"}]


def test_openapi_documents_comparison_fields(two_indexes: RAGRuntime) -> None:
    schema = make_client(two_indexes).get("/openapi.json").json()
    fields = schema["components"]["schemas"]["AnswerRequest"]["properties"]
    assert {"index", "llm_provider"} <= set(fields)
    assert "/v1/llm" in schema["paths"]
    assert "meta" in schema["components"]["schemas"]["AnswerResponse"]["required"]
