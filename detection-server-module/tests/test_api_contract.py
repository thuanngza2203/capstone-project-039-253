"""Mục 5 (P2) của plan: Swagger đầy đủ, /admin có xác thực, MAX_HISTORY_TURNS có tác dụng."""

import asyncio
import base64

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.config import get_settings
from app.schemas import ChatTurn, Intent, ReviewItem


@pytest.fixture
def client():
    from app.api import app

    # Không dùng `with`: không chạy lifespan (tải checkpoint) trong test.
    return TestClient(app)


@pytest.fixture
def admin_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret")
    get_settings.cache_clear()
    yield "s3cret"
    get_settings.cache_clear()


def basic(user, password):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_every_api_route_has_response_model_and_tag():
    from app.api import app

    routes = [r for r in app.routes if isinstance(r, APIRoute) and r.include_in_schema]
    assert routes
    missing = [r.path for r in routes if r.response_model is None or not r.tags]
    assert missing == []


def test_admin_requires_login_when_password_is_set(client, admin_password, monkeypatch):
    # Không phụ thuộc ANSWER_BACKEND trong .env của máy chạy test.
    monkeypatch.setenv("ANSWER_BACKEND", "groq")
    get_settings.cache_clear()
    assert client.get("/admin/status").status_code == 401
    assert client.get("/admin/status", headers=basic("admin", "sai")).status_code == 401
    assert client.get("/admin").status_code == 401
    response = client.get("/admin/status", headers=basic("admin", admin_password))
    assert response.status_code == 200
    assert response.json() == {"answer_backend": "groq", "feedback_examples_used": True}
    assert client.get("/admin", headers=basic("admin", admin_password)).status_code == 200
    # Trang chat và health không bị khóa.
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_web_login_returns_token_for_admin_api(client, admin_password):
    wrong = client.post("/admin/login", json={"username": "admin", "password": "sai"})
    assert wrong.status_code == 401
    assert "www-authenticate" not in wrong.headers  # không bật hộp đăng nhập của trình duyệt

    session = client.post("/admin/login", json={"username": "admin", "password": admin_password}).json()
    bearer = {"Authorization": f"Bearer {session['token']}"}
    assert client.get("/admin/status", headers=bearer).status_code == 200

    forged = client.get("/admin/status", headers={"Authorization": f"Bearer {session['token']}x"})
    assert forged.status_code == 401
    assert forged.headers["www-authenticate"] == "Bearer"
    # Trang admin cũ vẫn dùng HTTP Basic.
    assert client.get("/admin/status", headers=basic("admin", admin_password)).status_code == 200


def test_token_expires_and_dies_when_password_changes(admin_password, monkeypatch):
    from app.security import issue_admin_token, verify_admin_token

    settings = get_settings()
    token, expires = issue_admin_token(settings, now=1_000)
    assert expires == 1_000 + 12 * 3600
    assert verify_admin_token(token, settings, now=1_001)
    assert not verify_admin_token(token, settings, now=expires)
    monkeypatch.setenv("ADMIN_PASSWORD", "mat-khau-moi")
    get_settings.cache_clear()
    assert not verify_admin_token(token, get_settings(), now=1_001)


def test_web_login_needs_admin_password(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    get_settings.cache_clear()
    try:
        response = client.post("/admin/login", json={"username": "admin", "password": ""})
    finally:
        get_settings.cache_clear()
    assert response.status_code == 503
    assert "ADMIN_PASSWORD" in response.json()["detail"]


@pytest.fixture
def rag_backend(monkeypatch):
    monkeypatch.setenv("ANSWER_BACKEND", "rag")
    monkeypatch.setenv("CHAT_LLM_PROVIDERS", "gemini, vllm")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_models_lists_choices_only_for_rag_backend(client, rag_backend, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MODEL", "openai/gpt-oss-120b")
    get_settings.cache_clear()
    assert client.get("/api/models").json() == {
        "answer_backend": "rag", "providers": ["gemini", "vllm"], "default": "gemini", "web_search": True}
    monkeypatch.setenv("ANSWER_BACKEND", "groq")
    monkeypatch.setenv("WEB_SEARCH_MODEL", "")
    get_settings.cache_clear()
    assert client.get("/api/models").json() == {
        "answer_backend": "groq", "providers": [], "default": None, "web_search": False}


@pytest.mark.parametrize("model", ["openai/gpt-oss-120b", ""])
def test_chat_web_search_skips_model_choice(client, rag_backend, monkeypatch, model):
    from app.api import app, get_chat_service

    monkeypatch.setenv("WEB_SEARCH_MODEL", model)
    get_settings.cache_clear()
    seen = {}

    class Service:
        async def chat(self, **kwargs):
            seen.update(kwargs)  # chỉ kiểm tra tham số route truyền vào
            raise RuntimeError("dừng ở đây")

    app.dependency_overrides[get_chat_service] = lambda: Service()
    try:
        response = client.post("/api/chat", data={
            "session_id": "s1", "message": "x", "web_search": "true", "llm_provider": "ollama"})
    finally:
        app.dependency_overrides.clear()
    if model:
        # Model đã chọn (kể cả không hợp lệ) bị bỏ qua: tìm web luôn trả lời bằng Groq.
        assert seen["web_search"] is True and seen["llm_provider"] is None
    else:
        assert response.status_code == 400 and "Tìm trên web" in response.json()["detail"]
        assert seen == {}


def test_unknown_provider_in_config_fails_at_startup():
    from app.config import Settings

    with pytest.raises(ValueError, match="CHAT_LLM_PROVIDERS"):
        Settings(groq_api_key="x", chat_llm_providers="vllm,gpt", _env_file=None)


@pytest.mark.parametrize(("sent", "status", "used"), [
    ("vllm", 200, "vllm"), ("", 200, "gemini"), ("ollama", 400, None),
])
def test_chat_passes_the_chosen_model(client, rag_backend, sent, status, used):
    from app.api import app, get_chat_service
    from test_pipeline import Harness, analysis

    harness = Harness()
    harness.normalizer.next = analysis(plant="apple", disease="apple_scab")
    app.dependency_overrides[get_chat_service] = lambda: harness.service
    try:
        response = client.post("/api/chat", data={"session_id": "s1", "message": "ghẻ táo?", "llm_provider": sent})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == status
    if used:
        assert harness.payload()["llm_provider"] == used
    else:
        assert harness.backend.requests == []


def test_admin_is_open_without_password(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    monkeypatch.delenv("ANSWER_BACKEND", raising=False)
    get_settings.cache_clear()
    try:
        assert client.get("/admin/status").status_code == 200
    finally:
        get_settings.cache_clear()


def test_review_item_keeps_mongo_id_and_extra_fields():
    raw = {"_id": "abc", "question": "q", "answer": "a", "image_filename": "la.jpg",
           "metadata": {"sources": ["apple/apple_scab.txt"]}}
    dumped = ReviewItem.model_validate(raw).model_dump(by_alias=True)
    assert dumped["_id"] == "abc"
    assert dumped["image_filename"] == "la.jpg"
    assert dumped["metadata"]["sources"] == ["apple/apple_scab.txt"]


def test_max_history_turns_limits_llm_history():
    from app.chat.session import InMemorySessionStore
    from test_pipeline import Harness, analysis

    harness = Harness()
    harness.service.sessions = harness.sessions = InMemorySessionStore(max_turns=2)
    seen = {}
    original = harness.normalizer.analyze

    async def capture(raw_query, session_context):
        seen["context"] = session_context
        return await original(raw_query, session_context)

    harness.normalizer.analyze = capture
    for text in ["câu 1", "đáp 1", "câu 2", "đáp 2"]:
        role = "user" if text.startswith("câu") else "assistant"
        asyncio.run(harness.sessions.add_turn("s1", ChatTurn(role=role, content=text)))

    harness.chat("hỏi tiếp", analysis(plant="apple", disease="apple_scab", intent=Intent.TREATMENT))
    assert "câu 2" in seen["context"] and "đáp 2" in seen["context"]
    assert "câu 1" not in seen["context"]


def test_chat_route_returns_400_with_message_for_invalid_image(client):
    from app.api import app, get_chat_service
    from app.plant_ai.pipeline.prediction import NOT_A_LEAF, InvalidImageError

    class Service:
        async def chat(self, **kwargs):
            raise InvalidImageError(NOT_A_LEAF)

    app.dependency_overrides[get_chat_service] = lambda: Service()
    try:
        response = client.post(
            "/api/chat",
            data={"session_id": "s1", "message": ""},
            files={"image": ("meo.jpg", b"fake", "image/jpeg")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {"detail": NOT_A_LEAF}


def test_reviews_filter_defaults_to_rated_and_can_include_all():
    from datetime import datetime, timezone

    from app.routes.admin import review_filter

    assert review_filter() == {"user_feedback.rating": {"$in": ["like", "unlike"]}}
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert review_filter(include_all=True, since=since) == {"created_at": {"$gte": since}}


def test_reviews_accepts_all_from_to_params():
    from app.api import app

    params = {p["name"] for p in app.openapi()["paths"]["/admin/reviews"]["get"]["parameters"]}
    assert {"all", "from", "to", "limit"} <= params
